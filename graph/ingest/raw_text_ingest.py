"""
Raw Text Ingest Pipeline
========================
Reads ``corpus/raw/*.txt`` files in the ``{page}.{line}: text`` format,
creates ``WakeToken`` nodes for each surface token, and creates ``PRECEDES``
edges between adjacent tokens on the same line and ``PRECEDES_LINE`` edges
between the last token of a line and the first token of the next line.

Expected file format
--------------------
Each line in a raw corpus file must begin with a ``{page}.{line}:`` prefix::

    3.1: riverrun, past Eve and Adam's, from swerve of shore to bend
    3.2: of bay, brings us by a commodius vicus of recirculation back to
    3.3: Howth Castle and Environs.

Tokenisation
------------
Tokens are split on whitespace.  Punctuation *attached* to a token is
preserved in the ``surface`` field (e.g. ``"riverrun,"``), matching the
convention used in the McHugh annotation JSON.  This means the same surface
form can be looked up across both datasets without normalisation, and the
graph can carry the ``surface`` property exactly as Joyce wrote it.

Graph writes
------------
For each token ``t`` at position ``pos`` on page ``page``, line ``line``:

* ``MERGE (t:WakeToken {token_id: "{page}_{line}_{pos}"})``
  with properties ``surface``, ``page``, ``line``, ``position``.
* ``MERGE (t)-[:PRECEDES]->(t_next)`` for consecutive tokens on the same line.
* ``MERGE (t_last_of_line_N)-[:PRECEDES_LINE]->(t_first_of_line_N+1)``
  so that sentence-spanning paths work correctly.

Usage
-----
Run directly::

    python -m graph.ingest.raw_text_ingest corpus/raw/ \\
        --neo4j-uri bolt://localhost:7687

Or import and call programmatically::

    from graph.ingest.raw_text_ingest import RawTextIngester
    ingester = RawTextIngester()
    ingester.ingest_directory(Path("corpus/raw"))
    ingester.close()
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from neo4j import GraphDatabase, Driver, Session

logger = logging.getLogger(__name__)

# Regex for the required line prefix: ``{page}.{line}: ``
_LINE_PREFIX_RE = re.compile(r"^(\d+)\.(\d+):\s*(.*)")

# Batch size: number of token-nodes written per transaction
_DEFAULT_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RawToken:
    token_id: str
    surface: str
    page: int
    line: int
    position: int  # 0-based within the line


@dataclass
class LineTokens:
    page: int
    line: int
    tokens: list[RawToken] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_line(raw_line: str) -> tuple[int, int, str] | None:
    """
    Parse a single text line of the corpus.

    Returns ``(page, line_num, text)`` or ``None`` if the line does not
    match the required prefix format.
    """
    raw_line = raw_line.rstrip("\n\r")
    match = _LINE_PREFIX_RE.match(raw_line)
    if not match:
        return None
    page = int(match.group(1))
    line_num = int(match.group(2))
    text = match.group(3)
    return page, line_num, text


def _tokenise(text: str) -> list[str]:
    """
    Split ``text`` on whitespace, returning non-empty tokens.

    Punctuation attached to words (e.g. ``"riverrun,"`` or ``"Adam's"``)
    is preserved in the token surface form — this matches the McHugh JSON
    convention and allows direct key-based cross-referencing.
    """
    return [tok for tok in text.split() if tok]


def _iter_line_tokens(file_path: Path) -> Iterator[LineTokens]:
    """
    Yield a ``LineTokens`` object for every valid line in ``file_path``.

    Lines that do not match the ``{page}.{line}:`` prefix are skipped with
    a warning.
    """
    with open(file_path, encoding="utf-8") as fh:
        for raw in fh:
            parsed = _parse_line(raw)
            if parsed is None:
                stripped = raw.strip()
                if stripped:
                    logger.warning("Skipping malformed line in %s: %r", file_path.name, stripped[:80])
                continue
            page, line_num, text = parsed
            surfaces = _tokenise(text)
            lt = LineTokens(page=page, line=line_num)
            for pos, surface in enumerate(surfaces):
                token_id = f"{page}_{line_num}_{pos}"
                lt.tokens.append(
                    RawToken(
                        token_id=token_id,
                        surface=surface,
                        page=page,
                        line=line_num,
                        position=pos,
                    )
                )
            yield lt


# ---------------------------------------------------------------------------
# Ingester
# ---------------------------------------------------------------------------

class RawTextIngester:
    """
    Reads raw corpus text files and writes ``WakeToken`` nodes and
    ``PRECEDES`` / ``PRECEDES_LINE`` edges to Neo4j.

    Parameters
    ----------
    uri : str
        Neo4j Bolt URI.
    user : str
        Neo4j username.
    password : str
        Neo4j password.
    database : str
        Target database (default ``neo4j``).
    batch_size : int
        Number of tokens committed per transaction (default 500).
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "wakeanamnesis",
        database: str = "neo4j",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.batch_size = batch_size
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("RawTextIngester connected to %s", uri)
        self._ensure_indexes()

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self._driver.close()
        logger.info("RawTextIngester driver closed.")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create WakeToken uniqueness constraint and useful indexes."""
        statements = [
            "CREATE CONSTRAINT wake_token_id IF NOT EXISTS FOR (t:WakeToken) REQUIRE t.token_id IS UNIQUE",
            "CREATE INDEX wake_token_page IF NOT EXISTS FOR (t:WakeToken) ON (t.page)",
            "CREATE INDEX wake_token_page_line IF NOT EXISTS FOR (t:WakeToken) ON (t.page, t.line)",
            "CREATE INDEX wake_token_surface IF NOT EXISTS FOR (t:WakeToken) ON (t.surface)",
        ]
        with self._driver.session(database=self.database) as session:
            for stmt in statements:
                try:
                    session.run(stmt)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("DDL note: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: Path) -> int:
        """
        Ingest all tokens from a single corpus text file.

        Parameters
        ----------
        file_path : Path
            Path to a ``fw_book{N}_chapter{M}.txt`` or ``fw_full.txt`` file.

        Returns
        -------
        int
            Number of token nodes written.
        """
        logger.info("Starting ingest: %s", file_path.name)
        all_lines: list[LineTokens] = list(_iter_line_tokens(file_path))
        total_tokens = sum(len(lt.tokens) for lt in all_lines)
        logger.info("  Parsed %d lines, %d tokens from %s",
                    len(all_lines), total_tokens, file_path.name)

        written = 0
        with self._driver.session(database=self.database) as session:
            # Phase 1: write token nodes in batches
            token_buffer: list[RawToken] = []
            for lt in all_lines:
                token_buffer.extend(lt.tokens)
                while len(token_buffer) >= self.batch_size:
                    batch, token_buffer = token_buffer[:self.batch_size], token_buffer[self.batch_size:]
                    self._write_token_batch(session, batch)
                    written += len(batch)
                    logger.info("  Written %d / %d tokens (%s)",
                                written, total_tokens, file_path.name)
            if token_buffer:
                self._write_token_batch(session, token_buffer)
                written += len(token_buffer)
                logger.info("  Written %d / %d tokens (%s)",
                            written, total_tokens, file_path.name)

            # Phase 2: write PRECEDES and PRECEDES_LINE edges
            logger.info("  Writing sequence edges for %s ...", file_path.name)
            edge_count = self._write_sequence_edges(session, all_lines)
            logger.info("  Wrote %d sequence edges from %s", edge_count, file_path.name)

        logger.info("Ingest complete: %s — %d tokens", file_path.name, written)
        return written

    def ingest_directory(self, directory: Path) -> int:
        """
        Ingest all ``*.txt`` corpus files in ``directory`` (non-recursive).

        Files are processed in alphabetical order.

        Returns
        -------
        int
            Total token nodes written across all files.
        """
        txt_files = sorted(directory.glob("*.txt"))
        if not txt_files:
            logger.warning("No .txt files found in %s", directory)
            return 0
        logger.info("Found %d text files in %s", len(txt_files), directory)
        total = 0
        for f in txt_files:
            total += self.ingest_file(f)
        logger.info("Directory ingest complete: %d total tokens from %s", total, directory)
        return total

    # ------------------------------------------------------------------
    # Private write helpers
    # ------------------------------------------------------------------

    def _write_token_batch(self, session: Session, batch: list[RawToken]) -> None:
        """Write a batch of RawToken nodes using UNWIND for efficiency."""
        params = [
            {
                "token_id": t.token_id,
                "surface": t.surface,
                "page": t.page,
                "line": t.line,
                "position": t.position,
            }
            for t in batch
        ]
        session.run(
            """
            UNWIND $tokens AS tok
            MERGE (t:WakeToken {token_id: tok.token_id})
            ON CREATE SET
                t.surface   = tok.surface,
                t.page      = tok.page,
                t.line      = tok.line,
                t.position  = tok.position
            """,
            tokens=params,
        )

    def _write_sequence_edges(self, session: Session, all_lines: list[LineTokens]) -> int:
        """
        Write PRECEDES and PRECEDES_LINE edges for the given sequence of lines.

        PRECEDES: consecutive tokens within the same line.
        PRECEDES_LINE: last token of line N → first token of line N+1.

        Returns the total number of edges written.
        """
        edge_count = 0
        precedes_pairs: list[dict[str, str]] = []
        precedes_line_pairs: list[dict[str, str]] = []

        prev_line: LineTokens | None = None
        for lt in all_lines:
            tokens = lt.tokens
            # Within-line PRECEDES edges
            for i in range(len(tokens) - 1):
                precedes_pairs.append({
                    "from_id": tokens[i].token_id,
                    "to_id": tokens[i + 1].token_id,
                })
            # Cross-line PRECEDES_LINE edge
            if prev_line and prev_line.tokens and lt.tokens:
                precedes_line_pairs.append({
                    "from_id": prev_line.tokens[-1].token_id,
                    "to_id": lt.tokens[0].token_id,
                })
            prev_line = lt

        # Batch-write PRECEDES
        for i in range(0, len(precedes_pairs), self.batch_size):
            batch = precedes_pairs[i : i + self.batch_size]
            session.run(
                """
                UNWIND $pairs AS pair
                MATCH (a:WakeToken {token_id: pair.from_id})
                MATCH (b:WakeToken {token_id: pair.to_id})
                MERGE (a)-[:PRECEDES]->(b)
                """,
                pairs=batch,
            )
            edge_count += len(batch)

        # Batch-write PRECEDES_LINE
        for i in range(0, len(precedes_line_pairs), self.batch_size):
            batch = precedes_line_pairs[i : i + self.batch_size]
            session.run(
                """
                UNWIND $pairs AS pair
                MATCH (a:WakeToken {token_id: pair.from_id})
                MATCH (b:WakeToken {token_id: pair.to_id})
                MERGE (a)-[:PRECEDES_LINE]->(b)
                """,
                pairs=batch,
            )
            edge_count += len(batch)

        return edge_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raw_text_ingest",
        description=(
            "Ingest Finnegans Wake raw text corpus files into the WAKE Neo4j graph. "
            "Provide a directory of .txt files or a single file."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a single .txt corpus file or directory of .txt files.",
    )
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j Bolt URI (default: bolt://localhost:7687)",
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username (default: neo4j)",
    )
    parser.add_argument(
        "--neo4j-password",
        default="wakeanamnesis",
        help="Neo4j password (default: wakeanamnesis)",
    )
    parser.add_argument(
        "--database",
        default="neo4j",
        help="Neo4j database name (default: neo4j)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=f"Tokens per write transaction (default: {_DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser


if __name__ == "__main__":
    _parser = _build_arg_parser()
    _args = _parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, _args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    _ingester = RawTextIngester(
        uri=_args.neo4j_uri,
        user=_args.neo4j_user,
        password=_args.neo4j_password,
        database=_args.database,
        batch_size=_args.batch_size,
    )

    try:
        _target = _args.path
        if _target.is_dir():
            _total = _ingester.ingest_directory(_target)
        elif _target.is_file():
            _total = _ingester.ingest_file(_target)
        else:
            logger.error("Path does not exist: %s", _target)
            sys.exit(1)
        print(f"Ingest complete: {_total} token nodes written.")
    finally:
        _ingester.close()
