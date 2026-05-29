"""
McHugh Annotation Ingest Pipeline
==================================
Reads JSON annotation files produced from Roland McHugh's
*Annotations to Finnegans Wake* (3rd ed., Johns Hopkins, 2006) and writes
the structured data into the WAKE Neo4j graph.

JSON Schema (one file per page, e.g. ``corpus/mcHugh/page_003.json``):

.. code-block:: json

    {
      "page": 3,
      "annotations": [
        {
          "line": 1,
          "surface": "riverrun",
          "languages": ["en", "it", "non", "la"],
          "roots": [
            {
              "language": "en",
              "form": "river",
              "gloss": "a natural watercourse",
              "confidence": 1.0
            }
          ],
          "semantic_fields": ["water", "flow", "cycle"],
          "vico_cycle": 4,
          "kabbalah_sefirah": null,
          "cross_references": ["Anna Livia Plurabelle", "Liffey"],
          "notes": "Opening word; Vico's ricorso."
        }
      ]
    }

Graph nodes and relationships created:
    - ``(WakeToken)``               — one node per annotated surface form
    - ``(EtymRoot)``                — one node per (language, form) pair
    - ``(WakeToken)-[:HAS_ROOT]->`` — with ``confidence`` property
    - ``(SemanticField)``           — one node per field name (MERGE)
    - ``(WakeToken)-[:ACTIVATES]->``
    - ``(VicoCycle)``               — one node per cycle 1-4 (MERGE)
    - ``(WakeToken)-[:BELONGS_TO_CYCLE]->``
    - ``(KabbalahNode)``            — one node per sefirah (MERGE)
    - ``(WakeToken)-[:RESONATES_WITH]->``
    - ``(Language)``                — one node per ISO code (MERGE)
    - ``(EtymRoot)-[:IN_LANGUAGE]->``
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase, Driver, Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vico cycle metadata (seed data written to graph on first run)
# ---------------------------------------------------------------------------
_VICO_CYCLES: dict[int, dict[str, str]] = {
    1: {
        "name": "Theocratic",
        "description": (
            "The divine age: gods and giants; humanity explains nature through myth "
            "and divine will. Thunderous beginnings, fear of gods, first institutions."
        ),
    },
    2: {
        "name": "Aristocratic",
        "description": (
            "The heroic age: patriarchal heroes and aristocratic rule; "
            "strong families, feudal law, written language, epic poetry."
        ),
    },
    3: {
        "name": "Democratic",
        "description": (
            "The human age: popular government, individual rights, "
            "philosophy, prose, and the full development of reason."
        ),
    },
    4: {
        "name": "Ricorso",
        "description": (
            "The return: dissolution of the democratic age back into barbarism "
            "and a new divine age. The cycle recommences — river running back to source."
        ),
    },
}

_KABBALAH_SEFIROTH: dict[str, dict[str, Any]] = {
    "Kether":    {"number": 1,  "description": "The Crown; infinite light; the unknowable divine source."},
    "Chokmah":   {"number": 2,  "description": "Wisdom; the first flash of divine intellect."},
    "Binah":     {"number": 3,  "description": "Understanding; the great mother; structured comprehension."},
    "Chesed":    {"number": 4,  "description": "Mercy/Loving-kindness; expansive grace."},
    "Gevurah":   {"number": 5,  "description": "Strength/Judgement; severity, power, limitation."},
    "Tiphareth": {"number": 6,  "description": "Beauty/Harmony; the heart of the tree; balance."},
    "Netzach":   {"number": 7,  "description": "Victory/Eternity; emotions, nature, desire."},
    "Hod":       {"number": 8,  "description": "Splendour; intellect, communication, magic."},
    "Yesod":     {"number": 9,  "description": "Foundation; the astral realm; dreams and the unconscious."},
    "Malkuth":   {"number": 10, "description": "The Kingdom; the material world; earth; embodiment."},
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EtymRootRecord:
    language: str
    form: str
    gloss: str
    confidence: float = 1.0

    @property
    def root_id(self) -> str:
        return f"{self.language}_{self.form}"


@dataclass
class AnnotationRecord:
    page: int
    line: int
    surface: str
    languages: list[str]
    roots: list[EtymRootRecord]
    semantic_fields: list[str]
    vico_cycle: int | None
    kabbalah_sefirah: str | None
    cross_references: list[str]
    notes: str

    @property
    def token_id(self) -> str:
        # Position within line is not available from McHugh data alone;
        # use a stable composite key: page_line_surface
        safe_surface = self.surface.replace(" ", "_")
        return f"{self.page}_{self.line}_{safe_surface}"

    @classmethod
    def from_dict(cls, page: int, data: dict[str, Any]) -> "AnnotationRecord":
        roots = [
            EtymRootRecord(
                language=r["language"],
                form=r["form"],
                gloss=r.get("gloss", ""),
                confidence=float(r.get("confidence", 1.0)),
            )
            for r in data.get("roots", [])
        ]
        return cls(
            page=page,
            line=int(data["line"]),
            surface=str(data["surface"]),
            languages=list(data.get("languages", [])),
            roots=roots,
            semantic_fields=list(data.get("semantic_fields", [])),
            vico_cycle=data.get("vico_cycle"),
            kabbalah_sefirah=data.get("kabbalah_sefirah"),
            cross_references=list(data.get("cross_references", [])),
            notes=str(data.get("notes", "")),
        )


# ---------------------------------------------------------------------------
# McHughIngester
# ---------------------------------------------------------------------------

class McHughIngester:
    """
    Stateful ingester that opens a Neo4j driver and provides methods to
    load McHugh annotation JSON files into the WAKE graph.

    Parameters
    ----------
    uri : str
        Neo4j Bolt URI, e.g. ``bolt://localhost:7687``.
    user : str
        Neo4j username.
    password : str
        Neo4j password.
    database : str
        Target database name (default ``neo4j``).
    batch_size : int
        Number of annotation records written per transaction (default 100).
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "wakeanamnesis",
        database: str = "neo4j",
        batch_size: int = 100,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.batch_size = batch_size
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("McHughIngester connected to %s (database=%s)", uri, database)
        self._ensure_indexes()

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self._driver.close()
        logger.info("McHughIngester driver closed.")

    # ------------------------------------------------------------------
    # Schema / indexes
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """
        Create constraints and indexes required by the McHugh ingest if they
        do not already exist.  These supplement the full schema in
        ``graph/schema/wake_schema.cypher``; running that file first is
        preferred, but this method makes the ingester self-sufficient.
        """
        ddl_statements = [
            "CREATE CONSTRAINT wake_token_id IF NOT EXISTS FOR (t:WakeToken) REQUIRE t.token_id IS UNIQUE",
            "CREATE CONSTRAINT etym_root_id IF NOT EXISTS FOR (r:EtymRoot) REQUIRE r.root_id IS UNIQUE",
            "CREATE CONSTRAINT semantic_field_name IF NOT EXISTS FOR (s:SemanticField) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT vico_cycle_number IF NOT EXISTS FOR (v:VicoCycle) REQUIRE v.cycle IS UNIQUE",
            "CREATE CONSTRAINT kabbalah_node_name IF NOT EXISTS FOR (k:KabbalahNode) REQUIRE k.sefirah IS UNIQUE",
            "CREATE CONSTRAINT language_code IF NOT EXISTS FOR (lang:Language) REQUIRE lang.code IS UNIQUE",
            "CREATE INDEX wake_token_page IF NOT EXISTS FOR (t:WakeToken) ON (t.page)",
            "CREATE INDEX wake_token_surface IF NOT EXISTS FOR (t:WakeToken) ON (t.surface)",
            "CREATE INDEX etym_root_language IF NOT EXISTS FOR (r:EtymRoot) ON (r.language)",
        ]
        with self._driver.session(database=self.database) as session:
            for stmt in ddl_statements:
                try:
                    session.run(stmt)
                except Exception as exc:  # noqa: BLE001
                    # Constraint already exists or minor DDL variance — log and continue
                    logger.debug("DDL note: %s — %s", stmt[:60], exc)
        self._seed_reference_nodes()

    def _seed_reference_nodes(self) -> None:
        """Write VicoCycle and KabbalahNode seed data if not present."""
        with self._driver.session(database=self.database) as session:
            for cycle_num, meta in _VICO_CYCLES.items():
                session.run(
                    """
                    MERGE (v:VicoCycle {cycle: $cycle})
                    ON CREATE SET v.name = $name, v.description = $description
                    """,
                    cycle=cycle_num,
                    name=meta["name"],
                    description=meta["description"],
                )
            for sefirah, meta in _KABBALAH_SEFIROTH.items():
                session.run(
                    """
                    MERGE (k:KabbalahNode {sefirah: $sefirah})
                    ON CREATE SET k.number = $number, k.description = $description
                    """,
                    sefirah=sefirah,
                    number=meta["number"],
                    description=meta["description"],
                )
        logger.debug("Reference nodes seeded (VicoCycle × %d, KabbalahNode × %d)",
                     len(_VICO_CYCLES), len(_KABBALAH_SEFIROTH))

    # ------------------------------------------------------------------
    # File-level ingest
    # ------------------------------------------------------------------

    def ingest_file(self, path: Path) -> int:
        """
        Parse a single McHugh JSON annotation file and write its contents
        to Neo4j.

        Parameters
        ----------
        path : Path
            Path to the JSON file (e.g. ``corpus/mcHugh/page_003.json``).

        Returns
        -------
        int
            Number of annotation records successfully written.
        """
        logger.info("Ingesting %s", path)
        try:
            with open(path, encoding="utf-8") as fh:
                data: dict[str, Any] = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read %s: %s", path, exc)
            return 0

        page: int = int(data.get("page", 0))
        if page == 0:
            logger.warning("File %s has no 'page' field — skipping", path)
            return 0

        raw_annotations: list[dict[str, Any]] = data.get("annotations", [])
        records: list[AnnotationRecord] = []
        for raw in raw_annotations:
            try:
                records.append(AnnotationRecord.from_dict(page, raw))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping malformed annotation in %s: %s — %s", path, raw, exc)

        count = 0
        with self._driver.session(database=self.database) as session:
            for i in range(0, len(records), self.batch_size):
                batch = records[i : i + self.batch_size]
                try:
                    self._write_batch(session, batch)
                    count += len(batch)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Batch write error at offset %d in %s: %s", i, path, exc)

        logger.info("Ingested %d annotations from %s (page %d)", count, path.name, page)
        return count

    def ingest_directory(self, directory: Path) -> int:
        """
        Ingest all ``*.json`` files found in ``directory`` (non-recursive).

        Files are processed in alphabetical order so that page ordering is
        deterministic in logs.

        Parameters
        ----------
        directory : Path
            Directory to search for JSON annotation files.

        Returns
        -------
        int
            Total number of annotation records written across all files.
        """
        json_files = sorted(directory.glob("*.json"))
        if not json_files:
            logger.warning("No JSON files found in %s", directory)
            return 0

        logger.info("Found %d JSON files in %s", len(json_files), directory)
        total = 0
        for json_path in json_files:
            total += self.ingest_file(json_path)
        logger.info("Directory ingest complete: %d total annotations from %s",
                    total, directory)
        return total

    def ingest_notebooks(self, path: Path) -> None:
        """
        Ingest annotations embedded in Jupyter notebooks (.ipynb).

        Notebooks may contain annotation cells with inline JSON blocks
        identified by the cell metadata key ``wake_annotations: true``.
        This method extracts those blocks and delegates to
        ``ingest_file`` on a temporary JSON file.

        Parameters
        ----------
        path : Path
            Path to a ``.ipynb`` file or directory containing notebooks.
        """
        # TODO: implement nbformat-based extraction of wake_annotations cells,
        #       serialise each block to a NamedTemporaryFile, and call ingest_file.
        #       Requires: import nbformat; nbformat.read(path, as_version=4)
        logger.info("Notebook ingest from %s not yet implemented", path)

    # ------------------------------------------------------------------
    # Batch write helpers
    # ------------------------------------------------------------------

    def _write_batch(self, session: Session, batch: list[AnnotationRecord]) -> None:
        """Write a list of AnnotationRecords in a single transaction."""
        with session.begin_transaction() as tx:
            for record in batch:
                self._write_token(tx, record)
            tx.commit()

    def _write_token(self, tx: Any, record: AnnotationRecord) -> None:  # noqa: ANN401
        """Write one AnnotationRecord: token, roots, fields, cycle, sefirah."""
        # 1. WakeToken
        tx.run(
            """
            MERGE (t:WakeToken {token_id: $token_id})
            ON CREATE SET
                t.surface   = $surface,
                t.page      = $page,
                t.line      = $line,
                t.notes     = $notes
            ON MATCH SET
                t.notes     = $notes
            """,
            token_id=record.token_id,
            surface=record.surface,
            page=record.page,
            line=record.line,
            notes=record.notes,
        )

        # 2. EtymRoot nodes + HAS_ROOT relationships
        for root in record.roots:
            tx.run(
                """
                MERGE (r:EtymRoot {root_id: $root_id})
                ON CREATE SET r.language = $language, r.form = $form, r.gloss = $gloss
                WITH r
                MATCH (t:WakeToken {token_id: $token_id})
                MERGE (t)-[rel:HAS_ROOT]->(r)
                ON CREATE SET rel.confidence = $confidence
                ON MATCH SET  rel.confidence = $confidence
                """,
                root_id=root.root_id,
                language=root.language,
                form=root.form,
                gloss=root.gloss,
                token_id=record.token_id,
                confidence=root.confidence,
            )
            # Language node
            tx.run(
                """
                MERGE (lang:Language {code: $code})
                WITH lang
                MATCH (r:EtymRoot {root_id: $root_id})
                MERGE (r)-[:IN_LANGUAGE]->(lang)
                """,
                code=root.language,
                root_id=root.root_id,
            )

        # 3. SemanticField nodes + ACTIVATES relationships
        for sf_name in record.semantic_fields:
            tx.run(
                """
                MERGE (s:SemanticField {name: $name})
                WITH s
                MATCH (t:WakeToken {token_id: $token_id})
                MERGE (t)-[:ACTIVATES]->(s)
                """,
                name=sf_name,
                token_id=record.token_id,
            )

        # 4. VicoCycle relationship
        if record.vico_cycle is not None:
            tx.run(
                """
                MATCH (v:VicoCycle {cycle: $cycle})
                MATCH (t:WakeToken {token_id: $token_id})
                MERGE (t)-[:BELONGS_TO_CYCLE]->(v)
                """,
                cycle=record.vico_cycle,
                token_id=record.token_id,
            )

        # 5. KabbalahNode relationship
        if record.kabbalah_sefirah is not None:
            tx.run(
                """
                MATCH (k:KabbalahNode {sefirah: $sefirah})
                MATCH (t:WakeToken {token_id: $token_id})
                MERGE (t)-[:RESONATES_WITH]->(k)
                """,
                sefirah=record.kabbalah_sefirah,
                token_id=record.token_id,
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> "argparse.ArgumentParser":  # noqa: F821
    import argparse

    parser = argparse.ArgumentParser(
        prog="mcHugh_ingest",
        description=(
            "Ingest McHugh annotation JSON files into the WAKE Neo4j graph. "
            "Provide either a single JSON file or a directory of JSON files."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a single annotation JSON file or directory of JSON files.",
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
        default=100,
        help="Records per write transaction (default: 100)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    ingester = McHughIngester(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        database=args.database,
        batch_size=args.batch_size,
    )

    try:
        target = args.path
        if target.is_dir():
            total = ingester.ingest_directory(target)
        elif target.is_file():
            total = ingester.ingest_file(target)
        else:
            logger.error("Path does not exist: %s", target)
            sys.exit(1)
        print(f"Ingest complete: {total} annotation records written.")
    finally:
        ingester.close()
