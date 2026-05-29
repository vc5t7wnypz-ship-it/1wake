"""
WAKE Engine — Tokenizer
=======================
Implements WakeToken and WakeTokenizer for mechanistic interpretability of
Finnegans Wake.  Each surface token is decomposed into candidate morphemes,
attributed to languages, and assigned a confidence score.

Spec reference: section 2.1
"""

from __future__ import annotations

import json
import re
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Named tuple for raw surface fragments produced before full tokenisation
# ---------------------------------------------------------------------------

SurfaceToken = namedtuple("SurfaceToken", ["value", "span"])


# ---------------------------------------------------------------------------
# Morpheme database loader (module-level function)
# ---------------------------------------------------------------------------

def _load_morpheme_db(path: str) -> Dict[str, Any]:
    """Load JSON morpheme database from *path*.

    Each entry maps a surface form to a dict with at minimum:
        ``confidence`` (float), ``language`` (str), ``meaning`` (str).

    If *path* does not exist the function returns a hardcoded seed dictionary
    covering the most-annotated morphemes from Finnegans Wake pp. 3-10.
    """
    db_path = Path(path)
    if db_path.exists():
        with db_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]

    # --- hardcoded seed (used when no external DB is found) ---
    seed: Dict[str, Any] = {
        "river": {"confidence": 0.95, "language": "en", "meaning": "watercourse"},
        "run": {"confidence": 0.90, "language": "en", "meaning": "to flow"},
        "rune": {"confidence": 0.85, "language": "non", "meaning": "secret character"},
        "past": {"confidence": 0.95, "language": "en", "meaning": "beyond"},
        "Adam": {"confidence": 0.95, "language": "he", "meaning": "earth"},
        "Eve": {"confidence": 0.95, "language": "he", "meaning": "life"},
        # additional morphemes from FW pp. 3-10
        "Howth": {"confidence": 0.95, "language": "non", "meaning": "headland promontory"},
        "head": {"confidence": 0.92, "language": "en", "meaning": "top of the body; promontory"},
        "cast": {"confidence": 0.90, "language": "en", "meaning": "to throw; bend in a river"},
        "swerve": {"confidence": 0.90, "language": "en", "meaning": "to deviate from a course"},
        "long": {"confidence": 0.90, "language": "en", "meaning": "extending a great distance"},
        "shore": {"confidence": 0.90, "language": "en", "meaning": "land along the edge of water"},
        "bend": {"confidence": 0.90, "language": "en", "meaning": "a curve in a river"},
        "bay": {"confidence": 0.90, "language": "en", "meaning": "broad inlet of the sea"},
        "double": {"confidence": 0.88, "language": "en", "meaning": "consisting of two parts"},
        "vicus": {"confidence": 0.95, "language": "la", "meaning": "street; village; Vico Road"},
        "commodius": {
            "confidence": 0.93,
            "language": "la",
            "meaning": "convenient; Emperor Commodus",
        },
        "vicorium": {
            "confidence": 0.85,
            "language": "la",
            "meaning": "of or pertaining to Vico",
        },
        "Finn": {
            "confidence": 0.95,
            "language": "ga",
            "meaning": "fair, bright; Fionn Mac Cumhaill",
        },
        "anna": {"confidence": 0.90, "language": "he", "meaning": "grace; Anna Livia"},
        "liff": {"confidence": 0.90, "language": "non", "meaning": "the Liffey; life water"},
        "plurabelle": {
            "confidence": 0.85,
            "language": "la",
            "meaning": "more beautiful",
        },
        "thunder": {
            "confidence": 0.95,
            "language": "en",
            "meaning": "divine voice; Viconian thunderclap",
        },
        "water": {"confidence": 0.95, "language": "en", "meaning": "flowing liquid"},
        "night": {"confidence": 0.93, "language": "en", "meaning": "period of darkness"},
        "fall": {"confidence": 0.90, "language": "en", "meaning": "the Fall of Man; autumn"},
        "rivus": {"confidence": 0.90, "language": "la", "meaning": "stream, brook"},
    }
    return seed


# ---------------------------------------------------------------------------
# WakeToken dataclass
# ---------------------------------------------------------------------------

@dataclass
class WakeToken:
    """A single token from Finnegans Wake with full linguistic annotation.

    Attributes
    ----------
    surface:
        The exact text as it appears in the Wake.
    page:
        Faber/Viking 1939 edition page number (3-628).
    line:
        1-based line number within the page.
    position:
        0-based token index within the line.
    morphemes:
        Ordered list of component morpheme strings from portmanteau decomposition.
    languages:
        Deduplicated list of ISO language codes attributed to this token.
    confidence:
        Overall confidence score for the decomposition (0.0-1.0).
    token_id:
        Canonical identifier ``"{page}_{line}_{position}"``.
    span:
        Character span ``(start, end)`` within the original line string.
    notes:
        Optional free-text annotation.
    graph_nodes:
        IDs of knowledge-graph nodes activated by this token (populated lazily).
    """

    surface: str
    page: int
    line: int
    position: int
    morphemes: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    confidence: float = 0.0
    token_id: str = ""
    span: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    notes: Optional[str] = None
    graph_nodes: List[str] = field(default_factory=list)
    # Legacy compatibility: expose is_portmanteau for callers that check it
    is_portmanteau: bool = False

    def __post_init__(self) -> None:
        if not self.token_id:
            self.token_id = f"{self.page}_{self.line}_{self.position}"
        # Derive is_portmanteau from morphemes list length
        if not self.is_portmanteau:
            self.is_portmanteau = len(self.morphemes) > 1

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary suitable for JSON / Neo4j ingestion."""
        return {
            "token_id": self.token_id,
            "surface": self.surface,
            "page": self.page,
            "line": self.line,
            "position": self.position,
            "morphemes": self.morphemes,
            "languages": self.languages,
            "confidence": self.confidence,
            "span": list(self.span),
            "notes": self.notes,
            "graph_nodes": self.graph_nodes,
            "is_portmanteau": self.is_portmanteau,
        }


# ---------------------------------------------------------------------------
# WakeTokenizer
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = Path(__file__).parent / "morpheme_db" / "seed_morphemes.json"

# Minimum sub-string length to consider as a morpheme candidate.
_MIN_MORPH_LEN = 2


class WakeTokenizer:
    """Tokenizer specialised for Finnegans Wake text.

    Parameters
    ----------
    morpheme_db_path:
        Path to a JSON morpheme database.  Defaults to the bundled seed file
        in ``engine/tokenizer/morpheme_db/seed_morphemes.json``.
    min_morpheme_len:
        Minimum character length for a morpheme candidate (default 2).
    confidence_threshold:
        Minimum decomposition confidence required before accepting a split
        (default 0.5).  Below this threshold the surface form is returned
        as a single-morpheme token.
    graph:
        Optional Neo4j driver or wrapper whose ``.run(query, **kwargs)``
        method is called to enrich tokens with graph-node IDs.  Pass None
        for offline mode.
    """

    # Unicode-aware regex capturing runs of alphabetic characters, including
    # internal hyphens and apostrophes (e.g. "Adam's", "father-figure").
    _WORD_RE = re.compile(
        r"[A-Za-zÀ-ÖØ-öø-ÿЀ-ӿԀ-ԯͰ-ϿЀ-ӿ]+"
        r"(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿЀ-ӿԀ-ԯͰ-ϿЀ-ӿ]+)*",
        re.UNICODE,
    )

    def __init__(
        self,
        morpheme_db_path: Optional[str] = None,
        min_morpheme_len: int = _MIN_MORPH_LEN,
        confidence_threshold: float = 0.5,
        graph: Optional[Any] = None,
        morpheme_db: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Allow passing a pre-loaded dict for testing / injection
        if morpheme_db is not None:
            self.morpheme_db: Dict[str, Any] = morpheme_db
        else:
            if morpheme_db_path is None:
                morpheme_db_path = str(_DEFAULT_DB_PATH)
            self.morpheme_db = _load_morpheme_db(morpheme_db_path)

        self.min_morpheme_len = min_morpheme_len
        self.confidence_threshold = confidence_threshold
        self.graph = graph

        # Build a normalised lower-case lookup for case-insensitive matching
        # while preserving the original-case keys for language attribution.
        self._lower_db: Dict[str, str] = {k.lower(): k for k in self.morpheme_db}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(
        self,
        text: str,
        page: int = 3,
        line: int = 1,
        line_offset: int = 0,
    ) -> List[WakeToken]:
        """Tokenize a single line of Wake text.

        Parameters
        ----------
        text:
            Raw line text.
        page:
            Page number (Faber 1939 edition).
        line:
            1-based line number.
        line_offset:
            Character offset of the first character of *text* within a
            larger passage (used when tokenizing passages split across
            multiple lines).

        Returns
        -------
        List of :class:`WakeToken` objects, one per surface word.
        """
        surface_tokens = self._split_surface(text)
        wake_tokens: List[WakeToken] = []

        for position, st in enumerate(surface_tokens):
            morphemes, confidence = self._decompose_portmanteau(st.value)
            languages = self._attribute_languages(st.value, morphemes)

            adjusted_span = (
                st.span[0] + line_offset,
                st.span[1] + line_offset,
            )

            tok = WakeToken(
                surface=st.value,
                page=page,
                line=line,
                position=position,
                morphemes=morphemes,
                languages=languages,
                confidence=confidence,
                span=adjusted_span,
            )

            if self.graph is not None:
                tok.graph_nodes = self._fetch_graph_nodes(st.value)

            wake_tokens.append(tok)

        return wake_tokens

    def tokenize_passage(
        self,
        text: str,
        page: int = 3,
        line: int = 1,
    ) -> List[WakeToken]:
        """Tokenize a multi-line passage.

        Splits *text* on newlines first, then tokenizes each line
        independently, incrementing the line counter.  All tokens across
        all lines are returned in a single flat list with correct line
        metadata.

        Parameters
        ----------
        text:
            Multi-line passage text (newline-separated).
        page:
            Starting page number.
        line:
            Starting 1-based line number.

        Returns
        -------
        Flat list of :class:`WakeToken` objects.
        """
        all_tokens: List[WakeToken] = []
        current_line = line

        for raw_line in text.splitlines():
            line_tokens = self.tokenize(raw_line, page=page, line=current_line)
            all_tokens.extend(line_tokens)
            current_line += 1

        return all_tokens

    # ------------------------------------------------------------------
    # Surface splitting
    # ------------------------------------------------------------------

    def _split_surface(self, text: str) -> List[SurfaceToken]:
        """Split *text* into surface tokens preserving span information.

        Uses a Unicode-aware word regex to identify token boundaries.
        Punctuation and whitespace are silently dropped (not emitted as
        separate tokens).

        Returns
        -------
        List of :class:`SurfaceToken` namedtuples, each with a ``value``
        (str) and ``span`` ((start, end) character positions).
        """
        return [
            SurfaceToken(value=m.group(0), span=(m.start(), m.end()))
            for m in self._WORD_RE.finditer(text)
        ]

    # ------------------------------------------------------------------
    # Language attribution
    # ------------------------------------------------------------------

    def _attribute_languages(
        self,
        token: str,
        candidates: List[str],
    ) -> List[str]:
        """Return deduplicated ISO language codes for *candidates*.

        For each candidate morpheme the morpheme_db is consulted.  If the
        morpheme is found, its ``language`` value is added to the result
        set, ordered by first appearance.  When a candidate is not in the
        database the original surface ``token`` is tried as a fallback.

        Parameters
        ----------
        token:
            The original surface form (used as a fallback lookup key).
        candidates:
            List of candidate morpheme strings to look up.

        Returns
        -------
        Deduplicated list of ISO 639-1 / extended language codes.
        """
        # Use insertion-ordered dict as an ordered set
        seen: Dict[str, None] = {}

        def _add_lang(key: str) -> None:
            # Try exact match first
            if key in self.morpheme_db:
                lang = self.morpheme_db[key].get("language", "")
                if lang and lang not in seen:
                    seen[lang] = None
                return
            # Then case-insensitive fallback
            lower = key.lower()
            if lower in self._lower_db:
                canonical = self._lower_db[lower]
                lang = self.morpheme_db[canonical].get("language", "")
                if lang and lang not in seen:
                    seen[lang] = None

        for candidate in candidates:
            _add_lang(candidate)

        # Fallback: look up the full surface form when no candidate matched
        if not seen:
            _add_lang(token)

        # Second fallback: default to English
        if not seen:
            seen["en"] = None

        return list(seen.keys())

    # ------------------------------------------------------------------
    # Portmanteau decomposition — dynamic programming
    # ------------------------------------------------------------------

    def _decompose_portmanteau(
        self,
        surface: str,
    ) -> Tuple[List[str], float]:
        """Decompose *surface* into morpheme constituents via DP word-break.

        Algorithm
        ---------
        Uses a forward DP pass: ``best_conf[i]`` holds the highest
        length-weighted average confidence achievable for the prefix
        ``surface[:i]``.  Back-pointers allow path reconstruction.

        When the DP yields only the original surface (one segment) the
        whole-word confidence from the DB is used; if the DP finds a
        multi-morpheme split whose joint confidence exceeds the whole-word
        entry the split is preferred.

        Returns
        -------
        ``(morphemes, confidence)``
            *morphemes* — list of constituent strings in left-to-right order.
            *confidence* — confidence estimate in [0.0, 1.0].
        """
        s_lower = surface.lower()
        n = len(s_lower)

        if n == 0:
            return [], 0.0

        def _db_confidence(key: str) -> float:
            """Look up confidence for *key* (tries original case then lower)."""
            if key in self.morpheme_db:
                return float(self.morpheme_db[key].get("confidence", 0.5))
            lk = key.lower()
            if lk in self._lower_db:
                canonical = self._lower_db[lk]
                return float(self.morpheme_db[canonical].get("confidence", 0.5))
            return 0.0

        full_conf = _db_confidence(surface)

        # ------------------------------------------------------------------
        # DP tables
        # ------------------------------------------------------------------
        # best_conf[i] = best length-weighted average confidence for s_lower[:i]
        # back[i]      = j: the split point such that s_lower[j:i] is the last
        #                morpheme in the best segmentation of s_lower[:i]
        best_conf: List[float] = [-1.0] * (n + 1)
        back: List[Optional[int]] = [None] * (n + 1)
        best_conf[0] = 1.0  # empty prefix: trivially perfect

        max_morph_len = 30  # cap look-back to keep complexity linear

        for i in range(1, n + 1):
            for j in range(max(0, i - max_morph_len), i):
                if best_conf[j] < 0:
                    continue
                fragment = s_lower[j:i]
                if len(fragment) < self.min_morpheme_len:
                    continue
                frag_conf = _db_confidence(fragment)
                if frag_conf <= 0.0:
                    continue
                # Length-weighted running average: weight longer morphemes more
                weight = len(fragment)
                new_conf = (best_conf[j] * j + frag_conf * weight) / (j + weight)
                if new_conf > best_conf[i]:
                    best_conf[i] = new_conf
                    back[i] = j

        # ------------------------------------------------------------------
        # Handle no valid DP segmentation found
        # ------------------------------------------------------------------
        if best_conf[n] < 0 or best_conf[n] < self.confidence_threshold:
            conf = full_conf if full_conf > 0.0 else 0.5
            return [surface], conf

        # ------------------------------------------------------------------
        # Backtrack to reconstruct morphemes
        # ------------------------------------------------------------------
        morphemes_reversed: List[str] = []
        idx = n
        while idx > 0:
            j = back[idx]
            if j is None:
                # Should not happen given DP filled correctly, but guard anyway
                break
            # Recover original-case fragment from surface
            morphemes_reversed.append(surface[j:idx])
            idx = j

        if not morphemes_reversed:
            conf = full_conf if full_conf > 0.0 else 0.5
            return [surface], conf

        morphemes = list(reversed(morphemes_reversed))

        # ------------------------------------------------------------------
        # Prefer whole-word DB entry when it has higher confidence than split
        # ------------------------------------------------------------------
        split_conf = best_conf[n]

        if len(morphemes) == 1:
            conf = full_conf if full_conf > 0.0 else split_conf
            return morphemes, conf

        # Multi-morpheme split: only use it when its confidence beats the
        # whole-word entry
        if full_conf >= split_conf and full_conf > 0.0:
            return [surface], full_conf

        return morphemes, split_conf

    # ------------------------------------------------------------------
    # Graph enrichment
    # ------------------------------------------------------------------

    def _fetch_graph_nodes(self, surface: str) -> List[str]:
        """Query the knowledge graph for node IDs matching *surface*."""
        if self.graph is None:
            return []
        try:
            query = (
                "MATCH (t:WakeToken {surface: $surface}) "
                "RETURN t.token_id AS id LIMIT 10"
            )
            results = self.graph.run(query, surface=surface)
            return [r["id"] for r in results if r.get("id")]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Legacy compatibility — load_seed_db used by older call-sites
    # ------------------------------------------------------------------

    @staticmethod
    def _load_seed_db() -> Dict[str, Any]:
        """Load the seed morpheme JSON from the package data directory."""
        try:
            with _DEFAULT_DB_PATH.open("r", encoding="utf-8") as fh:
                return json.load(fh)  # type: ignore[no-any-return]
        except FileNotFoundError:
            return {}
