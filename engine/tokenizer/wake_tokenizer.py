"""
WakeTokenizer — morphological tokeniser for Finnegans Wake.

Splits surface text into WakeToken objects that carry:
  - the raw surface string
  - candidate morpheme decompositions (portmanteau analysis)
  - attributed source languages for each component
  - page / line provenance from the source text
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WakeToken:
    """A single token extracted from Finnegans Wake.

    Attributes
    ----------
    surface:
        The raw, un-normalised surface string as it appears in the text.
    page:
        Source page number (1-indexed, matching the standard Faber edition).
    line:
        Line number on that page (1-indexed).
    morphemes:
        Decomposition candidates: list of (morpheme_string, confidence) pairs.
    languages:
        BCP-47 / ISO 639-3 language codes attributed to this token's roots.
    is_portmanteau:
        True when the token appears to fuse two or more distinct morphemes.
    graph_nodes:
        IDs of knowledge-graph nodes that this token activates (populated
        lazily by the graph traversal layer).
    """

    surface: str
    page: int
    line: int
    morphemes: List[Tuple[str, float]] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    is_portmanteau: bool = False
    graph_nodes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = Path(__file__).parent / "morpheme_db" / "seed_morphemes.json"

# Minimum sub-string length to consider as a morpheme candidate.
_MIN_MORPH_LEN = 3


class WakeTokenizer:
    """Morphological tokeniser for Finnegans Wake passages.

    Parameters
    ----------
    morpheme_db:
        Dictionary mapping morpheme strings to metadata dicts (``language``,
        ``confidence``, ``meaning`` …).  When *None* the class attempts to
        load from the seed JSON on disk.
    graph:
        An optional Neo4j driver / wrapper whose ``.run(query, **kwargs)``
        method is called to enrich tokens with graph-node IDs.  Pass *None*
        for offline mode.
    """

    def __init__(
        self,
        morpheme_db: Optional[Dict[str, Any]] = None,
        graph: Optional[Any] = None,
    ) -> None:
        if morpheme_db is None:
            morpheme_db = self._load_seed_db()
        self.morpheme_db: Dict[str, Any] = morpheme_db
        self.graph = graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self, text: str, page: int, line: int) -> List[WakeToken]:
        """Tokenise *text* into a list of :class:`WakeToken` objects.

        Parameters
        ----------
        text:
            A raw passage string from Finnegans Wake.
        page:
            Source page number.
        line:
            Starting line number for this passage.

        Returns
        -------
        list[WakeToken]
            One token per surface word.  Punctuation is stripped.
        """
        raw_surfaces = self._split_surface(text)
        tokens: List[WakeToken] = []
        for surface in raw_surfaces:
            morphemes = self._decompose_portmanteau(surface)
            languages = self._attribute_languages(surface, [m for m, _ in morphemes])
            is_portmanteau = len(set(languages)) > 1 or len(morphemes) > 1
            tok = WakeToken(
                surface=surface,
                page=page,
                line=line,
                morphemes=morphemes,
                languages=languages,
                is_portmanteau=is_portmanteau,
            )
            if self.graph is not None:
                tok.graph_nodes = self._fetch_graph_nodes(surface)
            tokens.append(tok)
        return tokens

    # ------------------------------------------------------------------
    # Surface splitting
    # ------------------------------------------------------------------

    def _split_surface(self, text: str) -> List[str]:
        """Split *text* into non-empty word strings, stripping punctuation."""
        # Preserve apostrophes inside words (e.g., "Adam's") but strip
        # leading/trailing punctuation from each token.
        raw_words = text.split()
        surfaces: List[str] = []
        for w in raw_words:
            # Strip surrounding punctuation but keep internal apostrophes/hyphens.
            cleaned = re.sub(r"^[^\w''-]+|[^\w''-]+$", "", w, flags=re.UNICODE)
            # Further strip trailing punctuation like commas, periods
            cleaned = cleaned.rstrip(".,;:!?\"'")
            cleaned = cleaned.lstrip("\"'(")
            if cleaned:
                surfaces.append(cleaned)
        return surfaces

    # ------------------------------------------------------------------
    # Portmanteau decomposition
    # ------------------------------------------------------------------

    def _decompose_portmanteau(self, surface: str) -> List[Tuple[str, float]]:
        """Return candidate morpheme decompositions for *surface*.

        Strategy
        --------
        1. Check if the whole surface (lower-cased) is in the morpheme DB →
           return as a single-element list.
        2. Try all substring splits of length ≥ _MIN_MORPH_LEN that are
           both in the DB.
        3. Fall back to the raw surface with confidence 0.5 if no split found.

        Returns
        -------
        list of (morpheme_string, confidence) pairs, sorted descending by
        confidence.
        """
        lower = surface.lower()

        # Exact match
        if lower in self.morpheme_db:
            conf = float(self.morpheme_db[lower].get("confidence", 0.9))
            return [(lower, conf)]

        # Check without trailing punctuation / case variants
        stripped = re.sub(r"[^a-z]", "", lower)
        if stripped in self.morpheme_db:
            conf = float(self.morpheme_db[stripped].get("confidence", 0.85))
            return [(stripped, conf)]

        # Substring split scan
        candidates: List[Tuple[str, float]] = []
        n = len(stripped)
        for split in range(_MIN_MORPH_LEN, n - _MIN_MORPH_LEN + 1):
            left = stripped[:split]
            right = stripped[split:]
            if left in self.morpheme_db and right in self.morpheme_db:
                conf_l = float(self.morpheme_db[left].get("confidence", 0.8))
                conf_r = float(self.morpheme_db[right].get("confidence", 0.8))
                # Joint confidence as geometric mean of the two components.
                joint = (conf_l * conf_r) ** 0.5
                candidates.append((left, conf_l))
                candidates.append((right, conf_r))
                # Return the best split found (first valid split wins).
                return sorted(candidates, key=lambda x: x[1], reverse=True)

        # Prefix scan — partial overlap with a known morpheme
        for morph in sorted(self.morpheme_db.keys(), key=len, reverse=True):
            if stripped.startswith(morph) and len(morph) >= _MIN_MORPH_LEN:
                conf = float(self.morpheme_db[morph].get("confidence", 0.7))
                return [(morph, conf), (stripped[len(morph):], 0.5)]
            if stripped.endswith(morph) and len(morph) >= _MIN_MORPH_LEN:
                conf = float(self.morpheme_db[morph].get("confidence", 0.7))
                return [(stripped[: -len(morph)], 0.5), (morph, conf)]

        # Fallback: return the whole surface with moderate confidence.
        return [(stripped or surface.lower(), 0.5)]

    # ------------------------------------------------------------------
    # Language attribution
    # ------------------------------------------------------------------

    def _attribute_languages(self, surface: str, morphemes: List[str]) -> List[str]:
        """Return a list of ISO 639 language codes associated with *morphemes*.

        Parameters
        ----------
        surface:
            Original surface string (used as fallback when no morpheme matches).
        morphemes:
            List of morpheme strings to look up.

        Returns
        -------
        list[str]
            De-duplicated language codes, most-confident first.
        """
        langs_seen: dict[str, float] = {}
        for morph in morphemes:
            entry = self.morpheme_db.get(morph.lower())
            if entry:
                lang = entry.get("language", "en")
                conf = float(entry.get("confidence", 0.5))
                # Keep the highest confidence per language.
                if lang not in langs_seen or langs_seen[lang] < conf:
                    langs_seen[lang] = conf

        if not langs_seen:
            # Check the surface itself
            entry = self.morpheme_db.get(surface.lower())
            if entry:
                lang = entry.get("language", "en")
                conf = float(entry.get("confidence", 0.5))
                langs_seen[lang] = conf
            else:
                langs_seen["en"] = 0.5

        return sorted(langs_seen, key=lambda l: langs_seen[l], reverse=True)

    # ------------------------------------------------------------------
    # Graph enrichment
    # ------------------------------------------------------------------

    def _fetch_graph_nodes(self, surface: str) -> List[str]:
        """Query the knowledge graph for node IDs matching *surface*."""
        if self.graph is None:
            return []
        try:
            query = (
                "MATCH (t:WakeToken {surface: $surface}) RETURN t.id AS id LIMIT 10"
            )
            results = self.graph.run(query, surface=surface)
            return [r["id"] for r in results if r.get("id")]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # DB loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_seed_db() -> Dict[str, Any]:
        """Load the seed morpheme JSON from the package data directory."""
        try:
            with _DEFAULT_DB_PATH.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {}
