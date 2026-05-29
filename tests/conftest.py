"""
Pytest fixtures shared across the WAKE test suite.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# sample_passage
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_passage() -> Dict[str, Any]:
    """Return a canonical FW opening passage dict with page/line metadata."""
    return {
        "passage": "riverrun, past Eve and Adam's, from swerve of shore to bend of bay",
        "page": 3,
        "line": 1,
    }


# ---------------------------------------------------------------------------
# mock_graph
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_graph() -> MagicMock:
    """Return a Mock graph client whose run() always returns []."""
    graph = MagicMock()
    graph.run.return_value = []
    return graph


# ---------------------------------------------------------------------------
# mock_wake_token_list
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_wake_token_list(sample_passage: Dict[str, Any]) -> List[Any]:
    """Return a list of 5 WakeToken objects for the sample passage."""
    from engine.tokenizer.wake_tokenizer import WakeToken

    passage = sample_passage["passage"]
    page = sample_passage["page"]
    line = sample_passage["line"]

    # Hand-craft 5 representative tokens from the opening of FW p.3
    tokens = [
        WakeToken(
            surface="riverrun",
            page=page,
            line=line,
            position=0,
            morphemes=["river", "run"],
            languages=["en"],
            confidence=0.92,
            is_portmanteau=True,
        ),
        WakeToken(
            surface="past",
            page=page,
            line=line,
            position=1,
            morphemes=["past"],
            languages=["en"],
            confidence=0.95,
        ),
        WakeToken(
            surface="Eve",
            page=page,
            line=line,
            position=2,
            morphemes=["Eve"],
            languages=["he"],
            confidence=0.95,
        ),
        WakeToken(
            surface="Adam",
            page=page,
            line=line,
            position=3,
            morphemes=["Adam"],
            languages=["he"],
            confidence=0.95,
        ),
        WakeToken(
            surface="swerve",
            page=page,
            line=line,
            position=4,
            morphemes=["swerve"],
            languages=["en"],
            confidence=0.90,
        ),
    ]
    return tokens


# ---------------------------------------------------------------------------
# minimal_morpheme_db
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_morpheme_db() -> Dict[str, Any]:
    """Return a small in-memory morpheme dict for testing without disk access.

    Contains the most important morphemes needed for the FW opening passage.
    """
    return {
        "river": {
            "confidence": 0.95,
            "language": "en",
            "meaning": "a natural watercourse",
            "period": "Middle English",
            "cognates": ["rivus", "rivo"],
        },
        "run": {
            "confidence": 0.90,
            "language": "en",
            "meaning": "to flow, to move continuously",
            "period": "Old English",
            "cognates": ["rinnan"],
        },
        "rune": {
            "confidence": 0.85,
            "language": "non",
            "meaning": "secret character, mystery",
            "period": "Old Norse",
            "cognates": ["runa"],
        },
        "past": {
            "confidence": 0.95,
            "language": "en",
            "meaning": "gone by in time",
            "period": "Middle English",
            "cognates": ["passus"],
        },
        "Eve": {
            "confidence": 0.95,
            "language": "he",
            "meaning": "life, the living one",
            "period": "Biblical Hebrew",
            "cognates": ["Chavah"],
        },
        "Adam": {
            "confidence": 0.95,
            "language": "he",
            "meaning": "earth, red clay",
            "period": "Biblical Hebrew",
            "cognates": ["adamah"],
        },
        "swerve": {
            "confidence": 0.90,
            "language": "en",
            "meaning": "to deviate suddenly",
            "period": "Middle English",
            "cognates": ["clinamen"],
        },
        "shore": {
            "confidence": 0.90,
            "language": "en",
            "meaning": "land along the edge of water",
            "period": "Middle English",
            "cognates": ["schor"],
        },
        "bend": {
            "confidence": 0.90,
            "language": "en",
            "meaning": "a curve in a river",
            "period": "Old English",
            "cognates": ["biegung"],
        },
        "bay": {
            "confidence": 0.90,
            "language": "en",
            "meaning": "a broad inlet of the sea",
            "period": "Middle English",
            "cognates": ["baie"],
        },
        "from": {
            "confidence": 1.0,
            "language": "en",
            "meaning": "preposition indicating origin",
            "period": "Old English",
            "cognates": ["von"],
        },
        "of": {
            "confidence": 1.0,
            "language": "en",
            "meaning": "preposition of belonging",
            "period": "Old English",
            "cognates": ["von"],
        },
        "to": {
            "confidence": 1.0,
            "language": "en",
            "meaning": "preposition of direction",
            "period": "Old English",
            "cognates": ["zu"],
        },
        "and": {
            "confidence": 1.0,
            "language": "en",
            "meaning": "conjunction",
            "period": "Old English",
            "cognates": ["und"],
        },
    }
