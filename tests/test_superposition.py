"""Tests for superposition detection."""

from __future__ import annotations

import numpy as np
import pytest
from interpretability.probes.language_probes import ProbeResult, SuperpositionDetector


def make_probe_result(preds: dict) -> ProbeResult:
    top_field = max(preds, key=preds.get)
    return ProbeResult(
        layer=16,
        position=0,
        token="test",
        predictions=preds,
        confidence=max(preds.values()),
        top_field=top_field,
    )


def test_collapsed_detection():
    det = SuperpositionDetector()
    # One field dominant — should NOT be superposed
    preds = {"en_standard": 0.9, "ga_irish": 0.05, "de_german": 0.03, "la_latin": 0.02}
    result = make_probe_result(preds)
    analysis = det.analyze([result], np.random.randn(4096), {})
    assert not analysis["is_superposed"]
    assert analysis["dominant_field"] == "en_standard"


def test_superposition_detection():
    det = SuperpositionDetector()
    # Multiple fields active — should BE superposed
    preds = {
        "en_standard": 0.45,
        "ga_irish": 0.42,
        "non_norse": 0.40,
        "la_latin": 0.38,
        "he_hebrew": 0.1,
        "de_german": 0.08,
        "it_italian": 0.05,
        "fr_french": 0.03,
    }
    result = make_probe_result(preds)
    analysis = det.analyze([result], np.random.randn(4096), {})
    assert analysis["is_superposed"]
    assert analysis["n_active_fields"] >= 3
