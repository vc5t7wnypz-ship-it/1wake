"""Tests for the WAKE lens system."""

from __future__ import annotations

import pytest
from lenses.base import Lens, LensConfig
from lenses.viconian import ViconianLens
from lenses.kabbalistic import KabbalisticLens
from lenses.registry import ALL_LENSES, get_lens, list_lenses, ComposedLens


def test_all_lenses_instantiate():
    for name, cls in ALL_LENSES.items():
        lens = cls()
        assert lens.config.name == name
        assert len(lens.config.system_prompt) > 100
        assert len(lens.config.foregrounded_fields) > 0


def test_lens_config_completeness():
    lens = ViconianLens()
    assert lens.config.foregrounded_fields
    assert lens.config.probe_targets
    assert lens.config.graph_traversal


def test_composed_lens(mock_graph):
    lens_a = ViconianLens()
    lens_b = KabbalisticLens()
    composed = ComposedLens([lens_a, lens_b])
    assert composed.name == "viconian+kabbalistic"
    assert abs(sum(composed.weights) - 1.0) < 0.01


def test_list_lenses():
    names = list_lenses()
    assert "viconian" in names
    assert "kabbalistic" in names
    assert "freudian" in names
    assert "irish_mythology" in names
    assert "norse" in names
    assert "brunian" in names


def test_get_context_without_graph():
    lens = ViconianLens(graph_client=None)
    tokens = []
    ctx = lens.get_context_for_passage("riverrun", tokens)
    assert "VICONIAN" in ctx.upper() or "vico" in ctx.lower()
