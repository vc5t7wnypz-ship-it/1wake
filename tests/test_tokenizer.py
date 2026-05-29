"""Tests for WakeTokenizer."""

from __future__ import annotations

import pytest
from engine.tokenizer.wake_tokenizer import WakeToken, WakeTokenizer


def test_basic_tokenization(minimal_morpheme_db, mock_graph):
    tokenizer = WakeTokenizer.__new__(WakeTokenizer)
    tokenizer.morpheme_db = minimal_morpheme_db
    tokenizer.graph = mock_graph
    # Build the _lower_db that __init__ normally builds
    tokenizer._lower_db = {k.lower(): k for k in minimal_morpheme_db}
    tokenizer.min_morpheme_len = 2
    tokenizer.confidence_threshold = 0.5
    tokens = tokenizer.tokenize("riverrun", 3, 1)
    assert len(tokens) >= 1
    assert tokens[0].surface == "riverrun"
    assert tokens[0].page == 3
    assert tokens[0].line == 1


def test_portmanteau_decomposition(minimal_morpheme_db, mock_graph):
    tokenizer = WakeTokenizer.__new__(WakeTokenizer)
    tokenizer.morpheme_db = minimal_morpheme_db
    tokenizer.graph = mock_graph
    tokenizer._lower_db = {k.lower(): k for k in minimal_morpheme_db}
    tokenizer.min_morpheme_len = 2
    tokenizer.confidence_threshold = 0.5
    candidates, confidence = tokenizer._decompose_portmanteau("riverrun")
    assert len(candidates) >= 1


def test_language_attribution(minimal_morpheme_db, mock_graph):
    tokenizer = WakeTokenizer.__new__(WakeTokenizer)
    tokenizer.morpheme_db = minimal_morpheme_db
    tokenizer.graph = mock_graph
    tokenizer._lower_db = {k.lower(): k for k in minimal_morpheme_db}
    tokenizer.min_morpheme_len = 2
    tokenizer.confidence_threshold = 0.5
    langs = tokenizer._attribute_languages("riverrun", ["river", "run"])
    assert isinstance(langs, list)
    # Should find at least English
    assert len(langs) >= 1


def test_split_surface():
    tokenizer = WakeTokenizer.__new__(WakeTokenizer)
    tokenizer.morpheme_db = {}
    tokenizer.graph = None
    tokenizer._lower_db = {}
    tokenizer.min_morpheme_len = 2
    tokenizer.confidence_threshold = 0.5
    surface_tokens = tokenizer._split_surface("riverrun, past Eve")
    # Should yield 3 tokens: riverrun, past, Eve (comma stripped)
    values = [st.value for st in surface_tokens]
    assert len(values) == 3  # riverrun, past, Eve
    assert "riverrun" in values
    assert "past" in values
    assert "Eve" in values
