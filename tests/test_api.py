"""Tests for the WAKE API using FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock


def test_health_endpoint():
    with patch("api.state.AppState.initialize", new_callable=AsyncMock):
        from api.main import app
        client = TestClient(app)
        response = client.get("/api/wake/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


def test_lenses_endpoint():
    from api.main import app
    client = TestClient(app)
    response = client.get("/api/wake/lenses")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list) or "lenses" in data
