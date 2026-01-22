"""Tests for persisted annotations in the FastAPI service.

Run with:
    pytest tests/test_annotations_api.py
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from highest_volatility.app import api


async def _noop_refresh(**_: object) -> None:
    await asyncio.sleep(0)


def _reset_annotation_state() -> None:
    for attr in ("annotation_store", "annotation_lock"):
        if hasattr(api.app.state, attr):
            delattr(api.app.state, attr)


def _build_client(monkeypatch: pytest.MonkeyPatch, db_path) -> TestClient:
    monkeypatch.setattr(api, "schedule_cache_refresh", _noop_refresh, raising=False)
    monkeypatch.setattr(api.settings, "annotations_db_path", str(db_path), raising=False)
    _reset_annotation_state()
    return TestClient(api.app)


def test_annotations_persist_across_clients(tmp_path, monkeypatch):
    db_path = tmp_path / "annotations.db"

    with _build_client(monkeypatch, db_path) as client:
        response = client.put(
            "/annotations/AAPL",
            json={
                "note": "first-note",
                "client_timestamp": "2024-01-02T00:00:00+00:00",
            },
        )
        assert response.status_code == 200

    with _build_client(monkeypatch, db_path) as client:
        response = client.get("/annotations")
        assert response.status_code == 200
        payload = response.json()
        assert any(item["ticker"] == "AAPL" and item["note"] == "first-note" for item in payload)

        history = client.get("/annotations/history/AAPL")
        assert history.status_code == 200
        assert len(history.json()) == 1


def test_annotations_returns_503_on_store_failure(tmp_path, monkeypatch):
    db_path = tmp_path / "annotations.db"

    def _boom(self):
        raise OSError("boom")

    monkeypatch.setattr(api.AnnotationStore, "list_annotations", _boom, raising=False)

    with _build_client(monkeypatch, db_path) as client:
        response = client.get("/annotations")
        assert response.status_code == 503
