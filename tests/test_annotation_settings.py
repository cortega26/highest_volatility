"""Tests for annotation database settings wiring."""

from __future__ import annotations

import pytest

from highest_volatility.app.api import Settings
from highest_volatility.storage.annotation_store import AnnotationStore


def test_settings_accepts_annotations_db_env(tmp_path, monkeypatch):
    db_path = tmp_path / "annotations.db"
    monkeypatch.delenv("HV_ANNOTATIONS_DB_PATH", raising=False)
    monkeypatch.setenv("HV_ANNOTATIONS_DB", str(db_path))

    settings = Settings()

    assert settings.annotations_db_path == str(db_path)


def test_settings_accepts_legacy_annotations_db_path_env(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    monkeypatch.delenv("HV_ANNOTATIONS_DB", raising=False)
    monkeypatch.setenv("HV_ANNOTATIONS_DB_PATH", str(db_path))

    settings = Settings()

    assert settings.annotations_db_path == str(db_path)


def test_annotation_store_rejects_directory(tmp_path):
    with pytest.raises(ValueError):
        AnnotationStore(tmp_path)
