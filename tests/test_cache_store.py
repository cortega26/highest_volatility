import importlib
import os

import json

import pandas as pd
import pytest

from highest_volatility.cache import store


def make_df():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    return pd.DataFrame({"Adj Close": [1.0, 2.0]}, index=idx)


def test_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CACHE_ROOT", tmp_path)
    df = make_df()
    manifest = store.save_cache("ABC", "1d", df, "test")
    assert manifest.ticker == "ABC"
    assert manifest.rows == 2
    assert manifest.updated_at.endswith("Z")

    loaded_df, loaded_manifest = store.load_cached("ABC", "1d")
    assert loaded_manifest == manifest
    pd.testing.assert_frame_equal(loaded_df, df)

    with pytest.raises(ValueError):
        store.save_cache("DEF", "1d", df.iloc[0:0], "test")


def test_load_cached_removes_outdated_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CACHE_ROOT", tmp_path)
    df = make_df()
    store.save_cache("OLD", "1d", df, "test")

    parquet_path = tmp_path / "1d" / "OLD.parquet"
    manifest_path = tmp_path / "1d" / "OLD.json"
    manifest_data = json.loads(manifest_path.read_text())
    outdated_version = max(store.CACHE_VERSION - 1, 0)
    manifest_data["version"] = outdated_version
    manifest_path.write_text(json.dumps(manifest_data))

    cached_df, manifest = store.load_cached("OLD", "1d")

    assert cached_df is None
    assert manifest is None
    assert not parquet_path.exists()
    assert not manifest_path.exists()


def test_load_cached_purges_corrupt_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CACHE_ROOT", tmp_path)
    df = make_df()
    store.save_cache("BROKEN", "1d", df, "test")

    parquet_path = tmp_path / "1d" / "BROKEN.parquet"
    manifest_path = tmp_path / "1d" / "BROKEN.json"
    manifest_path.write_text("{\n")

    cached_df, manifest = store.load_cached("BROKEN", "1d")

    assert cached_df is None
    assert manifest is None
    assert not parquet_path.exists()
    assert not manifest_path.exists()


def test_cache_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HV_CACHE_ROOT_BASE", str(tmp_path))
    template = (
        "%HV_CACHE_ROOT_BASE%/custom-cache"
        if os.name == "nt"
        else "$HV_CACHE_ROOT_BASE/custom-cache"
    )
    monkeypatch.setenv("HV_CACHE_ROOT", template)
    reloaded = importlib.reload(store)
    try:
        assert reloaded.CACHE_ROOT == tmp_path / "custom-cache"
    finally:
        monkeypatch.delenv("HV_CACHE_ROOT_BASE", raising=False)
        monkeypatch.delenv("HV_CACHE_ROOT", raising=False)
        importlib.reload(store)


def test_cache_root_env_rejects_unresolved_placeholder(monkeypatch):
    template = "%HV_CACHE_ROOT_MISSING%/cache" if os.name == "nt" else "$HV_CACHE_ROOT_MISSING/cache"
    monkeypatch.setenv("HV_CACHE_ROOT", template)
    with pytest.raises(ValueError):
        importlib.reload(store)
    monkeypatch.delenv("HV_CACHE_ROOT", raising=False)
    importlib.reload(store)
