import pandas as pd
import pytest

from cache import store
from highest_volatility.pipeline import validate_cache


def _make_df():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    return pd.DataFrame({"Adj Close": [1.0, 2.0, 3.0]}, index=idx)


def _make_manifest(df):
    return store.Manifest(
        ticker="ABC",
        interval="1d",
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        rows=len(df),
        source="test",
        version=1,
        updated_at="2020-01-01T00:00:00Z",
    )


def test_validate_cache_ok():
    df = _make_df()
    manifest = _make_manifest(df)
    validate_cache(df, manifest)  # should not raise


def test_validate_cache_nan():
    df = _make_df()
    df.iloc[1, 0] = None
    manifest = _make_manifest(df)
    with pytest.raises(ValueError):
        validate_cache(df, manifest)


def test_validate_cache_gap():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"])
    df = pd.DataFrame({"Adj Close": [1.0, 2.0, 4.0, 5.0]}, index=idx)
    manifest = store.Manifest(
        ticker="ABC",
        interval="1d",
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        rows=len(df),
        source="test",
        version=1,
        updated_at="2020-01-01T00:00:00Z",
    )
    with pytest.raises(ValueError):
        validate_cache(df, manifest)


def test_validate_cache_range_mismatch():
    df = _make_df()
    manifest = store.Manifest(
        ticker="ABC",
        interval="1d",
        start="2020-01-01",
        end="2020-01-02",
        rows=len(df),
        source="test",
        version=1,
        updated_at="2020-01-01T00:00:00Z",
    )
    with pytest.raises(ValueError):
        validate_cache(df, manifest)


def test_save_cache_uses_validator(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CACHE_ROOT", tmp_path)
    df = _make_df()
    df.iloc[1, 0] = None
    with pytest.raises(ValueError):
        store.save_cache("ABC", "1d", df, "test")
    assert not (tmp_path / "1d" / "ABC.parquet").exists()
