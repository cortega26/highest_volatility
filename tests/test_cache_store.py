import pandas as pd
import pytest

from cache import store


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
