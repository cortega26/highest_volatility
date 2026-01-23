import json

import pandas as pd

from highest_volatility.cache import store


def test_load_cached_hydrates_from_api(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CACHE_ROOT", tmp_path)
    monkeypatch.setenv("HV_API_BASE_URL", "https://example.test")
    monkeypatch.setenv("GITHUB_ACTIONS", "false")

    index = pd.to_datetime(["2020-01-02", "2020-01-03"])
    multi = pd.DataFrame(
        [[1.0, 2.0], [1.5, 2.5]],
        index=index,
        columns=pd.MultiIndex.from_tuples(
            [("Open", "AAPL"), ("Close", "AAPL")]
        ),
    )
    payload = json.loads(multi.to_json(orient="split", date_format="iso"))
    expected = pd.DataFrame({"Open": [1.0, 1.5], "Close": [2.0, 2.5]}, index=index)

    class Response:
        status_code = 200

        def json(self):
            return payload

    def fake_get(url, params=None, headers=None, timeout=10):
        assert url == "https://example.test/prices"
        assert params == {"tickers": "AAPL", "interval": "1d"}
        assert headers is None
        return Response()

    monkeypatch.setattr(store.requests, "get", fake_get)

    loaded_df, manifest = store.load_cached("AAPL", "1d")

    assert manifest is not None
    pd.testing.assert_frame_equal(loaded_df, expected)


def test_load_cached_hydration_skips_on_http_error(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CACHE_ROOT", tmp_path)
    monkeypatch.setenv("HV_API_BASE_URL", "https://example.test")
    monkeypatch.setenv("GITHUB_ACTIONS", "false")

    class Response:
        status_code = 500

        def json(self):
            return {}

    def fake_get(url, params=None, headers=None, timeout=10):
        return Response()

    monkeypatch.setattr(store.requests, "get", fake_get)

    loaded_df, manifest = store.load_cached("AAPL", "1d")

    assert loaded_df is None
    assert manifest is None
