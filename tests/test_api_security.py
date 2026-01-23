from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from highest_volatility.api import app as cache_app
from highest_volatility.app.api import app as hv_app


def test_highest_volatility_api_reexports_main_app() -> None:
    """Ensure ``highest_volatility.api`` exposes the production FastAPI app."""

    assert cache_app is hv_app
    routes = {route.path for route in cache_app.routes}
    assert "/metrics" in routes
    assert "/prices" in routes


def test_prices_rejects_traversal(auth_headers) -> None:
    with TestClient(cache_app) as client:
        resp = client.get(
            "/prices",
            params={"tickers": "AAPL", "interval": "../"},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "invalid" in resp.json()["detail"].lower()
        for header in (
            "strict-transport-security",
            "x-content-type-options",
            "x-frame-options",
            "referrer-policy",
        ):
            assert header in resp.headers


def test_metrics_rejects_header_injection(auth_headers) -> None:
    with TestClient(hv_app) as client:
        resp = client.get(
            "/metrics",
            params={"tickers": "AAPL\r\nX", "metric": "cc_vol"},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        for header in (
            "strict-transport-security",
            "x-content-type-options",
            "x-frame-options",
            "referrer-policy",
        ):
            assert header in resp.headers


def test_prices_rejects_excessive_lookback(
    monkeypatch: pytest.MonkeyPatch, auth_headers
) -> None:
    monkeypatch.setattr(
        "highest_volatility.app.api.download_price_history",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("download_price_history should not be invoked")
        ),
    )
    with TestClient(hv_app) as client:
        resp = client.get(
            "/prices",
            params={"tickers": "AAPL", "lookback_days": 5000},
            headers=auth_headers,
        )
        assert resp.status_code == 400


def test_metrics_rejects_excessive_min_days(
    monkeypatch: pytest.MonkeyPatch, auth_headers
) -> None:
    monkeypatch.setattr(
        "highest_volatility.app.api.download_price_history",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("download_price_history should not be invoked")
        ),
    )
    with TestClient(hv_app) as client:
        resp = client.get(
            "/metrics",
            params={"tickers": "AAPL", "metric": "cc_vol", "min_days": 1000},
            headers=auth_headers,
        )
        assert resp.status_code == 400


def test_prices_rejects_excessive_ticker_count(
    monkeypatch: pytest.MonkeyPatch, auth_headers
) -> None:
    monkeypatch.setattr(
        "highest_volatility.app.api.download_price_history",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("download_price_history should not be invoked")
        ),
    )
    tickers = ",".join(f"T{i}" for i in range(101))
    with TestClient(hv_app) as client:
        resp = client.get(
            "/prices",
            params={"tickers": tickers},
            headers=auth_headers,
        )
        assert resp.status_code == 400


def test_api_key_required_for_prices(monkeypatch: pytest.MonkeyPatch, auth_headers) -> None:
    def _fake_prices(*_args, **_kwargs):
        index = pd.date_range("2024-01-01", periods=1)
        return pd.DataFrame({"Close": [1.0]}, index=index)

    monkeypatch.setattr(
        "highest_volatility.app.api.download_price_history",
        _fake_prices,
    )

    with TestClient(hv_app) as client:
        missing = client.get("/prices", params={"tickers": "AAPL", "lookback_days": 30})
        assert missing.status_code == 401

        bad = client.get(
            "/prices",
            params={"tickers": "AAPL", "lookback_days": 30},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert bad.status_code == 401

        ok = client.get(
            "/prices",
            params={"tickers": "AAPL", "lookback_days": 30},
            headers=auth_headers,
        )
        assert ok.status_code == 200
