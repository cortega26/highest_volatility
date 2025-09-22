from __future__ import annotations

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


def test_prices_rejects_traversal() -> None:
    with TestClient(cache_app) as client:
        resp = client.get("/prices", params={"tickers": "AAPL", "interval": "../"})
        assert resp.status_code == 400
        assert "invalid" in resp.json()["detail"].lower()
        for header in (
            "strict-transport-security",
            "x-content-type-options",
            "x-frame-options",
            "referrer-policy",
        ):
            assert header in resp.headers


def test_metrics_rejects_header_injection() -> None:
    with TestClient(hv_app) as client:
        resp = client.get("/metrics", params={"tickers": "AAPL\r\nX", "metric": "cc_vol"})
        assert resp.status_code == 400
        for header in (
            "strict-transport-security",
            "x-content-type-options",
            "x-frame-options",
            "referrer-policy",
        ):
            assert header in resp.headers


def test_prices_rejects_excessive_lookback(monkeypatch: pytest.MonkeyPatch) -> None:
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
        )
        assert resp.status_code == 400


def test_metrics_rejects_excessive_min_days(monkeypatch: pytest.MonkeyPatch) -> None:
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
        )
        assert resp.status_code == 400


def test_prices_rejects_excessive_ticker_count(monkeypatch: pytest.MonkeyPatch) -> None:
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
        )
        assert resp.status_code == 400
