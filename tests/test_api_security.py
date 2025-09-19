from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app as cache_app
from src.highest_volatility.app.api import app as hv_app


def test_prices_rejects_traversal() -> None:
    with TestClient(cache_app) as client:
        resp = client.get("/prices/AAPL", params={"interval": "../"})
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
