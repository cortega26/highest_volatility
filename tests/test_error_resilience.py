from __future__ import annotations

import json
import logging

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from highest_volatility.ingest.fetch_async import fetch_many_async
from highest_volatility.app import api as hv_api
from highest_volatility.app.api import app as hv_app
from highest_volatility.errors import (
    DataSourceError,
    ErrorCode,
    get_error_metrics,
    reset_error_metrics,
)


class _StubFetcher:
    async def fetch_one(self, ticker: str, interval: str, *, force_refresh: bool = False):
        if ticker == "FAIL":
            raise DataSourceError("forced failure", context={"ticker": ticker})
        idx = pd.date_range("2020-01-01", periods=2)
        return pd.DataFrame({"Close": [1.0, 2.0]}, index=idx)


@pytest.mark.asyncio
async def test_fetch_many_async_records_structured_errors(caplog):
    reset_error_metrics()
    caplog.set_level(logging.ERROR)
    fetcher = _StubFetcher()

    results = await fetch_many_async(fetcher, ["OK", "FAIL"], "1d")

    assert "OK" in results and "FAIL" not in results
    metrics = get_error_metrics()
    assert metrics.get(ErrorCode.DATA_SOURCE.value, 0) >= 1
    payloads = [json.loads(rec.message) for rec in caplog.records if rec.levelname == "ERROR"]
    assert any(entry.get("event") == "async_fetch_failed" for entry in payloads)


def test_prices_endpoint_surfaces_datasource_error(monkeypatch, auth_headers):
    reset_error_metrics()

    def _boom(*args, **kwargs):
        raise DataSourceError("Upstream feed unavailable", context={"tickers": args[0]})

    monkeypatch.setattr(
        "highest_volatility.app.api.download_price_history",
        _boom,
    )

    with TestClient(hv_app) as client:
        response = client.get(
            "/prices",
            params={"tickers": "AAPL", "lookback_days": 30},
            headers=auth_headers,
        )

    assert response.status_code == 502
    assert "Upstream feed" in response.json()["detail"]
    metrics = get_error_metrics()
    assert metrics.get(ErrorCode.DATA_SOURCE.value, 0) >= 1


def test_metrics_endpoint_wraps_compute_error(monkeypatch, caplog, auth_headers):
    reset_error_metrics()
    caplog.set_level(logging.ERROR)

    def _mock_prices(*args, **kwargs):
        idx = pd.date_range("2020-01-01", periods=5)
        return pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)

    def _boom_metric(*args, **kwargs):
        raise RuntimeError("metric exploded")

    monkeypatch.setattr(
        "highest_volatility.app.api.download_price_history",
        _mock_prices,
    )
    monkeypatch.setitem(hv_api.METRIC_REGISTRY, "boom", _boom_metric)

    with TestClient(hv_app) as client:
        response = client.get(
            "/metrics",
            params={
                "tickers": "AAPL",
                "metric": "boom",
                "lookback_days": 30,
                "min_days": 30,
            },
            headers=auth_headers,
        )

    assert response.status_code == 500
    error_logs = [
        json.loads(record.message)
        for record in caplog.records
        if record.levelno >= logging.ERROR
    ]
    metrics_failures = [
        payload
        for payload in error_logs
        if payload.get("event") == "metrics_failure"
    ]
    assert metrics_failures, "expected metrics_failure log entry"
    assert metrics_failures[0]["error"]["code"] == ErrorCode.COMPUTE.value
