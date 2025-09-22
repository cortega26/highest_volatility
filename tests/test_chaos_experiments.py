"""Chaos experiments ensuring graceful degradation under infrastructure faults."""

from __future__ import annotations

import asyncio
import time

import pandas as pd
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from redis.exceptions import ConnectionError as RedisConnectionError

from highest_volatility.app import api as hv_api
from highest_volatility.errors import (
    DataSourceError,
    ErrorCode,
    get_error_metrics,
    reset_error_metrics,
)
from highest_volatility.pipeline import cache_refresh


API_SLO_SECONDS = 0.5
pytestmark = pytest.mark.chaos


@pytest.fixture(autouse=True)
def _stub_schedule(monkeypatch):
    """Prevent the real refresh scheduler from running during the tests."""

    async def _noop_schedule(*args, **kwargs):
        await asyncio.sleep(0)

    monkeypatch.setattr(hv_api, "schedule_cache_refresh", _noop_schedule)


@pytest_asyncio.fixture(autouse=True)
async def _clear_cache_backend():
    """Ensure ``FastAPICache`` does not leak state across tests."""

    try:
        await FastAPICache.clear()
    except AssertionError:
        pass
    yield
    try:
        await FastAPICache.clear()
    except AssertionError:
        pass


def _stub_prices_dataframe() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5)
    return pd.DataFrame({"Close": [1, 1.2, 1.3, 1.5, 1.4]}, index=idx)


def test_rest_api_survives_redis_outage(monkeypatch):
    """The API should stay responsive if Redis is unavailable on startup."""

    reset_error_metrics()

    class _BrokenRedis:
        async def ping(self):
            raise RedisConnectionError("redis offline")

    monkeypatch.setattr(hv_api.redis, "from_url", lambda *a, **k: _BrokenRedis())
    monkeypatch.setattr(
        "highest_volatility.app.api.download_price_history",
        lambda *a, **k: _stub_prices_dataframe(),
    )

    with TestClient(hv_api.app) as client:
        start = time.perf_counter()
        response = client.get("/prices", params={"tickers": "AAPL", "lookback_days": 30})
        latency = time.perf_counter() - start
        assert hv_api.app.state.cache_health == "degraded"

    assert response.status_code == 200
    assert latency <= API_SLO_SECONDS
    backend = FastAPICache.get_backend()
    assert isinstance(backend, InMemoryBackend)


def test_prices_endpoint_handles_datasource_outage(monkeypatch):
    """Even during upstream failures the API should respond quickly with 502."""

    reset_error_metrics()

    def _boom(*args, **kwargs):
        raise DataSourceError("feed unavailable", context={"tickers": args[0]})

    monkeypatch.setattr(
        "highest_volatility.app.api.download_price_history",
        _boom,
    )

    with TestClient(hv_api.app) as client:
        start = time.perf_counter()
        response = client.get("/prices", params={"tickers": "FAIL", "lookback_days": 30})
        latency = time.perf_counter() - start

    assert response.status_code == 502
    assert latency <= API_SLO_SECONDS
    metrics = get_error_metrics()
    assert metrics.get(ErrorCode.DATA_SOURCE.value, 0) >= 1


@pytest.mark.asyncio
async def test_cache_refresh_cancellation_is_not_logged_as_failure(monkeypatch, caplog):
    """Cancelling the refresh loop should not emit failure events."""

    caplog.clear()
    caplog.set_level("ERROR", logger="highest_volatility.pipeline.cache_refresh")

    async def _no_op_refresh(*args, **kwargs):
        await asyncio.sleep(0)

    async def _cancel_sleep(delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(cache_refresh, "refresh_cached_prices", _no_op_refresh)
    monkeypatch.setattr(cache_refresh.asyncio, "sleep", _cancel_sleep)

    with pytest.raises(asyncio.CancelledError):
        await cache_refresh.schedule_cache_refresh(delay=5)

    failure_events = [rec for rec in caplog.records if "cache_refresh_iteration_failed" in rec.message]
    assert not failure_events
