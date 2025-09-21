"""Resilience tests for the cache refresh scheduler."""

import pytest

from highest_volatility.errors import (
    ErrorCode,
    HVError,
    get_error_metrics,
    reset_error_metrics,
)
from highest_volatility.pipeline import cache_refresh


class StopScheduler(RuntimeError):
    """Sentinel exception used to terminate the infinite scheduler loop."""


@pytest.mark.asyncio
async def test_schedule_cache_refresh_recovers_from_ticker_failure(monkeypatch):
    """The scheduler should log failures and continue scheduling iterations."""

    reset_error_metrics()

    tickers = ["GOOD", "FAIL"]
    monkeypatch.setattr(
        cache_refresh,
        "_cached_tickers",
        lambda interval="1d": list(tickers),
    )

    failure_triggered = {"value": False}
    download_calls: list[str] = []

    def fake_download(tickers_list, lookback_days, *, interval, force_refresh):
        ticker = tickers_list[0]
        download_calls.append(ticker)
        if ticker == "FAIL" and not failure_triggered["value"]:
            failure_triggered["value"] = True
            raise HVError("boom", code=ErrorCode.DATA_SOURCE)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    logged_events: list[str] = []
    original_log_exception = cache_refresh.log_exception

    def fake_log_exception(logger, error, *, event, context=None):
        logged_events.append(event)
        original_log_exception(logger, error, event=event, context=context)

    sleeps = 0

    async def fake_sleep(delay):
        nonlocal sleeps
        sleeps += 1
        if sleeps >= 2:
            raise StopScheduler

    monkeypatch.setattr(cache_refresh, "download_price_history", fake_download)
    monkeypatch.setattr(cache_refresh.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(cache_refresh, "log_exception", fake_log_exception)
    monkeypatch.setattr(cache_refresh.asyncio, "sleep", fake_sleep)

    with pytest.raises(StopScheduler):
        await cache_refresh.schedule_cache_refresh(
            interval="1d",
            lookback_days=30,
            delay=0.0,
        )

    assert failure_triggered["value"] is True
    assert download_calls == ["GOOD", "FAIL", "GOOD", "FAIL"]
    assert logged_events == ["cache_refresh_failed"]
    metrics = get_error_metrics()
    assert metrics.get(ErrorCode.DATA_SOURCE.value) == 1
