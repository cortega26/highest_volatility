"""Tests for cache refresh across intervals.

Run with:
    pytest tests/test_refresh_cache_all_intervals.py
"""

from __future__ import annotations

import pandas as pd

from scripts.refresh_cache_all_intervals import refresh_intervals


def _sample_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    return pd.DataFrame({"Close": [100.0, 101.0]}, index=index)


def test_refresh_intervals_reports_no_failures() -> None:
    tickers = ["AAPL", "MSFT"]
    calls: list[str] = []

    def fake_download(tickers, lookback_days, *, interval, **_kwargs):
        calls.append(f"{interval}:{lookback_days}:{len(tickers)}")

    def fake_load_cached(ticker, interval):
        return _sample_frame(), object()

    failures = refresh_intervals(
        tickers,
        ["1d"],
        lookback_days_daily=365,
        lookback_days_intraday=30,
        prepost=False,
        force_refresh=False,
        max_workers=1,
        chunk_sleep=0.0,
        max_retries=1,
        download_fn=fake_download,
        load_cached_fn=fake_load_cached,
    )

    assert failures == []
    assert calls == ["1d:365:2"]


def test_refresh_intervals_reports_missing_cache() -> None:
    tickers = ["AAPL", "MSFT"]

    def fake_download(*_args, **_kwargs):
        return None

    def fake_load_cached(ticker, interval):
        if ticker == "MSFT":
            return None, None
        return _sample_frame(), object()

    failures = refresh_intervals(
        tickers,
        ["1d"],
        lookback_days_daily=365,
        lookback_days_intraday=30,
        prepost=False,
        force_refresh=False,
        max_workers=1,
        chunk_sleep=0.0,
        max_retries=1,
        download_fn=fake_download,
        load_cached_fn=fake_load_cached,
    )

    assert len(failures) == 1
    assert failures[0].ticker == "MSFT"
    assert failures[0].reason == "missing_cache"
