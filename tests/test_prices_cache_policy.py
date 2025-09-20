from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from highest_volatility.ingest import prices


def _setup_time(monkeypatch, today: date) -> None:
    class _Date(date):
        @classmethod
        def today(cls) -> date:  # type: ignore[override]
            return today

    class _DateTime(datetime):
        @classmethod
        def utcnow(cls) -> datetime:  # type: ignore[override]
            return datetime.combine(today, datetime.min.time())

    monkeypatch.setattr(prices, "date", _Date)
    monkeypatch.setattr(prices, "datetime", _DateTime)


def _install_cache_stubs(
    monkeypatch, expected_start: date, interval: str, freq: str
) -> tuple[list[Any], dict[tuple[str, str], pd.DataFrame], list[tuple[str, date | None]]]:
    starts: list[Any] = []
    saved: dict[tuple[str, str], pd.DataFrame] = {}
    full_backfill_calls: list[tuple[str, date | None]] = []

    def fake_full_backfill_start(interval_arg: str, today: date | None = None) -> date:
        assert interval_arg == interval
        assert today is not None
        full_backfill_calls.append((interval_arg, today))
        return expected_start

    def fake_load_cached(ticker: str, interval_arg: str) -> tuple[pd.DataFrame | None, Any]:
        assert interval_arg == interval
        return None, None

    def fake_save_cache(ticker: str, interval_arg: str, df: pd.DataFrame, source: str, **_: Any) -> None:
        assert interval_arg == interval
        saved[(ticker, interval_arg)] = df.copy()

    def fake_download(*_, **kwargs: Any) -> pd.DataFrame:
        starts.append(kwargs["start"])
        idx = pd.date_range(kwargs["start"], periods=3, freq=freq)
        return pd.DataFrame({"Adj Close": range(len(idx))}, index=idx)

    monkeypatch.setattr(prices, "full_backfill_start", fake_full_backfill_start)
    monkeypatch.setattr(prices, "load_cached", fake_load_cached)
    monkeypatch.setattr(prices, "save_cache", fake_save_cache)
    monkeypatch.setattr(prices.yf, "download", fake_download)

    return starts, saved, full_backfill_calls


def test_first_run_cache_uses_full_backfill_daily(monkeypatch):
    today = date(2020, 1, 10)
    _setup_time(monkeypatch, today)
    expected_start = date(2020, 1, 1)
    starts, saved, calls = _install_cache_stubs(monkeypatch, expected_start, "1d", "D")

    df = prices.download_price_history(["AAA"], 5, matrix_mode="cache", use_cache=True, max_workers=1)

    assert calls == [("1d", today)]
    assert [pd.Timestamp(s).date() for s in starts] == [expected_start]
    cached = saved[("AAA", "1d")]
    assert cached.index.min().date() == expected_start
    # Returned frame is trimmed to lookback window but should still include data
    assert not df.empty


def test_first_run_cache_uses_full_backfill_intraday(monkeypatch):
    today = date(2020, 1, 10)
    _setup_time(monkeypatch, today)
    expected_start = date(2020, 1, 9)
    starts, saved, calls = _install_cache_stubs(monkeypatch, expected_start, "1m", "min")

    df = prices.download_price_history(
        ["AAA"],
        5,
        matrix_mode="cache",
        use_cache=True,
        max_workers=1,
        prepost=True,
        interval="1m",
    )

    assert calls == [("1m", today)]
    assert [pd.Timestamp(s).date() for s in starts] == [expected_start]
    cached = saved[("AAA", "1m")]
    assert cached.index.min().date() == expected_start
    assert not df.empty
