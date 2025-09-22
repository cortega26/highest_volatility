from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from highest_volatility.ingest import prices
from highest_volatility.ingest import downloaders


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


def test_batch_mode_honors_use_cache(monkeypatch):
    today = date(2020, 1, 10)
    _setup_time(monkeypatch, today)
    expected_start = date(2020, 1, 1)
    starts, saved, calls = _install_cache_stubs(monkeypatch, expected_start, "1d", "D")

    df = prices.download_price_history(
        ["AAA"],
        5,
        matrix_mode="batch",
        use_cache=True,
        max_workers=1,
    )

    assert calls == [("1d", today)]
    assert [pd.Timestamp(s).date() for s in starts] == [expected_start]
    assert ("AAA", "1d") in saved
    assert not df.empty


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


def test_incremental_intraday_fetch_extends_same_day(monkeypatch):
    today = date(2020, 1, 10)
    _setup_time(monkeypatch, today)

    cached = pd.DataFrame(
        {"Adj Close": [101.0, 102.0]},
        index=pd.to_datetime([f"{today} 09:30", f"{today} 09:31"]),
    )

    starts: list[Any] = []
    saved: dict[tuple[str, str], pd.DataFrame] = {}
    full_calls: list[tuple[str, date | None]] = []

    def fake_full_backfill_start(interval_arg: str, today_arg: date | None = None) -> date:
        full_calls.append((interval_arg, today_arg))
        assert today_arg is not None
        return today_arg

    def fake_load_cached(ticker: str, interval_arg: str) -> tuple[pd.DataFrame | None, Any]:
        assert ticker == "AAA"
        assert interval_arg == "1m"
        return cached.copy(), None

    def fake_save_cache(ticker: str, interval_arg: str, df: pd.DataFrame, source: str, **_: Any) -> None:
        assert ticker == "AAA"
        assert interval_arg == "1m"
        saved[(ticker, interval_arg)] = df.copy()

    def fake_download(*_, **kwargs: Any) -> pd.DataFrame:
        starts.append(kwargs["start"])
        idx = pd.to_datetime([f"{today} 09:30", f"{today} 09:31", f"{today} 09:32"])
        return pd.DataFrame({"Adj Close": [111.0, 112.0, 113.0]}, index=idx)

    monkeypatch.setattr(prices, "full_backfill_start", fake_full_backfill_start)
    monkeypatch.setattr(prices, "load_cached", fake_load_cached)
    monkeypatch.setattr(prices, "save_cache", fake_save_cache)
    monkeypatch.setattr(prices.yf, "download", fake_download)

    df = prices.download_price_history(
        ["AAA"],
        5,
        matrix_mode="cache",
        use_cache=True,
        max_workers=1,
        prepost=True,
        interval="1m",
    )

    assert full_calls == []
    assert starts == [today]
    saved_frame = saved[("AAA", "1m")]
    assert list(saved_frame.index) == list(pd.to_datetime([f"{today} 09:30", f"{today} 09:31", f"{today} 09:32"]))
    assert saved_frame.loc[pd.Timestamp(f"{today} 09:32"), "Adj Close"] == 113.0
    assert ("Adj Close", "AAA") in df.columns
    result_series = df["Adj Close"]["AAA"]
    assert pd.Timestamp(f"{today} 09:32") in result_series.index


def test_plan_cache_fetch_force_refresh():
    today = date(2024, 5, 1)

    def fake_full_backfill(interval: str, today: date | None = None) -> date:
        return date(2024, 1, 1)

    cached_df = pd.DataFrame({"Adj Close": [1.0]}, index=pd.date_range("2024-04-30", periods=1))

    def fake_load_cached(ticker: str, interval: str):
        assert interval == "1d"
        return cached_df.copy(), None

    plan = downloaders.plan_cache_fetch(
        tickers=["AAA"],
        interval="1d",
        start_date=date(2024, 4, 1),
        end_date=today,
        force_refresh=True,
        load_cached=fake_load_cached,
        full_backfill_start_fn=fake_full_backfill,
    )

    assert plan.to_fetch == ["AAA"]
    assert plan.fetch_starts["AAA"] == date(2024, 1, 1)


def test_plan_fingerprint_refresh_detects_clusters():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    base = pd.DataFrame({"Adj Close": range(10)}, index=idx)
    frames = {
        "AAA": base.copy(),
        "BBB": base.copy(),
        "CCC": base.copy(),
        "DDD": pd.DataFrame({"Adj Close": range(10, 20)}, index=idx),
    }

    plan = downloaders.plan_fingerprint_refresh(frames, lookback_days=40)
    assert set(plan.suspects) == {"AAA", "BBB", "CCC"}
    assert plan.field == "Adj Close"
