import asyncio
import threading

import pandas as pd
import pytest

from highest_volatility.ingest import prices
from datetime import date, datetime
from highest_volatility.config.interval_policy import full_backfill_start
from highest_volatility.ingest import downloaders


class FakeAsyncDS:
    def __init__(self):
        self.calls: list[tuple[str, date, date, str]] = []

    async def get_prices(self, ticker, start, end, interval):
        self.calls.append((ticker, start, end, interval))
        idx = pd.date_range(start, periods=3)
        return pd.DataFrame({"Adj Close": [1, 2, 3]}, index=idx)


def test_download_price_history_async(monkeypatch):
    class D(date):
        @classmethod
        def today(cls):
            return date(2020, 1, 6)

    class DT(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return datetime(2020, 1, 6, tzinfo=tz)
            return datetime(2020, 1, 6)

        @classmethod
        def utcnow(cls):
            return datetime(2020, 1, 6)

    fake_ds = FakeAsyncDS()

    def fake_full_backfill(interval, today=None):
        assert interval == "1d"
        assert today == date(2020, 1, 6)
        return date(2019, 12, 30)

    monkeypatch.setattr(prices, "date", D)
    monkeypatch.setattr(prices, "datetime", DT)
    monkeypatch.setattr(prices, "full_backfill_start", fake_full_backfill)
    monkeypatch.setattr(prices, "YahooAsyncDataSource", lambda: fake_ds)

    df = prices.download_price_history(["AAA", "BBB"], 5, matrix_mode="async", use_cache=False)
    assert set(df.columns.get_level_values(1)) == {"AAA", "BBB"}
    assert "Adj Close" in df.columns.get_level_values(0)
    assert all(call[1] == date(2019, 12, 30) for call in fake_ds.calls)


def test_download_price_history_async_uses_60m_alias(monkeypatch):
    class FrozenDate(date):
        @classmethod
        def today(cls):
            return date(2024, 1, 1)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return datetime(2024, 1, 1, tzinfo=tz)
            return datetime(2024, 1, 1)

        @classmethod
        def utcnow(cls):
            return datetime(2024, 1, 1)

    class IntradayFakeAsyncDS:
        def __init__(self):
            self.calls: list[tuple[str, date, date, str]] = []

        async def get_prices(self, ticker, start, end, interval):
            self.calls.append((ticker, start, end, interval))
            idx = pd.date_range(start, periods=2, freq="h")
            return pd.DataFrame({"Adj Close": [1.0, 2.0]}, index=idx)

    fake_ds = IntradayFakeAsyncDS()

    monkeypatch.setattr(prices, "date", FrozenDate)
    monkeypatch.setattr(prices, "datetime", FrozenDateTime)
    monkeypatch.setattr("highest_volatility.config.interval_policy.date", FrozenDate)
    monkeypatch.setattr(prices, "YahooAsyncDataSource", lambda: fake_ds)

    df = prices.download_price_history(
        ["AAA"],
        400,
        matrix_mode="async",
        use_cache=False,
        interval="60m",
    )

    assert fake_ds.calls, "datasource should be invoked"
    _, start_arg, end_arg, interval_arg = fake_ds.calls[0]
    expected_start = full_backfill_start("1h", today=FrozenDate.today())
    assert start_arg == expected_start
    assert end_arg == FrozenDate.today()
    assert interval_arg == "60m"
    assert not df.empty


@pytest.mark.asyncio
async def test_download_price_history_async_respects_running_loop(monkeypatch):
    class FrozenDate(date):
        @classmethod
        def today(cls):
            return date(2021, 6, 1)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2021, 6, 1, tzinfo=tz)

    async def fake_download(async_request):
        idx = pd.date_range(datetime(2021, 5, 30), periods=2)
        frames = {"AAA": pd.DataFrame({"Adj Close": [10.0, 11.0]}, index=idx)}
        return downloaders.AsyncDownloadResult(frames=frames)

    monkeypatch.setattr(prices, "date", FrozenDate)
    monkeypatch.setattr(prices, "datetime", FrozenDateTime)
    monkeypatch.setattr(prices, "full_backfill_start", lambda *_, **__: date(2021, 5, 1))
    monkeypatch.setattr(prices, "YahooAsyncDataSource", None)
    monkeypatch.setattr(downloaders, "download_async", fake_download)

    run_threads: list[str] = []
    original_run = asyncio.run

    def tracking_run(coro):
        run_threads.append(threading.current_thread().name)
        return original_run(coro)

    monkeypatch.setattr(prices.asyncio, "run", tracking_run)

    caller_thread = threading.current_thread().name
    df = prices.download_price_history(["AAA"], 30, matrix_mode="async", use_cache=False)

    assert ("Adj Close", "AAA") in df.columns
    assert run_threads, "asyncio.run should be invoked for async downloads"
    assert all(name != caller_thread for name in run_threads), "asyncio.run must execute off the running loop thread"

def test_async_download_result_dataframe():
    frames = {
        "AAA": pd.DataFrame({"Adj Close": [1.0]}, index=pd.to_datetime(["2020-01-02"])),
        "BBB": pd.DataFrame({"Adj Close": [2.0]}, index=pd.to_datetime(["2020-01-03"])),
    }
    result = downloaders.AsyncDownloadResult(frames=frames)
    combined = result.to_dataframe(trim_start=pd.Timestamp("2020-01-02"))
    assert ("Adj Close", "AAA") in combined.columns
    assert ("Adj Close", "BBB") in combined.columns

