from datetime import date
import asyncio
from typing import List

import pandas as pd

from cache import store
from ingest.async_fetch_prices import AsyncPriceFetcher
from ingest.fetch_async import fetch_many_async
from datasource.base_async import AsyncDataSource


class FakeAsyncDataSource(AsyncDataSource):
    def __init__(self):
        self.calls: List[tuple[str, date, date, str]] = []

    async def get_prices(self, ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
        self.calls.append((ticker, start, end, interval))
        idx = pd.date_range(start, end, freq="D")
        await asyncio.sleep(0)
        return pd.DataFrame({"Adj Close": range(len(idx))}, index=idx)

    async def validate_ticker(self, ticker: str) -> bool:
        return True


def test_async_incremental_and_force_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr("ingest.async_fetch_prices.full_backfill_start", lambda interval, today=None: date(2020, 1, 1))

    ds = FakeAsyncDataSource()
    fetcher = AsyncPriceFetcher(ds, throttle=0)

    class D1(date):
        @classmethod
        def today(cls):
            return date(2020, 1, 5)

    monkeypatch.setattr("ingest.async_fetch_prices.date", D1)
    df1 = asyncio.run(fetcher.fetch_one("ABC", "1d"))
    assert ds.calls[0][1] == date(2020, 1, 1)

    class D2(date):
        @classmethod
        def today(cls):
            return date(2020, 1, 6)

    monkeypatch.setattr("ingest.async_fetch_prices.date", D2)
    df2 = asyncio.run(fetcher.fetch_one("ABC", "1d"))
    assert ds.calls[1][1] == date(2020, 1, 6)
    assert len(df2) == len(df1) + 1

    asyncio.run(fetcher.fetch_one("ABC", "1d", force_refresh=True))
    assert ds.calls[2][1] == date(2020, 1, 1)

def test_fetch_many_async(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr("ingest.async_fetch_prices.full_backfill_start", lambda interval, today=None: date(2020, 1, 1))

    ds = FakeAsyncDataSource()
    fetcher = AsyncPriceFetcher(ds, throttle=0)

    class D(date):
        @classmethod
        def today(cls):
            return date(2020, 1, 5)

    monkeypatch.setattr("ingest.async_fetch_prices.date", D)
    res = asyncio.run(fetch_many_async(fetcher, ["AAA", "BBB"], "1d", max_concurrency=2))
    assert set(res.keys()) == {"AAA", "BBB"}
    assert all(isinstance(df, pd.DataFrame) for df in res.values())
