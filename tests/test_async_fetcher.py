import asyncio
from datetime import date
from typing import List

import aiohttp
import pandas as pd
import pytest

from cache import store
from ingest.async_fetch_prices import AsyncPriceFetcher
from ingest.fetch_async import fetch_many_async
from datasource.base_async import AsyncDataSource
from datasource.yahoo_http_async import YahooHTTPAsyncDataSource


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


@pytest.mark.asyncio
async def test_http_async_get_prices(monkeypatch):
    FAKE_JSON = {
        "chart": {
            "result": [
                {
                    "timestamp": [1577836800, 1577923200, 1578009600],
                    "indicators": {"adjclose": [{"adjclose": [1.0, 2.0, 3.0]}]},
                }
            ]
        }
    }

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        async def json(self):
            return self._data

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            return FakeResponse(FAKE_JSON)

    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: FakeSession())

    ds = YahooHTTPAsyncDataSource()
    df = await ds.get_prices("TEST", date(2020, 1, 1), date(2020, 1, 3), "1d")
    assert list(df["Adj Close"]) == [1.0, 2.0, 3.0]
    assert df.index[0].year == 2020


@pytest.mark.asyncio
async def test_http_async_validate(monkeypatch):
    ds = YahooHTTPAsyncDataSource()

    async def fake_get(*args, **kwargs):
        idx = pd.date_range("2020-01-01", periods=1)
        return pd.DataFrame({"Adj Close": [1.0]}, index=idx)

    monkeypatch.setattr(ds, "get_prices", fake_get)
    assert await ds.validate_ticker("AAA")

    async def fake_fail(*args, **kwargs):
        raise ValueError("bad")

    monkeypatch.setattr(ds, "get_prices", fake_fail)
    assert not await ds.validate_ticker("AAA")
