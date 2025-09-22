import asyncio
from datetime import date
from typing import List

import aiohttp
from aiohttp.client_reqrep import RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
import pandas as pd
import pytest
from yarl import URL

from highest_volatility.cache import store
from highest_volatility.ingest.async_fetch_prices import AsyncPriceFetcher
from highest_volatility.ingest.fetch_async import fetch_many_async
from highest_volatility.datasource.base_async import AsyncDataSource
from highest_volatility.datasource.yahoo_async import YahooAsyncDataSource
from highest_volatility.datasource.yahoo_http_async import YahooHTTPAsyncDataSource
from highest_volatility.config.interval_policy import full_backfill_start


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
    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.full_backfill_start", lambda interval, today=None: date(2020, 1, 1))

    ds = FakeAsyncDataSource()
    fetcher = AsyncPriceFetcher(ds, throttle=0)

    saved: dict[tuple[str, str], pd.DataFrame] = {}

    def fake_save_cache(
        ticker: str, interval: str, df: pd.DataFrame, source: str, **kwargs
    ) -> None:
        saved[(ticker, interval)] = df.copy()

    def fake_load_cached(ticker: str, interval: str):
        return saved.get((ticker, interval)), None

    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.save_cache", fake_save_cache)
    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.load_cached", fake_load_cached)

    class D1(date):
        @classmethod
        def today(cls):
            return date(2020, 1, 5)

    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.date", D1)
    df1 = asyncio.run(fetcher.fetch_one("ABC", "1d"))
    assert ds.calls[0][1] == date(2020, 1, 1)

    class D2(date):
        @classmethod
        def today(cls):
            return date(2020, 1, 6)

    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.date", D2)
    df2 = asyncio.run(fetcher.fetch_one("ABC", "1d"))
    assert ds.calls[1][1] == date(2020, 1, 6)
    assert len(df2) == len(df1) + 1

    asyncio.run(fetcher.fetch_one("ABC", "1d", force_refresh=True))
    assert ds.calls[2][1] == date(2020, 1, 1)


@pytest.mark.asyncio
async def test_async_fetcher_drops_nan_rows_before_persist(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(
        "highest_volatility.ingest.async_fetch_prices.full_backfill_start",
        lambda interval, today=None: date(2020, 1, 1),
    )
    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.load_cached", lambda *_, **__: (None, None))

    class NaNDataSource(AsyncDataSource):
        async def get_prices(
            self, ticker: str, start: date, end: date, interval: str
        ) -> pd.DataFrame:
            await asyncio.sleep(0)
            idx = pd.date_range(start, periods=3, freq="D")
            return pd.DataFrame(
                {"Adj Close": [1.0, float("nan"), 2.0]},
                index=idx,
            )

        async def validate_ticker(self, ticker: str) -> bool:
            return True

    saved: dict[str, pd.DataFrame] = {}

    def fake_save_cache(
        ticker: str, interval: str, df: pd.DataFrame, source: str, **kwargs
    ) -> None:
        saved["df"] = df.copy()

    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.save_cache", fake_save_cache)

    class FrozenDate(date):
        @classmethod
        def today(cls) -> date:
            return date(2020, 1, 3)

    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.date", FrozenDate)

    fetcher = AsyncPriceFetcher(NaNDataSource(), throttle=0)

    result = await fetcher.fetch_one("XYZ", "1d")

    assert not result.isna().any().any(), "result should not contain NaN rows"
    assert len(result) == 2, "only rows with valid data should remain"
    assert "df" in saved, "sanitised frame should be persisted"
    persisted = saved["df"]
    assert not persisted.isna().any().any(), "persisted frame should be clean"
    pd.testing.assert_frame_equal(result, persisted)


@pytest.mark.asyncio
async def test_intraday_incremental_fetch_avoids_same_day_gap(monkeypatch):
    cached_idx = pd.to_datetime(
        ["2020-01-05 09:30", "2020-01-05 10:00"], format="%Y-%m-%d %H:%M"
    )
    cached_df = pd.DataFrame({"Adj Close": [100.0, 101.0]}, index=cached_idx)

    new_idx = pd.to_datetime(
        ["2020-01-05 10:30", "2020-01-05 11:00"], format="%Y-%m-%d %H:%M"
    )
    new_df = pd.DataFrame({"Adj Close": [102.0, 103.0]}, index=new_idx)

    class IntradayDataSource(AsyncDataSource):
        def __init__(self) -> None:
            self.calls: list[tuple[str, date, date, str]] = []

        async def get_prices(
            self, ticker: str, start: date, end: date, interval: str
        ) -> pd.DataFrame:
            self.calls.append((ticker, start, end, interval))
            await asyncio.sleep(0)
            return new_df.copy()

        async def validate_ticker(self, ticker: str) -> bool:
            return True

    ds = IntradayDataSource()
    fetcher = AsyncPriceFetcher(ds, throttle=0)

    monkeypatch.setattr(
        "highest_volatility.ingest.async_fetch_prices.load_cached", lambda *_, **__: (cached_df.copy(), None)
    )

    saved: dict[str, pd.DataFrame] = {}

    def fake_save_cache(
        ticker: str, interval: str, df: pd.DataFrame, source: str, **kwargs
    ) -> None:
        saved["df"] = df.copy()

    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.save_cache", fake_save_cache)

    class FrozenDate(date):
        @classmethod
        def today(cls) -> date:
            return date(2020, 1, 5)

    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.date", FrozenDate)

    result = await fetcher.fetch_one("ABC", "30m")

    assert ds.calls, "datasource should be queried for intraday increments"
    start_arg = ds.calls[0][1]
    assert start_arg <= FrozenDate.today()

    expected_index = pd.Index(list(cached_idx) + list(new_idx))
    pd.testing.assert_index_equal(result.index, expected_index)
    pd.testing.assert_index_equal(saved["df"].index, expected_index)


@pytest.mark.asyncio
async def test_async_fetcher_uses_60m_alias(monkeypatch):
    class FrozenDate(date):
        @classmethod
        def today(cls) -> date:
            return date(2024, 1, 1)

    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.date", FrozenDate)
    monkeypatch.setattr("highest_volatility.config.interval_policy.date", FrozenDate)

    class RecordingDataSource(AsyncDataSource):
        def __init__(self) -> None:
            self.calls: list[tuple[str, date, date, str]] = []

        async def get_prices(
            self, ticker: str, start: date, end: date, interval: str
        ) -> pd.DataFrame:
            self.calls.append((ticker, start, end, interval))
            await asyncio.sleep(0)
            idx = pd.date_range(start, periods=1, freq="h")
            return pd.DataFrame({"Adj Close": [1.0]}, index=idx)

        async def validate_ticker(self, ticker: str) -> bool:
            return True

    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.save_cache", lambda *_, **__: None)

    datasource = RecordingDataSource()
    fetcher = AsyncPriceFetcher(datasource, throttle=0)

    result = await fetcher.fetch_one("XYZ", "60m", force_refresh=True)

    assert datasource.calls, "datasource should be invoked for 60m interval"
    _, start_arg, end_arg, interval_arg = datasource.calls[0]
    expected_start = full_backfill_start("1h", today=FrozenDate.today())
    assert start_arg == expected_start
    assert end_arg == FrozenDate.today()
    assert interval_arg == "60m"
    assert not result.empty


def test_fetch_many_async(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.full_backfill_start", lambda interval, today=None: date(2020, 1, 1))

    ds = FakeAsyncDataSource()
    fetcher = AsyncPriceFetcher(ds, throttle=0)

    saved: dict[tuple[str, str], pd.DataFrame] = {}

    def fake_save_cache(
        ticker: str, interval: str, df: pd.DataFrame, source: str, **kwargs
    ) -> None:
        saved[(ticker, interval)] = df.copy()

    def fake_load_cached(ticker: str, interval: str):
        return saved.get((ticker, interval)), None

    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.save_cache", fake_save_cache)
    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.load_cached", fake_load_cached)

    class D(date):
        @classmethod
        def today(cls):
            return date(2020, 1, 5)

    monkeypatch.setattr("highest_volatility.ingest.async_fetch_prices.date", D)
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
    assert df.columns.tolist() == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert list(df["Adj Close"]) == [1.0, 2.0, 3.0]
    assert list(df["Close"]) == [1.0, 2.0, 3.0]
    assert df["Volume"].tolist() == [0, 0, 0]
    assert df[["Open", "High", "Low"]].eq(df["Adj Close"], axis=0).all().all()
    assert df.index[0].year == 2020


@pytest.mark.asyncio
async def test_http_async_get_prices_fill_adjclose_none(monkeypatch):
    timestamps = [1577836800, 1577923200, 1578009600, 1578096000]
    FAKE_JSON = {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "adjclose": [{"adjclose": [1.0, None, 3.0, None]}],
                        "quote": [{"close": [1.05, 2.05, 3.05, 4.05]}],
                    },
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
    df = await ds.get_prices("TEST", date(2020, 1, 1), date(2020, 1, 4), "1d")

    expected_adj = [1.0, 2.05, 3.0, 4.05]
    expected_close = [1.05, 2.05, 3.05, 4.05]
    assert list(df["Adj Close"]) == expected_adj
    assert list(df["Close"]) == expected_close
    assert df["Volume"].tolist() == [0, 0, 0, 0]
    assert not df["Adj Close"].isna().any()
    pd.testing.assert_index_equal(df.index, pd.to_datetime(timestamps, unit="s"))


@pytest.mark.asyncio
async def test_http_async_get_prices_skips_missing_positions_and_warns(monkeypatch, caplog):
    timestamps = [1577836800, 1577923200, 1578009600, 1578096000, 1578182400]
    FAKE_JSON = {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "adjclose": [
                            {"adjclose": [1.0, None, None, 4.0, None]}
                        ],
                        "quote": [
                            {"close": [1.05, 2.05, None, None, None]}
                        ],
                    },
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
    with caplog.at_level("WARNING"):
        df = await ds.get_prices("TEST", date(2020, 1, 1), date(2020, 1, 5), "1d")

    expected_index = pd.to_datetime(
        [timestamps[0], timestamps[1], timestamps[3]], unit="s"
    )
    expected_adj = [1.0, 2.05, 4.0]
    expected_close = [1.05, 2.05, 4.0]

    pd.testing.assert_index_equal(df.index, expected_index)
    assert list(df["Adj Close"]) == expected_adj
    assert list(df["Close"]) == expected_close
    assert df["Volume"].tolist() == [0, 0, 0]
    assert "Dropped 2 missing Yahoo price rows" in caplog.text


@pytest.mark.asyncio
async def test_http_async_get_prices_raises_when_all_prices_missing(monkeypatch):
    timestamps = [1577836800, 1577923200, 1578009600]
    FAKE_JSON = {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "adjclose": [{"adjclose": [None, None, None]}],
                        "quote": [{"close": [None, None, None]}],
                    },
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

    with pytest.raises(ValueError) as excinfo:
        await ds.get_prices("TEST", date(2020, 1, 1), date(2020, 1, 3), "1d")

    message = str(excinfo.value)
    assert "Missing adjclose/close data" in message
    for ts in pd.to_datetime(timestamps, unit="s", utc=True):
        assert ts.isoformat() in message


@pytest.mark.asyncio
async def test_http_async_intraday_range_prevents_client_response_error(monkeypatch):
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

    captured: dict[str, dict[str, str | int]] = {}

    class FakeResponse:
        def __init__(self, params):
            self._params = params

        def raise_for_status(self):
            if "range" not in self._params:
                request_info = RequestInfo(
                    URL("https://example.test"),
                    "GET",
                    headers=CIMultiDictProxy(CIMultiDict()),
                    real_url=URL("https://example.test"),
                )
                raise aiohttp.ClientResponseError(
                    request_info,
                    (),
                    status=422,
                    message="Missing range parameter",
                )

        async def json(self):
            return FAKE_JSON

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
            captured["last"] = params or {}
            return FakeResponse(params or {})

    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: FakeSession())

    ds = YahooHTTPAsyncDataSource()
    df = await ds.get_prices("TEST", date(2020, 1, 1), date(2020, 3, 1), "60m")

    assert not df.empty
    params = captured.get("last", {})
    assert "range" in params
    assert "period1" not in params


@pytest.mark.asyncio
async def test_yahoo_async_get_prices_falls_back_to_close(monkeypatch):
    FAKE_JSON = {
        "chart": {
            "result": [
                {
                    "timestamp": [1577836800, 1577923200, 1578009600],
                    "indicators": {"quote": [{"close": [10.0, 11.0, 12.0]}]},
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

    ds = YahooAsyncDataSource()
    df = await ds.get_prices("TEST", date(2020, 1, 1), date(2020, 1, 3), "1d")
    assert df.columns.tolist() == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert list(df["Adj Close"]) == [10.0, 11.0, 12.0]
    pd.testing.assert_series_equal(df["Adj Close"], df["Close"], check_names=False)
    assert df["Volume"].tolist() == [0, 0, 0]
    assert df[["Open", "High", "Low"]].eq(df["Adj Close"], axis=0).all().all()
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
