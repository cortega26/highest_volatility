from datetime import date
from typing import List

import pandas as pd

from src.cache import store
from datasource.base import DataSource
from ingest.fetch_prices import PriceFetcher
from src.config.interval_policy import full_backfill_start


class FakeDataSource(DataSource):
    def __init__(self):
        self.calls: List[tuple[str, date, date, str]] = []

    def get_prices(self, ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
        self.calls.append((ticker, start, end, interval))
        idx = pd.date_range(start, end, freq="D")
        return pd.DataFrame({"Adj Close": range(len(idx))}, index=idx)

    def validate_ticker(self, ticker: str) -> bool:
        return True


def test_incremental_and_force_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr("ingest.fetch_prices.full_backfill_start", lambda interval, today=None: date(2020, 1, 1))

    ds = FakeDataSource()
    fetcher = PriceFetcher(ds, throttle=0)

    saved: dict[tuple[str, str], pd.DataFrame] = {}

    def fake_save_cache(
        ticker: str, interval: str, df: pd.DataFrame, source: str, **kwargs
    ) -> None:
        saved[(ticker, interval)] = df.copy()

    def fake_load_cached(ticker: str, interval: str):
        return saved.get((ticker, interval)), None

    monkeypatch.setattr("ingest.fetch_prices.save_cache", fake_save_cache)
    monkeypatch.setattr("ingest.fetch_prices.load_cached", fake_load_cached)

    class D1(date):
        @classmethod
        def today(cls):
            return date(2020, 1, 5)

    monkeypatch.setattr("ingest.fetch_prices.date", D1)
    df1 = fetcher.fetch_one("ABC", "1d")
    assert ds.calls[0][1] == date(2020, 1, 1)

    class D2(date):
        @classmethod
        def today(cls):
            return date(2020, 1, 6)

    monkeypatch.setattr("ingest.fetch_prices.date", D2)
    df2 = fetcher.fetch_one("ABC", "1d")
    assert ds.calls[1][1] == date(2020, 1, 6)
    assert len(df2) == len(df1) + 1

    fetcher.fetch_one("ABC", "1d", force_refresh=True)
    assert ds.calls[2][1] == date(2020, 1, 1)


def test_price_fetcher_uses_60m_alias(monkeypatch):
    class FrozenDate(date):
        @classmethod
        def today(cls):
            return date(2024, 1, 1)

    monkeypatch.setattr("ingest.fetch_prices.date", FrozenDate)
    monkeypatch.setattr("src.config.interval_policy.date", FrozenDate)

    class RecordingDataSource(DataSource):
        def __init__(self):
            self.calls: List[tuple[str, date, date, str]] = []

        def get_prices(self, ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
            self.calls.append((ticker, start, end, interval))
            idx = pd.date_range(start, periods=1, freq="h")
            return pd.DataFrame({"Adj Close": [1.0]}, index=idx)

        def validate_ticker(self, ticker: str) -> bool:
            return True

    monkeypatch.setattr("ingest.fetch_prices.save_cache", lambda *_, **__: None)

    datasource = RecordingDataSource()
    fetcher = PriceFetcher(datasource, throttle=0)

    result = fetcher.fetch_one("XYZ", "60m", force_refresh=True)

    assert datasource.calls, "datasource should receive the request"
    _, start_arg, end_arg, interval_arg = datasource.calls[0]
    expected_start = full_backfill_start("1h", today=FrozenDate.today())
    assert start_arg == expected_start
    assert end_arg == FrozenDate.today()
    assert interval_arg == "60m"
    assert not result.empty
