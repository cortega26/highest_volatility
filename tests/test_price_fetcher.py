from datetime import date
from typing import List

import pandas as pd

from src.cache import store
from datasource.base import DataSource
from ingest.fetch_prices import PriceFetcher


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
