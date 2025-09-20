import pandas as pd

from highest_volatility.ingest import prices
from datetime import date, datetime


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

