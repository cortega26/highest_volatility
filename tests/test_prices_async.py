import pandas as pd

from highest_volatility.ingest import prices
from datetime import date, datetime


class FakeAsyncDS:
    async def get_prices(self, ticker, start, end, interval):
        idx = pd.date_range("2020-01-01", periods=3)
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

    monkeypatch.setattr(prices, "date", D)
    monkeypatch.setattr(prices, "datetime", DT)
    monkeypatch.setattr(prices, "YahooAsyncDataSource", lambda: FakeAsyncDS())

    df = prices.download_price_history(["AAA", "BBB"], 5, matrix_mode="async", use_cache=False)
    assert set(df.columns.get_level_values(1)) == {"AAA", "BBB"}
    assert "Adj Close" in df.columns.get_level_values(0)

