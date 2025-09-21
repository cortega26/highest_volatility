"""Yahoo Finance data source adapter."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import DataSource


class YahooDataSource(DataSource):
    """DataSource implementation using :mod:`yfinance`."""

    def validate_ticker(self, ticker: str) -> bool:
        try:
            info = yf.Ticker(ticker).history(period="1d")
            return not info.empty
        except Exception:
            return False

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=8))
    def get_prices(self, ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
        df = yf.download(
            ticker,
            start=start,
            end=end + timedelta(days=1),  # yfinance end is exclusive
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty:
            raise ValueError("Empty DataFrame returned")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()
