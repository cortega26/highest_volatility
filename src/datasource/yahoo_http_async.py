"""Asynchronous Yahoo Finance data source using direct HTTP requests."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import aiohttp
import pandas as pd

from .base_async import AsyncDataSource


class YahooHTTPAsyncDataSource(AsyncDataSource):
    """Async data source hitting Yahoo's chart API via :mod:`aiohttp`."""

    _BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    async def get_prices(self, ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
        """Retrieve historical prices for ``ticker`` asynchronously."""
        params = {
            "interval": interval,
            "period1": int(datetime.combine(start, datetime.min.time()).timestamp()),
            "period2": int(datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp()),
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(self._BASE_URL.format(ticker=ticker), params=params) as resp:
                resp.raise_for_status()
                data: Any = await resp.json()
        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        if not timestamps:
            raise ValueError("Empty data returned")
        adj_close = result["indicators"]["adjclose"][0]["adjclose"]
        df = pd.DataFrame({"Adj Close": adj_close}, index=pd.to_datetime(timestamps, unit="s"))
        return df.sort_index()

    async def validate_ticker(self, ticker: str) -> bool:
        try:
            await self.get_prices(ticker, date.today() - timedelta(days=1), date.today(), "1d")
            return True
        except Exception:
            return False
