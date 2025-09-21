"""Asynchronous Yahoo Finance data source adapter."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict

import aiohttp
import pandas as pd

from .base_async import AsyncDataSource


class YahooAsyncDataSource(AsyncDataSource):
    """Async DataSource implementation using Yahoo Finance HTTP API."""

    _BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    async def get_prices(self, ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
        params: Dict[str, str | int] = {
            "interval": interval,
            "period1": int(datetime.combine(start, datetime.min.time()).timestamp()),
            "period2": int(
                datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp()
            ),
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
        indicators = result.get("indicators", {})
        series = None
        adj = indicators.get("adjclose")
        if isinstance(adj, list) and adj:
            series = adj[0].get("adjclose")
        if series is None:
            quote = indicators.get("quote")
            if isinstance(quote, list) and quote:
                series = quote[0].get("close")
        if not series:
            raise ValueError("Missing adjclose/close in Yahoo response")
        df = pd.DataFrame({"Adj Close": series}, index=pd.to_datetime(timestamps, unit="s"))
        return df.sort_index()

    async def validate_ticker(self, ticker: str) -> bool:
        try:
            await self.get_prices(ticker, date.today() - timedelta(days=1), date.today(), "1d")
            return True
        except Exception:
            return False
