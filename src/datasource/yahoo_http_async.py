"""Asynchronous Yahoo Finance data source using direct HTTP requests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import aiohttp
import pandas as pd

from .base_async import AsyncDataSource


class YahooHTTPAsyncDataSource(AsyncDataSource):
    """Async data source hitting Yahoo's chart API via :mod:`aiohttp`."""

    _BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    async def get_prices(self, ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
        """Retrieve historical prices for ``ticker`` asynchronously."""
        # Map common aliases to Yahoo API values
        yahoo_interval = "60m" if interval in ("1h", "60m") else interval
        # Compute UTC-clamped period range. Yahoo returns 422 for intraday if period2 is
        # in the future or for certain period1/period2 windows. For intraday, prefer the
        # "range=<Nd>" parameter which is accepted by the chart API.
        dt_start = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        # Include the full 'end' date by advancing to the next midnight UTC
        dt_end = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=1)
        now_utc = datetime.now(timezone.utc)
        if dt_end > now_utc:
            dt_end = now_utc
        if dt_end <= dt_start:
            # Degenerate or inverted window; nudge end forward minimally
            dt_end = dt_start + timedelta(seconds=1)

        is_intraday = yahoo_interval.endswith("m") or yahoo_interval in ("60m",)

        if is_intraday:
            # Convert window length to whole days, clamp to at least 1 day
            window_days = max(1, int((dt_end - dt_start).total_seconds() // 86400))
            params = {
                "interval": yahoo_interval,
                "range": f"{window_days}d",
                "events": "div,splits",
                "includeAdjustedClose": "true",
            }
        else:
            params = {
                "interval": yahoo_interval,
                "period1": int(dt_start.timestamp()),
                "period2": int(dt_end.timestamp()),
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
        # Prefer adjusted close when available
        adj = indicators.get("adjclose")
        if isinstance(adj, list) and adj:
            series = adj[0].get("adjclose")
        # Fallback to regular close from quote
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
