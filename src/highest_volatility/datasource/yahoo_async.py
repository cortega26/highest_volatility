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
        quote = indicators.get("quote")
        quote_block: Dict[str, list[Any]] = quote[0] if isinstance(quote, list) and quote else {}
        opens = quote_block.get("open")
        highs = quote_block.get("high")
        lows = quote_block.get("low")
        closes = quote_block.get("close")
        volumes = quote_block.get("volume")

        adj_values = None
        adj = indicators.get("adjclose")
        if isinstance(adj, list) and adj:
            adj_values = adj[0].get("adjclose")

        if adj_values is None and closes is None:
            raise ValueError("Missing adjclose/close in Yahoo response")

        def _value_at(values: list[Any] | None, idx: int) -> Any:
            if values is None or idx >= len(values):
                return None
            value = values[idx]
            if value is None or pd.isna(value):
                return None
            return value

        rows: list[dict[str, Any]] = []
        retained: list[int] = []
        for idx, ts in enumerate(timestamps):
            adj_value = _value_at(adj_values, idx)
            close_value = _value_at(closes, idx)
            price_value = adj_value if adj_value is not None else close_value
            if price_value is None:
                continue
            row = {
                "Open": _value_at(opens, idx),
                "High": _value_at(highs, idx),
                "Low": _value_at(lows, idx),
                "Close": close_value if close_value is not None else price_value,
                "Adj Close": price_value,
                "Volume": _value_at(volumes, idx),
            }
            for key in ("Open", "High", "Low"):
                if row[key] is None:
                    row[key] = price_value
            if row["Volume"] is None:
                row["Volume"] = 0
            rows.append(row)
            retained.append(ts)

        if not rows:
            raise ValueError("Empty data returned")

        df = pd.DataFrame(rows, index=pd.to_datetime(retained, unit="s"))
        return df.sort_index()

    async def validate_ticker(self, ticker: str) -> bool:
        try:
            await self.get_prices(ticker, date.today() - timedelta(days=1), date.today(), "1d")
            return True
        except Exception:
            return False
