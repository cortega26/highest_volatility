"""Asynchronous Yahoo Finance data source using direct HTTP requests.

Sparse gaps in the upstream payload are tolerated by dropping individual
timestamps where neither ``adjclose`` nor ``close`` is provided. Completely
missing price histories still raise ``ValueError``. Dropped timestamps are
reported through ``df.attrs["dropped_yahoo_rows"]`` so downstream cache
validators can exempt them.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import logging
from typing import Any, Dict

import aiohttp
import pandas as pd

from .base_async import AsyncDataSource

logger = logging.getLogger(__name__)


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

        params: Dict[str, str | int]
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
        retained_timestamps: list[int] = []
        missing_indices: list[int] = []
        for idx, ts in enumerate(timestamps):
            adj_value = _value_at(adj_values, idx)
            close_value = _value_at(closes, idx)
            price_value = adj_value if adj_value is not None else close_value
            if price_value is None:
                missing_indices.append(idx)
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
            retained_timestamps.append(ts)

        missing_timestamps = [
            pd.to_datetime(timestamps[i], unit="s", utc=True).isoformat()
            for i in missing_indices
        ]
        if missing_timestamps:
            logger.warning(
                "Dropped %d missing Yahoo price rows for ticker %s: %s",
                len(missing_timestamps),
                ticker,
                missing_timestamps,
            )

        if not rows:
            raise ValueError(
                "Missing adjclose/close data for ticker "
                f"{ticker} at timestamps {missing_timestamps}"
            )

        df = pd.DataFrame(rows, index=pd.to_datetime(retained_timestamps, unit="s"))
        if df.isna().any().any():
            def _to_iso(ts: pd.Timestamp) -> str:
                ts = pd.Timestamp(ts)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                return ts.isoformat()

            missing_iso = [
                _to_iso(ts)
                for ts in df.index[df.isna().any(axis=1)]
            ]
            raise ValueError(
                "Unexpected NaN values present after combining adjclose/close for ticker "
                f"{ticker}: {missing_iso}"
            )

        df = df.sort_index()
        df.attrs["dropped_yahoo_rows"] = missing_timestamps
        return df

    async def validate_ticker(self, ticker: str) -> bool:
        try:
            await self.get_prices(ticker, date.today() - timedelta(days=1), date.today(), "1d")
            return True
        except Exception:
            return False
