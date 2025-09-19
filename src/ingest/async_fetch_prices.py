"""Asynchronous price fetching orchestrator with on-disk caching."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional
import asyncio
import pandas as pd

from cache.merge import merge_incremental
from cache.store import load_cached, save_cache
from config.interval_policy import full_backfill_start
from datasource.base_async import AsyncDataSource


class AsyncPriceFetcher:
    """High level async price fetching with incremental caching."""

    def __init__(self, datasource: AsyncDataSource, source_name: str = "composite", throttle: float = 0.2):
        self.datasource = datasource
        self.source_name = source_name
        self.throttle = throttle

    def _next_start(self, cached_df: Optional[pd.DataFrame], interval: str) -> date:
        if cached_df is None or cached_df.empty:
            return full_backfill_start(interval)

        last = cached_df.index[-1]
        if not isinstance(last, pd.Timestamp):
            last = pd.to_datetime(last)
        last_date = last.date()

        if interval.endswith("m") or interval.endswith("h"):
            # For intraday data re-fetch the last session to capture additional bars
            # released later in the same day. ``merge_incremental`` will drop
            # duplicates from the overlapping window.
            return last_date

        return last_date + timedelta(days=1)

    async def fetch_one(self, ticker: str, interval: str, *, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch and cache price history for a single ``ticker`` asynchronously."""

        cached_df: Optional[pd.DataFrame]
        if force_refresh:
            cached_df = None
        else:
            cached_df, _ = await asyncio.to_thread(load_cached, ticker, interval)

        start = full_backfill_start(interval) if force_refresh or cached_df is None else self._next_start(cached_df, interval)
        end = date.today()

        if cached_df is not None and start > end:
            await asyncio.sleep(self.throttle)
            return cached_df

        try:
            df_new = await self.datasource.get_prices(ticker, start, end, interval)
        finally:
            await asyncio.sleep(self.throttle)

        if not isinstance(df_new.index, pd.DatetimeIndex):
            df_new.index = pd.to_datetime(df_new.index)

        df_new = df_new.sort_index()
        merged = merge_incremental(cached_df, df_new) if cached_df is not None else df_new

        if merged.empty:
            raise ValueError("No data retrieved")

        await asyncio.to_thread(save_cache, ticker, interval, merged, self.source_name)
        return merged
