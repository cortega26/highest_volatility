"""Price fetching orchestrator with on-disk caching."""

from __future__ import annotations

from datetime import date, timedelta
import time
from typing import Optional

import pandas as pd

from cache.merge import merge_incremental
from cache.store import load_cached, save_cache
from config.interval_policy import full_backfill_start
from datasource.base import DataSource


class PriceFetcher:
    """High level price fetching with incremental caching."""

    def __init__(self, datasource: DataSource, source_name: str = "composite", throttle: float = 0.2):
        self.datasource = datasource
        self.source_name = source_name
        self.throttle = throttle

    def _next_start(self, cached_df: Optional[pd.DataFrame], interval: str) -> date:
        if cached_df is None or cached_df.empty:
            return full_backfill_start(interval)
        last = cached_df.index[-1].date()
        # TODO: step by exact bar size for intraday intervals
        return last + timedelta(days=1)

    def fetch_one(self, ticker: str, interval: str, *, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch and cache price history for a single ``ticker``."""

        cached_df: Optional[pd.DataFrame]
        if force_refresh:
            cached_df = None
        else:
            cached_df, _ = load_cached(ticker, interval)

        start = full_backfill_start(interval) if force_refresh or cached_df is None else self._next_start(cached_df, interval)
        end = date.today()

        if cached_df is not None and start > end:
            time.sleep(self.throttle)
            return cached_df

        try:
            df_new = self.datasource.get_prices(ticker, start, end, interval)
        finally:
            time.sleep(self.throttle)

        if not isinstance(df_new.index, pd.DatetimeIndex):
            df_new.index = pd.to_datetime(df_new.index)

        df_new = df_new.sort_index()
        merged = merge_incremental(cached_df, df_new) if cached_df is not None else df_new

        if merged.empty:
            raise ValueError("No data retrieved")

        save_cache(ticker, interval, merged, self.source_name)
        return merged
