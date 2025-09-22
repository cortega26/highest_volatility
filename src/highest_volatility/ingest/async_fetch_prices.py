"""Asynchronous price fetching orchestrator with on-disk caching."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional
import asyncio
import pandas as pd

from highest_volatility.cache.merge import merge_incremental
from highest_volatility.cache.store import load_cached, save_cache
from highest_volatility.config.interval_policy import full_backfill_start
from highest_volatility.datasource.base_async import AsyncDataSource
from highest_volatility.errors import CacheError, DataSourceError, wrap_error
from highest_volatility.logging import get_logger, log_exception


logger = get_logger(__name__, component="async_price_fetcher")


def _drop_missing_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for column in ("Adj Close", "Close"):
        if column in df.columns:
            return df.dropna(subset=[column])
    return df.dropna()


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
            try:
                cached_df, _ = await asyncio.to_thread(load_cached, ticker, interval)
            except Exception as exc:
                error = wrap_error(
                    exc,
                    CacheError,
                    message="Failed to load cached prices",
                    context={"ticker": ticker, "interval": interval},
                    user_message="Unable to read cached prices; falling back to source.",
                )
                log_exception(logger, error, event="cache_load_failed")
                cached_df = None

        start = full_backfill_start(interval) if force_refresh or cached_df is None else self._next_start(cached_df, interval)
        end = date.today()

        if cached_df is not None and start > end:
            await asyncio.sleep(self.throttle)
            return cached_df

        try:
            df_new = await self.datasource.get_prices(ticker, start, end, interval)
        except Exception as exc:
            error = wrap_error(
                exc,
                DataSourceError,
                message="Datasource request failed",
                context={"ticker": ticker, "interval": interval},
            )
            log_exception(logger, error, event="datasource_fetch_failed")
            raise error
        finally:
            await asyncio.sleep(self.throttle)

        if not isinstance(df_new.index, pd.DatetimeIndex):
            df_new.index = pd.to_datetime(df_new.index)

        dropped_rows = df_new.attrs.get("dropped_yahoo_rows")
        if dropped_rows is not None:
            dropped_rows = list(dropped_rows)

        df_new = df_new.sort_index()
        df_new = _drop_missing_prices(df_new)
        if dropped_rows is not None:
            df_new.attrs["dropped_yahoo_rows"] = dropped_rows
        try:
            merged = (
                merge_incremental(cached_df, df_new) if cached_df is not None else df_new
            )
        except Exception as exc:
            error = wrap_error(
                exc,
                CacheError,
                message="Failed to merge incremental prices",
                context={"ticker": ticker, "interval": interval},
            )
            log_exception(logger, error, event="cache_merge_failed")
            raise error

        merged = _drop_missing_prices(merged)

        if merged.empty:
            error = DataSourceError(
                "No valid price data retrieved",
                context={"ticker": ticker, "interval": interval},
            )
            log_exception(logger, error, event="datasource_empty_result")
            raise error

        try:
            await asyncio.to_thread(
                save_cache,
                ticker,
                interval,
                merged,
                self.source_name,
                allowed_gaps=dropped_rows,
            )
        except Exception as exc:
            error = wrap_error(
                exc,
                CacheError,
                message="Failed to persist price cache",
                context={"ticker": ticker, "interval": interval},
            )
            log_exception(logger, error, event="cache_save_failed")
            raise error
        return merged
