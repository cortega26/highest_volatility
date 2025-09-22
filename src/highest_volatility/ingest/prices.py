"""Price history retrieval using yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta, date, timezone
from typing import Dict, List
from contextlib import contextmanager
import logging
import asyncio

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

try:  # pragma: no cover - optional dependency
    from highest_volatility.datasource.yahoo_async import YahooAsyncDataSource  # type: ignore
except Exception:  # pragma: no cover - optional
    YahooAsyncDataSource = None  # type: ignore

from highest_volatility.config.interval_policy import full_backfill_start
from highest_volatility.datasource.yahoo_http_async import YahooHTTPAsyncDataSource
from highest_volatility.ingest import downloaders

# Optional caching stack (present in this repo under src/cache and src/ingest)
try:  # pragma: no cover - optional import path
    from highest_volatility.cache.store import load_cached, save_cache  # type: ignore
    from highest_volatility.cache.merge import merge_incremental  # type: ignore
except Exception:  # pragma: no cover - optional
    load_cached = save_cache = merge_incremental = None  # type: ignore


def download_price_history(
    tickers: List[str],
    lookback_days: int,
    *,
    interval: str = "1d",
    prepost: bool = False,
    use_cache: bool = True,
    force_refresh: bool = False,
    max_workers: int = 8,
    matrix_mode: str = "batch",
    chunk_sleep: float = 0.0,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Download price history for ``tickers`` (with optional caching).

    Parameters
    ----------
    tickers:
        List of ticker symbols compatible with Yahoo Finance.
    lookback_days:
        Number of calendar days of history to request.
    interval:
        Data interval supported by Yahoo Finance (e.g. ``1d``, ``60m``, ``15m``).
    prepost:
        Include pre/post market data for intraday intervals.
    max_workers:
        Maximum number of threads used to download ticker chunks concurrently.
        Set to ``1`` to run chunk downloads sequentially.
    chunk_sleep:
        Optional pause in seconds between batches of concurrent chunk downloads.
    max_retries:
        Maximum retry attempts for failed downloads.

    Returns
    -------
    pandas.DataFrame
        Raw DataFrame as returned by :func:`yfinance.download`.
    """

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days * 2)

    def _download(*args, **kwargs):
        with _silence_yfinance():
            return yf.download(*args, **kwargs)

    download_with_retry = retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=1),
    )(_download)

    batch_request = downloaders.BatchDownloadRequest(
        tickers=tickers,
        start=start_dt,
        end=end_dt,
        interval=interval,
        prepost=prepost,
        max_workers=max_workers,
        chunk_sleep=chunk_sleep,
    )

    normalized_mode = matrix_mode.lower()
    use_cache_flag = use_cache
    if normalized_mode == "cache":
        normalized_mode = "batch"
        use_cache_flag = True

    if normalized_mode not in {"batch", "async"}:
        raise ValueError(f"Unsupported matrix_mode: {matrix_mode}")

    if normalized_mode == "async":
        async_end = date.today()
        async_start = full_backfill_start(interval if interval != "60m" else "1h", today=async_end)

        def _datasource_factory():
            if YahooAsyncDataSource is not None:
                return YahooAsyncDataSource()
            return YahooHTTPAsyncDataSource()

        async_request = downloaders.AsyncDownloadRequest(
            tickers=tickers,
            interval=interval,
            start=async_start,
            end=async_end,
            max_workers=max_workers,
            datasource_factory=_datasource_factory,
        )
        async_result = asyncio.run(downloaders.download_async(async_request))
        return async_result.to_dataframe(trim_start=start_dt)

    can_use_cache = (
        normalized_mode == "batch"
        and use_cache_flag
        and load_cached is not None
        and save_cache is not None
        and merge_incremental is not None
    )

    if not can_use_cache:
        batch_result = downloaders.download_batch(batch_request, download_with_retry)
        return batch_result.to_dataframe(trim_start=start_dt)

    cache_end_date = date.today()
    cache_start_date = cache_end_date - timedelta(days=lookback_days * 2)
    cache_plan = downloaders.plan_cache_fetch(
        tickers=tickers,
        interval=interval,
        start_date=cache_start_date,
        end_date=cache_end_date,
        force_refresh=force_refresh,
        load_cached=load_cached,  # type: ignore[misc]
        full_backfill_start_fn=full_backfill_start,
    )

    frames: Dict[str, pd.DataFrame] = dict(cache_plan.frames)

    cache_result = downloaders.execute_cache_fetch(
        cache_plan,
        download_with_retry,
        prepost=prepost,
        max_workers=max_workers,
        save_cache=save_cache,  # type: ignore[misc]
        merge_incremental=merge_incremental,  # type: ignore[misc]
    )
    frames.update(cache_result.frames)

    if not frames:
        batch_result = downloaders.download_batch(batch_request, download_with_retry)
        return batch_result.to_dataframe(trim_start=start_dt)

    fingerprint_plan = downloaders.plan_fingerprint_refresh(frames, lookback_days=lookback_days)
    fingerprint_result = downloaders.execute_fingerprint_refresh(
        fingerprint_plan,
        download_with_retry=download_with_retry,
        interval=interval,
        prepost=prepost,
        save_cache=save_cache,  # type: ignore[misc]
        max_workers=max_workers,
        start_date=cache_plan.start_date,
        end_date=cache_plan.end_date,
    )
    frames.update(fingerprint_result.frames)

    return downloaders.build_combined_dataframe(frames, trim_start=start_dt)


@contextmanager
def _silence_yfinance():
    """Lower logging for yfinance/urllib3 without touching sys.stdout/stderr.

    Redirecting stdout globally is unsafe with multi-threaded downloads; it can
    lead to printing to a closed file in other threads. We restrict ourselves
    to lowering logger levels which is thread-safe.
    """
    prev_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    for name in ("yfinance", "urllib3"):
        try:
            lg = logging.getLogger(name)
            lg.setLevel(logging.ERROR)
            lg.propagate = False
        except Exception:
            pass
    try:
        yield
    finally:
        try:
            logging.disable(prev_disable)
        except Exception:
            pass
