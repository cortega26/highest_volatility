"""Price history retrieval using yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import os
import logging
import asyncio

import pandas as pd
import yfinance as yf

try:  # pragma: no cover - optional dependency
    from datasource.yahoo_async import YahooAsyncDataSource  # type: ignore
except Exception:  # pragma: no cover - optional
    YahooAsyncDataSource = None  # type: ignore

# Optional caching stack (present in this repo under src/cache and src/ingest)
try:  # pragma: no cover - optional import path
    from cache.store import load_cached, save_cache  # type: ignore
    from cache.merge import merge_incremental  # type: ignore
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

    Returns
    -------
    pandas.DataFrame
        Raw DataFrame as returned by :func:`yfinance.download`.
    """

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=lookback_days * 2)

    # Fallback path: plain batch download
    def _batch_download() -> pd.DataFrame:
        # Download in chunks to reduce failures and then stitch per-ticker frames
        CHUNK = 40
        success: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []

        chunks = [tickers[i : i + CHUNK] for i in range(0, len(tickers), CHUNK)]
        for chunk in chunks:
            with _silence_yfinance():
                df = yf.download(
                    " ".join(chunk),
                    start=start_dt,
                    end=end_dt,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    prepost=prepost,
                    group_by="column",
                )
            if df is None or df.empty:
                # Fallback to single ticker
                for t in chunk:
                    with _silence_yfinance():
                        d1 = yf.download(
                            t,
                            start=start_dt,
                            end=end_dt,
                            interval=interval,
                            progress=False,
                            auto_adjust=True,
                            prepost=prepost,
                        )
                    if d1 is None or d1.empty:
                        failed.append(t)
                        continue
                    if isinstance(d1.columns, pd.MultiIndex):
                        d1 = d1.droplevel(1, axis=1)
                    success[t] = d1.sort_index()
                continue

            # Split multi-ticker frame into per-ticker DataFrames
            if isinstance(df.columns, pd.MultiIndex):
                for t in sorted(set(df.columns.get_level_values(1))):
                    try:
                        sub = df.xs(t, axis=1, level=1).dropna(how="all")
                        if not sub.empty:
                            success[str(t)] = sub.sort_index()
                        else:
                            failed.append(str(t))
                    except Exception:
                        failed.append(str(t))
            else:
                # Single ticker returned without MultiIndex
                t = chunk[0]
                sub = df.dropna(how="all")
                if not sub.empty:
                    success[t] = sub.sort_index()
                else:
                    failed.append(t)

        if not success:
            return pd.DataFrame()

        combined = pd.concat(success, axis=1)  # level0=ticker, level1=field
        combined = combined.swaplevel(0, 1, axis=1).sort_index(axis=1)
        # Trim window and drop all-empty columns
        combined = combined.loc[combined.index >= pd.to_datetime(start_dt)]
        return combined.dropna(how="all")

    async def _async_download() -> pd.DataFrame:
        if YahooAsyncDataSource is None:  # pragma: no cover - import guard
            raise RuntimeError("Async data source unavailable")
        datasource = YahooAsyncDataSource()
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days * 2)
        sem = asyncio.Semaphore(max_workers)

        async def _one(ticker: str) -> tuple[str, pd.DataFrame]:
            async with sem:
                df = await datasource.get_prices(ticker, start_date, end_date, interval)
                return ticker, df

        tasks = [asyncio.create_task(_one(t)) for t in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        frames: Dict[str, pd.DataFrame] = {}
        for res in results:
            if isinstance(res, Exception):
                continue
            t, df = res
            if df is not None and not df.empty:
                frames[t] = df.sort_index()
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, axis=1)
        combined = combined.swaplevel(0, 1, axis=1).sort_index(axis=1)
        combined = combined.loc[combined.index >= pd.to_datetime(start_dt)]
        return combined.dropna(how="all")

    # Prefer batch mode for matrix correctness unless explicitly overridden
    if matrix_mode == "batch":
        return _batch_download()
    if matrix_mode == "async":
        return asyncio.run(_async_download())

    # Use cache if available; otherwise fall back to batch download
    if not use_cache or load_cached is None or save_cache is None:
        return _batch_download()

    # Per-ticker cached fetch with incremental update
    frames: Dict[str, pd.DataFrame] = {}
    end_date = date.today()
    start_date = (end_date - timedelta(days=lookback_days * 2))

    # First pass: detect up-to-date tickers and prepare a plan for those needing fetch
    to_fetch: List[str] = []
    cache_map: Dict[str, Optional[pd.DataFrame]] = {}
    for t in tickers:
        cached_df: Optional[pd.DataFrame] = None
        if not force_refresh:
            try:
                cached_df, _ = load_cached(t, interval)  # type: ignore[misc]
            except Exception:
                cached_df = None
        cache_map[t] = cached_df
        if cached_df is None or cached_df.empty:
            to_fetch.append(t)
            continue
        # If cache already covers up to end_date (or later), reuse without network
        last = cached_df.index[-1]
        if not isinstance(last, pd.Timestamp):
            last = pd.to_datetime(last)
        if last.date() >= end_date:
            frames[t] = cached_df
        else:
            to_fetch.append(t)

    # Parallel fetch for missing/incremental updates
    def _fetch_and_merge(t: str) -> Optional[pd.DataFrame]:
        cached_df = cache_map.get(t)
        if cached_df is None or cached_df.empty:
            fetch_start = start_date
        else:
            last = cached_df.index[-1]
            if not isinstance(last, pd.Timestamp):
                last = pd.to_datetime(last)
            fetch_start = (last + pd.Timedelta(days=1)).date()
            if fetch_start > end_date:
                # Already up to date
                return cached_df
        with _silence_yfinance():
            df_new = yf.download(
                t,
                start=fetch_start,
                end=end_date + timedelta(days=1),
                interval=interval,
                auto_adjust=False,
                progress=False,
                prepost=prepost,
            )
        if isinstance(df_new.columns, pd.MultiIndex):
            df_new.columns = df_new.columns.droplevel(1)
        if not isinstance(df_new.index, pd.DatetimeIndex):
            df_new.index = pd.to_datetime(df_new.index)
        df_new = df_new.sort_index()
        merged = merge_incremental(cached_df, df_new) if (cached_df is not None and not cached_df.empty) else df_new  # type: ignore[misc]
        if merged is None or merged.empty:
            return cached_df
        try:
            save_cache(t, interval, merged, source="yahoo")  # type: ignore[misc]
        except Exception:
            pass
        return merged

    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_map = {pool.submit(_fetch_and_merge, t): t for t in to_fetch}
            for fut in as_completed(fut_map):
                t = fut_map[fut]
                try:
                    df_merged = fut.result()
                    if df_merged is not None and not df_merged.empty:
                        frames[t] = df_merged
                except Exception:
                    # Leave missing tickers out; downstream will drop empty columns
                    continue

    if not frames:
        return _batch_download()

    # Build MultiIndex columns with level 0 as OHLC field and level 1 as ticker
    # Concat with keys=tickers gives level0=ticker; swap so field is level0
    combined = pd.concat(frames, axis=1)  # level0=ticker
    combined = combined.swaplevel(0, 1, axis=1).sort_index(axis=1)

    # Optional sanity check: detect clusters of identical recent price series
    try:
        lvl0 = list(combined.columns.get_level_values(0))
        field = "Adj Close" if "Adj Close" in lvl0 else ("Close" if "Close" in lvl0 else None)
        if field is not None:
            wide = combined[field]
            # Build fingerprints for last ~60 rows
            fp_map: Dict[str, List[str]] = {}
            tail_n = min(60, max(5, lookback_days // 4))
            for t in wide.columns:
                s = wide[t].tail(tail_n).astype(float)
                if s.dropna().empty:
                    continue
                # Fill NaNs for stable hashing
                s = s.ffill().bfill()
                h = hashlib.sha1(s.to_numpy().tobytes()).hexdigest()
                fp_map.setdefault(h, []).append(t)
            # Any suspicious clusters? (size >= 3)
            suspects = [tick for group in fp_map.values() if len(group) >= 3 for tick in group]
            if suspects:
                # Force refresh suspects in parallel, ignore cache
                def _force_refresh_one(t: str) -> Optional[pd.DataFrame]:
                    with _silence_yfinance():
                        df_new = yf.download(
                            t,
                            start=start_date,
                            end=end_date + timedelta(days=1),
                            interval=interval,
                            auto_adjust=False,
                            progress=False,
                            prepost=prepost,
                        )
                    if isinstance(df_new.columns, pd.MultiIndex):
                        df_new.columns = df_new.columns.droplevel(1)
                    if not isinstance(df_new.index, pd.DatetimeIndex):
                        df_new.index = pd.to_datetime(df_new.index)
                    df_new = df_new.sort_index()
                    if df_new.empty:
                        return None
                    try:
                        save_cache(t, interval, df_new, source="yahoo")  # type: ignore[misc]
                    except Exception:
                        pass
                    return df_new

                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    fut_map = {pool.submit(_force_refresh_one, t): t for t in set(suspects)}
                    for fut in as_completed(fut_map):
                        t = fut_map[fut]
                        try:
                            df_new = fut.result()
                            if df_new is not None:
                                frames[t] = df_new
                        except Exception:
                            continue
                combined = pd.concat(frames, axis=1)
                combined = combined.swaplevel(0, 1, axis=1).sort_index(axis=1)
    except Exception:
        # Non-fatal: if detection fails, proceed with current combined
        pass

    # Trim to the lookback window
    combined = combined.loc[combined.index >= pd.to_datetime(start_dt)]
    return combined.dropna(how="all")


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
