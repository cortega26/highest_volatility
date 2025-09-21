"""Reusable helpers for price history downloads.

The helpers exported from this module follow public snake_case names. Legacy
aliases with leading underscores remain available for backwards compatibility.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pandas as pd


DownloadFn = Callable[..., pd.DataFrame]


__all__ = [
    "BatchDownloadRequest",
    "BatchDownloadResult",
    "AsyncDownloadRequest",
    "AsyncDownloadResult",
    "CacheFetchPlan",
    "CacheFetchResult",
    "FingerprintPlan",
    "FingerprintRefreshResult",
    "download_batch",
    "download_async",
    "plan_cache_fetch",
    "execute_cache_fetch",
    "plan_fingerprint_refresh",
    "execute_fingerprint_refresh",
]


@dataclass(frozen=True)
class BatchDownloadRequest:
    """Describe a multi-ticker batch download."""

    tickers: Sequence[str]
    start: datetime
    end: datetime
    interval: str
    prepost: bool
    max_workers: int
    chunk_size: int = 40
    chunk_sleep: float = 0.0


@dataclass
class BatchDownloadResult:
    """Outcome of a batch download."""

    frames: Dict[str, pd.DataFrame]
    failed: List[str]

    def to_dataframe(self, *, trim_start: Optional[datetime] = None) -> pd.DataFrame:
        if not self.frames:
            return pd.DataFrame()
        combined = pd.concat(self.frames, axis=1)
        combined = combined.swaplevel(0, 1, axis=1).sort_index(axis=1)
        if trim_start is not None:
            combined = combined.loc[combined.index >= pd.to_datetime(trim_start)]
        return combined.dropna(how="all")


def download_batch(
    request: BatchDownloadRequest, download_with_retry: DownloadFn
) -> BatchDownloadResult:
    """Download price history using chunked yfinance calls.

    Deprecated alias: :func:`_download_batch`.
    """

    chunks = [
        list(request.tickers[i : i + request.chunk_size])
        for i in range(0, len(request.tickers), request.chunk_size)
    ]
    if not chunks:
        return BatchDownloadResult(frames={}, failed=[])

    success: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    effective_workers = max(1, int(request.max_workers))
    chunk_results: List[tuple[int, Dict[str, pd.DataFrame], List[str]]] = []

    def _handle_chunk(idx: int, chunk: List[str]) -> tuple[int, Dict[str, pd.DataFrame], List[str]]:
        local_success: Dict[str, pd.DataFrame] = {}
        local_failed: List[str] = []

        df = download_with_retry(
            " ".join(chunk),
            start=request.start,
            end=request.end,
            interval=request.interval,
            progress=False,
            auto_adjust=True,
            prepost=request.prepost,
            group_by="column",
        )

        if df is None or df.empty:
            for ticker in chunk:
                d1 = download_with_retry(
                    ticker,
                    start=request.start,
                    end=request.end,
                    interval=request.interval,
                    progress=False,
                    auto_adjust=True,
                    prepost=request.prepost,
                )
                if d1 is None or d1.empty:
                    local_failed.append(ticker)
                    continue
                if isinstance(d1.columns, pd.MultiIndex):
                    d1 = d1.droplevel(1, axis=1)
                local_success[ticker] = d1.sort_index()
            return idx, local_success, local_failed

        if isinstance(df.columns, pd.MultiIndex):
            for ticker in sorted(set(df.columns.get_level_values(1))):
                try:
                    sub = df.xs(ticker, axis=1, level=1).dropna(how="all")
                    if not sub.empty:
                        local_success[str(ticker)] = sub.sort_index()
                    else:
                        local_failed.append(str(ticker))
                except Exception:
                    local_failed.append(str(ticker))
        else:
            ticker = chunk[0]
            sub = df.dropna(how="all")
            if not sub.empty:
                local_success[ticker] = sub.sort_index()
            else:
                local_failed.append(ticker)

        return idx, local_success, local_failed

    for batch_start in range(0, len(chunks), effective_workers):
        batch_indices = list(
            range(batch_start, min(batch_start + effective_workers, len(chunks)))
        )
        batch_workers = min(len(batch_indices), effective_workers)
        with ThreadPoolExecutor(max_workers=batch_workers) as executor:
            futures = {
                executor.submit(_handle_chunk, idx, chunks[idx]): idx for idx in batch_indices
            }
            for future in as_completed(futures):
                chunk_results.append(future.result())
        if request.chunk_sleep:
            time.sleep(request.chunk_sleep)

    for _, chunk_success, chunk_failed in sorted(chunk_results, key=lambda item: item[0]):
        success.update(chunk_success)
        failed.extend(chunk_failed)

    return BatchDownloadResult(frames=success, failed=failed)


@dataclass(frozen=True)
class AsyncDownloadRequest:
    """Describe an asynchronous download run."""

    tickers: Sequence[str]
    interval: str
    start: date
    end: date
    max_workers: int
    datasource_factory: Callable[[], Any]


@dataclass
class AsyncDownloadResult:
    """Outcome of an asynchronous download."""

    frames: Dict[str, pd.DataFrame]

    def to_dataframe(self, *, trim_start: Optional[datetime] = None) -> pd.DataFrame:
        if not self.frames:
            return pd.DataFrame()
        combined = pd.concat(self.frames, axis=1)
        combined = combined.swaplevel(0, 1, axis=1).sort_index(axis=1)
        if trim_start is not None:
            combined = combined.loc[combined.index >= pd.to_datetime(trim_start)]
        return combined.dropna(how="all")


async def download_async(request: AsyncDownloadRequest) -> AsyncDownloadResult:
    """Fetch price history concurrently using an async datasource.

    Deprecated alias: :func:`_download_async`.
    """

    datasource = request.datasource_factory()
    semaphore = asyncio.Semaphore(request.max_workers)

    async def _one(ticker: str) -> tuple[str, pd.DataFrame]:
        async with semaphore:
            df = await datasource.get_prices(ticker, request.start, request.end, request.interval)
            return ticker, df

    tasks = [asyncio.create_task(_one(ticker)) for ticker in request.tickers]
    results: List[tuple[str, pd.DataFrame] | BaseException] = await asyncio.gather(
        *tasks, return_exceptions=True
    )

    frames: Dict[str, pd.DataFrame] = {}
    for result in results:
        if isinstance(result, BaseException):
            continue
        ticker, df = result
        if df is not None and not df.empty:
            frames[ticker] = df.sort_index()

    return AsyncDownloadResult(frames=frames)


@dataclass
class CacheFetchPlan:
    """Plan describing which tickers require cache refresh."""

    interval: str
    start_date: date
    end_date: date
    frames: Dict[str, pd.DataFrame]
    to_fetch: List[str]
    cache_map: Dict[str, Optional[pd.DataFrame]]
    fetch_starts: Dict[str, date]


def plan_cache_fetch(
    *,
    tickers: Sequence[str],
    interval: str,
    start_date: date,
    end_date: date,
    force_refresh: bool,
    load_cached: Callable[[str, str], tuple[Optional[pd.DataFrame], Any]],
    full_backfill_start_fn: Callable[[str, Optional[date]], date],
) -> CacheFetchPlan:
    """Build a cache refresh plan for the requested tickers.

    Deprecated alias: :func:`_plan_cache_fetch`.
    """
    frames: Dict[str, pd.DataFrame] = {}
    to_fetch: List[str] = []
    cache_map: Dict[str, Optional[pd.DataFrame]] = {}
    fetch_starts: Dict[str, date] = {}

    def _next_fetch_start(cached_df: Optional[pd.DataFrame]) -> date:
        if cached_df is None or cached_df.empty:
            return full_backfill_start_fn(interval, today=end_date)
        last = cached_df.index[-1]
        if not isinstance(last, pd.Timestamp):
            last = pd.to_datetime(last)
        last_date = last.date()
        interval_key = interval.lower()
        if interval_key.endswith(("m", "h")):
            return last_date
        return last_date + timedelta(days=1)

    for ticker in tickers:
        cached_df: Optional[pd.DataFrame] = None
        if not force_refresh:
            try:
                cached_df, _ = load_cached(ticker, interval)
            except Exception:
                cached_df = None
        cache_map[ticker] = cached_df

        if cached_df is not None and not cached_df.empty and not force_refresh:
            fetch_start = _next_fetch_start(cached_df)
            if fetch_start > end_date:
                frames[ticker] = cached_df
                continue
            fetch_starts[ticker] = fetch_start
            to_fetch.append(ticker)
            continue

        fetch_start = full_backfill_start_fn(interval, today=end_date)
        fetch_starts[ticker] = fetch_start
        to_fetch.append(ticker)

    return CacheFetchPlan(
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        frames=frames,
        to_fetch=to_fetch,
        cache_map=cache_map,
        fetch_starts=fetch_starts,
    )


@dataclass
class CacheFetchResult:
    """Result of executing a cache fetch plan."""

    frames: Dict[str, pd.DataFrame]


def execute_cache_fetch(
    plan: CacheFetchPlan,
    download_with_retry: DownloadFn,
    *,
    prepost: bool,
    max_workers: int,
    save_cache: Callable[[str, str, pd.DataFrame, str], None],
    merge_incremental: Optional[Callable[[Optional[pd.DataFrame], pd.DataFrame], Optional[pd.DataFrame]]],
) -> CacheFetchResult:
    """Execute a cache fetch plan and persist refreshed frames.

    Deprecated alias: :func:`_execute_cache_fetch`.
    """
    frames: Dict[str, pd.DataFrame] = {}

    def _fetch_and_merge(ticker: str) -> Optional[pd.DataFrame]:
        cached_df = plan.cache_map.get(ticker)
        fetch_start = plan.fetch_starts.get(ticker, plan.start_date)
        if fetch_start > plan.end_date:
            return cached_df

        df_new = download_with_retry(
            ticker,
            start=fetch_start,
            end=plan.end_date + timedelta(days=1),
            interval=plan.interval,
            auto_adjust=False,
            progress=False,
            prepost=prepost,
        )
        if isinstance(df_new.columns, pd.MultiIndex):
            df_new.columns = df_new.columns.droplevel(1)
        if not isinstance(df_new.index, pd.DatetimeIndex):
            df_new.index = pd.to_datetime(df_new.index)
        df_new = df_new.sort_index()

        if cached_df is not None and not cached_df.empty and merge_incremental is not None:
            merged = merge_incremental(cached_df, df_new)
        else:
            merged = df_new if df_new is not None else cached_df

        if merged is None or merged.empty:
            return cached_df

        try:
            save_cache(ticker, plan.interval, merged, source="yahoo")
        except Exception:
            pass
        return merged

    if plan.to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_map = {pool.submit(_fetch_and_merge, ticker): ticker for ticker in plan.to_fetch}
            for fut in as_completed(fut_map):
                ticker = fut_map[fut]
                try:
                    df_merged = fut.result()
                    if df_merged is not None and not df_merged.empty:
                        frames[ticker] = df_merged
                except Exception:
                    continue

    return CacheFetchResult(frames=frames)


@dataclass
class FingerprintPlan:
    """Describe tickers that require fingerprint-based refresh."""

    suspects: List[str]
    field: Optional[str]
    tail_window: int


def plan_fingerprint_refresh(
    frames: Mapping[str, pd.DataFrame], *, lookback_days: int
) -> FingerprintPlan:
    """Identify tickers that share suspiciously similar price fingerprints.

    Deprecated alias: :func:`_plan_fingerprint_refresh`.
    """
    try:
        combined = pd.concat(frames, axis=1)
        combined = combined.swaplevel(0, 1, axis=1).sort_index(axis=1)
        level0 = list(combined.columns.get_level_values(0))
        field = "Adj Close" if "Adj Close" in level0 else ("Close" if "Close" in level0 else None)
        if field is None:
            return FingerprintPlan(suspects=[], field=None, tail_window=0)
        wide = combined[field]
        fingerprint_map: Dict[str, List[str]] = {}
        tail_n = min(60, max(5, lookback_days // 4))
        for ticker in wide.columns:
            series = wide[ticker].tail(tail_n).astype(float)
            if series.dropna().empty:
                continue
            series = series.ffill().bfill()
            fingerprint = hashlib.sha1(series.to_numpy().tobytes()).hexdigest()
            fingerprint_map.setdefault(fingerprint, []).append(ticker)
        suspects = [ticker for group in fingerprint_map.values() if len(group) >= 3 for ticker in group]
        return FingerprintPlan(suspects=list(set(suspects)), field=field, tail_window=tail_n)
    except Exception:
        return FingerprintPlan(suspects=[], field=None, tail_window=0)


@dataclass
class FingerprintRefreshResult:
    """Result of fingerprint-based refresh."""

    frames: Dict[str, pd.DataFrame]


def execute_fingerprint_refresh(
    plan: FingerprintPlan,
    *,
    download_with_retry: DownloadFn,
    interval: str,
    prepost: bool,
    save_cache: Callable[[str, str, pd.DataFrame, str], None],
    max_workers: int,
    start_date: date,
    end_date: date,
) -> FingerprintRefreshResult:
    """Force-refresh tickers flagged by :func:`plan_fingerprint_refresh`.

    Deprecated alias: :func:`_execute_fingerprint_refresh`.
    """
    frames: Dict[str, pd.DataFrame] = {}

    if not plan.suspects:
        return FingerprintRefreshResult(frames={})

    def _force_refresh_one(ticker: str) -> Optional[pd.DataFrame]:
        df_new = download_with_retry(
            ticker,
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
            save_cache(ticker, interval, df_new, source="yahoo")
        except Exception:
            pass
        return df_new

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {pool.submit(_force_refresh_one, ticker): ticker for ticker in plan.suspects}
        for fut in as_completed(fut_map):
            ticker = fut_map[fut]
            try:
                df_new = fut.result()
                if df_new is not None and not df_new.empty:
                    frames[ticker] = df_new
            except Exception:
                continue

    return FingerprintRefreshResult(frames=frames)


def build_combined_dataframe(
    frames: Mapping[str, pd.DataFrame], *, trim_start: Optional[datetime] = None
) -> pd.DataFrame:
    """Combine individual ticker frames into a multi-index DataFrame."""

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=1)
    combined = combined.swaplevel(0, 1, axis=1).sort_index(axis=1)
    if trim_start is not None:
        combined = combined.loc[combined.index >= pd.to_datetime(trim_start)]
    return combined.dropna(how="all")


# Deprecated aliases retained for backwards compatibility with callers that
# still reference the private helper names.
_download_batch = download_batch
_download_async = download_async
_plan_cache_fetch = plan_cache_fetch
_execute_cache_fetch = execute_cache_fetch
_plan_fingerprint_refresh = plan_fingerprint_refresh
_execute_fingerprint_refresh = execute_fingerprint_refresh

