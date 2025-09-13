"""Async parallel price fetching utilities."""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable, Dict

import pandas as pd

from .async_fetch_prices import AsyncPriceFetcher

logger = logging.getLogger(__name__)


async def fetch_many_async(
    fetcher: AsyncPriceFetcher,
    tickers: Iterable[str],
    interval: str,
    *,
    force_refresh: bool = False,
    max_concurrency: int = 50,
) -> Dict[str, pd.DataFrame]:
    """Fetch many tickers concurrently using asyncio."""

    results: Dict[str, pd.DataFrame] = {}
    sem = asyncio.Semaphore(max_concurrency)

    async def worker(t: str) -> tuple[str, pd.DataFrame]:
        async with sem:
            df = await fetcher.fetch_one(t, interval, force_refresh=force_refresh)
            return t, df

    tasks = [asyncio.create_task(worker(t)) for t in tickers]
    for fut in asyncio.as_completed(tasks):
        try:
            t, df = await fut
            results[t] = df
        except Exception as exc:  # pragma: no cover - logging path
            logger.warning("Failed to fetch ticker: %s", exc)
    return results
