"""Parallel price fetching utilities."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Dict

import pandas as pd

from .fetch_prices import PriceFetcher

logger = logging.getLogger(__name__)


def fetch_many(
    fetcher: PriceFetcher,
    tickers: Iterable[str],
    interval: str,
    *,
    force_refresh: bool = False,
    max_workers: int = 8,
) -> Dict[str, pd.DataFrame]:
    """Fetch many tickers in parallel."""

    results: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(fetcher.fetch_one, t, interval, force_refresh=force_refresh): t
            for t in tickers
        }
        for fut in as_completed(future_map):
            t = future_map[fut]
            try:
                results[t] = fut.result()
            except Exception as exc:  # pragma: no cover - logging path
                logger.warning("Failed to fetch %s: %s", t, exc)
    return results
