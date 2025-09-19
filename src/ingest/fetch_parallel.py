"""Parallel price fetching utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Dict

import pandas as pd

from .fetch_prices import PriceFetcher
from highest_volatility.errors import DataSourceError, HVError, wrap_error
from highest_volatility.logging import get_logger, log_exception

logger = get_logger(__name__, component="parallel_fetch")


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
            except HVError as error:  # pragma: no cover - logging path
                log_exception(logger, error.add_context(ticker=t, interval=interval), event="parallel_fetch_failed")
            except Exception as exc:  # pragma: no cover - defensive
                error = wrap_error(
                    exc,
                    DataSourceError,
                    message="Parallel fetch failed",
                    context={"ticker": t, "interval": interval},
                )
                log_exception(logger, error, event="parallel_fetch_failed")
    return results
