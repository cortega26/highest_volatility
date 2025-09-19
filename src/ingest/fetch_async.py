"""Async parallel price fetching utilities."""

from __future__ import annotations

import asyncio
from typing import Iterable, Dict

import pandas as pd

from .async_fetch_prices import AsyncPriceFetcher
from highest_volatility.errors import DataSourceError, HVError, wrap_error
from highest_volatility.logging import get_logger, log_exception

logger = get_logger(__name__, component="async_fetch")


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
            try:
                df = await fetcher.fetch_one(t, interval, force_refresh=force_refresh)
            except HVError as error:
                raise error.add_context(ticker=t, interval=interval)
            except Exception as exc:
                raise wrap_error(
                    exc,
                    DataSourceError,
                    message="Async fetch failed",
                    context={"ticker": t, "interval": interval},
                )
            return t, df

    tasks = [asyncio.create_task(worker(t)) for t in tickers]
    for fut in asyncio.as_completed(tasks):
        try:
            t, df = await fut
            results[t] = df
        except HVError as error:  # pragma: no cover - logging path
            log_exception(logger, error, event="async_fetch_failed")
        except Exception as exc:  # pragma: no cover - defensive
            error = wrap_error(
                exc,
                DataSourceError,
                message="Async fetch failed",
                context={"interval": interval},
            )
            log_exception(logger, error, event="async_fetch_failed")
    return results
