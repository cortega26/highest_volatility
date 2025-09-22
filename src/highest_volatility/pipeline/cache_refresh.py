"""Async helpers for refreshing cached price data."""

from __future__ import annotations

import asyncio
from typing import List

from highest_volatility.cache.store import CACHE_ROOT
from highest_volatility.ingest.prices import download_price_history
from highest_volatility.errors import HVError, wrap_error
from highest_volatility.logging import get_logger, log_exception


logger = get_logger(__name__)


def _cached_tickers(interval: str = "1d") -> List[str]:
    """Return tickers available in the local cache for ``interval``.

    Parameters
    ----------
    interval:
        Price interval directory under ``.cache/prices``.
    """

    interval_dir = CACHE_ROOT / interval
    if not interval_dir.exists():
        return []
    return [p.stem for p in interval_dir.glob("*.parquet")]


async def refresh_cached_prices(*, interval: str = "1d", lookback_days: int = 365) -> None:
    """Refresh cached prices for all stored tickers.

    Each ticker is downloaded sequentially using ``asyncio.to_thread`` to avoid
    blocking the event loop.
    """

    tickers = _cached_tickers(interval)
    for ticker in tickers:
        try:
            await asyncio.to_thread(
                download_price_history,
                [ticker],
                lookback_days,
                interval=interval,
                force_refresh=False,
            )
        except HVError as err:
            log_exception(
                logger,
                err.add_context(ticker=ticker, interval=interval),
                event="cache_refresh_failed",
            )


async def schedule_cache_refresh(
    *,
    interval: str = "1d",
    lookback_days: int = 365,
    delay: float = 60 * 60 * 24,
) -> None:
    """Run :func:`refresh_cached_prices` periodically.

    Parameters
    ----------
    interval:
        Cache interval to refresh.
    lookback_days:
        Number of days of history to fetch.
    delay:
        Seconds to wait between refresh runs. Defaults to 1 day.
    """

    while True:
        try:
            await refresh_cached_prices(interval=interval, lookback_days=lookback_days)
        except asyncio.CancelledError:
            logger.info(
                "Cache refresh loop cancelled", extra={"interval": interval, "event": "cache_refresh_cancelled"}
            )
            raise
        except Exception as err:  # pragma: no cover - defensive catch-all
            log_exception(
                logger,
                wrap_error(
                    err,
                    message="Scheduled cache refresh iteration failed",
                    context={
                        "interval": interval,
                        "lookback_days": lookback_days,
                    },
                ),
                event="cache_refresh_iteration_failed",
            )
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            logger.info(
                "Cache refresh delay cancelled", extra={"interval": interval, "event": "cache_refresh_cancelled"}
            )
            raise
