"""Background tasks and helpers for Highest Volatility pipelines."""

from __future__ import annotations

from .validation import validate_cache

__all__ = [
    "load_prices_from_cache",
    "refresh_cached_prices",
    "schedule_cache_refresh",
    "validate_cache",
]


def load_prices_from_cache(ticker: str, interval: str = "1d"):
    """Proxy to :func:`load_prices.load_prices_from_cache` without circular imports."""

    from .load_prices import load_prices_from_cache as _load_prices_from_cache

    return _load_prices_from_cache(ticker, interval)


async def refresh_cached_prices(*, interval: str = "1d", lookback_days: int = 365) -> None:
    """Proxy to :func:`cache_refresh.refresh_cached_prices` without circular imports."""

    from .cache_refresh import refresh_cached_prices as _refresh_cached_prices

    await _refresh_cached_prices(interval=interval, lookback_days=lookback_days)


async def schedule_cache_refresh(
    *, interval: str = "1d", lookback_days: int = 365, delay: float = 60 * 60 * 24
) -> None:
    """Proxy to :func:`cache_refresh.schedule_cache_refresh` without circular imports."""

    from .cache_refresh import schedule_cache_refresh as _schedule_cache_refresh

    await _schedule_cache_refresh(
        interval=interval, lookback_days=lookback_days, delay=delay
    )
