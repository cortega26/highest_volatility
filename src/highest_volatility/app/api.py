"""FastAPI application exposing Highest Volatility functionality.

Run with:
    uvicorn highest_volatility.app.api:app --reload
or:
    python -m highest_volatility.app.api

Environment variables prefixed with ``HV_`` (e.g. ``HV_LOOKBACK_DAYS``)
can override default configuration values.
"""

from __future__ import annotations

import asyncio
import json
from typing import cast

import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from pydantic_settings import BaseSettings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import Response

from highest_volatility.app.cli import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_DAYS,
    DEFAULT_TOP_N,
)
from highest_volatility.compute.metrics import METRIC_REGISTRY
from highest_volatility.ingest.prices import download_price_history
from highest_volatility.universe import build_universe
from highest_volatility.pipeline.cache_refresh import schedule_cache_refresh


class Settings(BaseSettings):
    """Configuration for the API.

    Values can be overridden via environment variables prefixed with
    ``HV_``.  For example, ``HV_LOOKBACK_DAYS=100``.
    """

    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    interval: str = "1d"
    prepost: bool = False
    top_n: int = DEFAULT_TOP_N
    metric: str = "cc_vol"
    min_days: int = DEFAULT_MIN_DAYS
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_universe: int = 60
    cache_ttl_prices: int = 60
    cache_ttl_metrics: int = 60
    rate_limit: str = "60/minute"
    cache_refresh_interval: float = 60 * 60 * 24

    class Config:
        env_prefix = "HV_"


settings = Settings()


def get_settings() -> Settings:
    return settings


app = FastAPI(title="Highest Volatility API")

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])
app.state.limiter = limiter
# Per-endpoint limits can be overridden with ``@app.state.limiter.limit("X/minute")``
def _rate_limit_handler(request: Request, exc: Exception) -> Response:
    """Forward SlowAPI rate-limit exceptions to its default handler."""

    return _rate_limit_exceeded_handler(request, cast(RateLimitExceeded, exc))


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)


@app.on_event("startup")
async def on_startup() -> None:
    """Initialize cache backend."""
    client = redis.from_url(settings.redis_url, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(client), prefix="hv-cache")
    app.state.cache_refresh_task = asyncio.create_task(
        schedule_cache_refresh(
            interval=settings.interval,
            lookback_days=settings.lookback_days,
            delay=settings.cache_refresh_interval,
        )
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    task = getattr(app.state, "cache_refresh_task", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@app.get("/universe")
@cache(expire=settings.cache_ttl_universe)
def universe_endpoint(
    top_n: int | None = None,
    settings: Settings = Depends(get_settings),
):
    """Return a validated ticker universe."""

    limit = top_n or settings.top_n
    tickers, fortune = build_universe(limit, validate=True)
    return {"tickers": tickers, "fortune": fortune.to_dict(orient="records")}


@app.get("/prices")
@cache(expire=settings.cache_ttl_prices)
def prices_endpoint(
    tickers: str,
    lookback_days: int | None = None,
    interval: str | None = None,
    prepost: bool | None = None,
    settings: Settings = Depends(get_settings),
):
    """Return price history for ``tickers``."""

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    lb = lookback_days or settings.lookback_days
    iv = interval or settings.interval
    pp = prepost if prepost is not None else settings.prepost
    df = download_price_history(ticker_list, lb, interval=iv, prepost=pp)
    if df.empty:
        return {"data": []}
    return json.loads(df.to_json(orient="split", date_format="iso"))


@app.get("/metrics")
@cache(expire=settings.cache_ttl_metrics)
def metrics_endpoint(
    tickers: str,
    metric: str | None = None,
    lookback_days: int | None = None,
    interval: str | None = None,
    min_days: int | None = None,
    settings: Settings = Depends(get_settings),
):
    """Compute ``metric`` for ``tickers``."""

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    met = metric or settings.metric
    lb = lookback_days or settings.lookback_days
    iv = interval or settings.interval
    md = min_days or settings.min_days
    if met not in METRIC_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown metric '{met}'")
    prices = download_price_history(ticker_list, lb, interval=iv)
    func = METRIC_REGISTRY[met]
    result = func(prices, tickers=ticker_list, min_periods=md, interval=iv)
    return result.to_dict(orient="records")


def main() -> None:
    """Run a development server using :mod:`uvicorn`.

    This mirrors running ``uvicorn highest_volatility.app.api:app``.
    """

    import uvicorn

    uvicorn.run("highest_volatility.app.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
