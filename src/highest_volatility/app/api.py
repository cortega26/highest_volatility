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
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from highest_volatility.app.cli import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_DAYS,
    DEFAULT_TOP_N,
)
from highest_volatility.compute.metrics import METRIC_REGISTRY
from highest_volatility.errors import (
    ComputeError,
    ErrorCode,
    HVError,
    IntegrationError,
    wrap_error,
)
from highest_volatility.ingest.prices import download_price_history
from highest_volatility.logging import get_logger, log_exception
from highest_volatility.universe import build_universe
from highest_volatility.pipeline.cache_refresh import schedule_cache_refresh
from highest_volatility.security.validation import (
    SanitizationError,
    sanitize_interval,
    sanitize_metric,
    sanitize_multiple_tickers,
    sanitize_positive_int,
)


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


logger = get_logger(__name__, component="rest_api")


REQUEST_TICKER_LIMIT = 100
LOOKBACK_MIN_DAYS = 30
LOOKBACK_MAX_DAYS = 2000
MIN_DAYS_MINIMUM = 10
TOP_N_MINIMUM = 10
TOP_N_MAXIMUM = 500


_STATUS_BY_CODE = {
    ErrorCode.VALIDATION: 400,
    ErrorCode.DATA_SOURCE: 502,
    ErrorCode.CACHE: 503,
    ErrorCode.INTEGRATION: 502,
    ErrorCode.CONFIG: 500,
    ErrorCode.COMPUTE: 500,
}


def _handle_error(error: HVError, *, event: str, endpoint: str) -> None:
    """Log ``error`` and raise an HTTP response."""

    log_exception(logger, error, event=event, context={"endpoint": endpoint})
    status = _STATUS_BY_CODE.get(error.code, 500)
    raise HTTPException(status_code=status, detail=error.user_message)


def get_settings() -> Settings:
    return settings


class SecureHeadersMiddleware(BaseHTTPMiddleware):
    """Attach standard security headers to responses."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response = await call_next(request)
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload"
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        return response


app = FastAPI(title="Highest Volatility API")
app.add_middleware(SecureHeadersMiddleware)

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

    try:
        limit = sanitize_positive_int(
            top_n if top_n is not None else settings.top_n,
            field="top_n",
            minimum=TOP_N_MINIMUM,
            maximum=TOP_N_MAXIMUM,
        )
    except SanitizationError as exc:
        _handle_error(exc, event="universe_validation", endpoint="universe")
    try:
        tickers, fortune = build_universe(limit, validate=True)
    except HVError as error:  # pragma: no cover - defensive
        _handle_error(error, event="universe_failure", endpoint="universe")
    except Exception as exc:  # pragma: no cover - defensive
        error = wrap_error(
            exc,
            IntegrationError,
            message="Failed to build Fortune universe",
            context={"top_n": limit},
        )
        _handle_error(error, event="universe_failure", endpoint="universe")
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

    try:
        ticker_list = sanitize_multiple_tickers(
            tickers, max_tickers=REQUEST_TICKER_LIMIT
        )
        lb = sanitize_positive_int(
            lookback_days if lookback_days is not None else settings.lookback_days,
            field="lookback_days",
            minimum=LOOKBACK_MIN_DAYS,
            maximum=LOOKBACK_MAX_DAYS,
        )
        iv = sanitize_interval(interval or settings.interval)
    except SanitizationError as exc:
        _handle_error(exc, event="prices_validation", endpoint="prices")
    pp = prepost if prepost is not None else settings.prepost
    try:
        df = download_price_history(ticker_list, lb, interval=iv, prepost=pp)
    except HVError as error:
        _handle_error(error, event="prices_failure", endpoint="prices")
    except Exception as exc:
        error = wrap_error(
            exc,
            IntegrationError,
            message="Failed to download price history",
            context={"tickers": ticker_list, "interval": iv},
        )
        _handle_error(error, event="prices_failure", endpoint="prices")
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

    try:
        ticker_list = sanitize_multiple_tickers(
            tickers, max_tickers=REQUEST_TICKER_LIMIT
        )
        met = sanitize_metric(metric or settings.metric)
        lb = sanitize_positive_int(
            lookback_days if lookback_days is not None else settings.lookback_days,
            field="lookback_days",
            minimum=LOOKBACK_MIN_DAYS,
            maximum=LOOKBACK_MAX_DAYS,
        )
        iv = sanitize_interval(interval or settings.interval)
        md = sanitize_positive_int(
            min_days if min_days is not None else settings.min_days,
            field="min_days",
            minimum=MIN_DAYS_MINIMUM,
            maximum=lb,
        )
    except SanitizationError as exc:
        _handle_error(exc, event="metrics_validation", endpoint="metrics")
    if met not in METRIC_REGISTRY:
        error = HVError(
            f"Unknown metric '{met}'",
            code=ErrorCode.VALIDATION,
            context={"metric": met},
        )
        _handle_error(error, event="metrics_validation", endpoint="metrics")
    try:
        prices = download_price_history(ticker_list, lb, interval=iv)
    except HVError as error:
        _handle_error(error, event="metrics_price_failure", endpoint="metrics")
    except Exception as exc:
        error = wrap_error(
            exc,
            IntegrationError,
            message="Failed to download price history",
            context={"tickers": ticker_list, "interval": iv},
        )
        _handle_error(error, event="metrics_price_failure", endpoint="metrics")
    func = METRIC_REGISTRY[met]
    try:
        result = func(prices, tickers=ticker_list, min_periods=md, interval=iv)
    except Exception as exc:
        error = wrap_error(
            exc,
            ComputeError,
            message="Metric computation failed",
            context={"metric": met, "tickers": len(ticker_list)},
        )
        _handle_error(error, event="metrics_failure", endpoint="metrics")
    return result.to_dict(orient="records")


def main() -> None:
    """Run a development server using :mod:`uvicorn`.

    This mirrors running ``uvicorn highest_volatility.app.api:app``.
    """

    import uvicorn

    uvicorn.run("highest_volatility.app.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
