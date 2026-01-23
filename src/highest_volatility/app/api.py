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
import hashlib
import json
import re
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any, Iterable, cast

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse, Response
from fastapi.responses import ORJSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

import pandas as pd

from highest_volatility.app.cli import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_DAYS,
    DEFAULT_TOP_N,
)
from highest_volatility.compute.metrics import METRIC_REGISTRY
from highest_volatility.errors import (
    CacheError,
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
from highest_volatility.storage.annotation_store import AnnotationStore
from highest_volatility.security.validation import (
    SanitizationError,
    sanitize_interval,
    sanitize_metric,
    sanitize_multiple_tickers,
    sanitize_positive_int,
)
from highest_volatility.config.paths import expand_env_path


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
    require_redis_for_readyz: bool = True
    annotations_db_path: str = Field(
        "cache/annotations.db",
        validation_alias=AliasChoices(
            "HV_ANNOTATIONS_DB",
            "HV_ANNOTATIONS_DB_PATH",
            "ANNOTATIONS_DB",
            "ANNOTATIONS_DB_PATH",
        ),
    )

    class Config:
        env_prefix = "HV_"

    @field_validator("annotations_db_path", mode="before")
    @classmethod
    def _expand_annotations_db_path(cls, value: str) -> str:
        return str(expand_env_path(str(value), field="HV_ANNOTATIONS_DB"))


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


REQUEST_COUNT = Counter(
    "hv_fastapi_requests_total",
    "Total count of HTTP requests.",
    ["method", "path", "status"],
)
REQUEST_LATENCY_MS = Histogram(
    "hv_fastapi_request_latency_ms",
    "FastAPI request latency in milliseconds.",
    ["method", "path"],
    buckets=(50, 100, 200, 300, 500, 1000, 2000, 5000, 10000),
)


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


app = FastAPI(title="Highest Volatility API", default_response_class=ORJSONResponse)
app.add_middleware(SecureHeadersMiddleware)

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
INDEX_PATH = WEB_ROOT / "index.html"
SERVICE_WORKER_PATH = WEB_ROOT / "service-worker.js"
MANIFEST_PATH = WEB_ROOT / "manifest.webmanifest"
if WEB_ROOT.exists():
    app.mount("/web", StaticFiles(directory=WEB_ROOT, html=True), name="web")


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Attach cache metadata and entity tags for selected endpoints."""

    _TTL_ATTR_BY_PATH = {
        "/universe": "cache_ttl_universe",
        "/prices": "cache_ttl_prices",
        "/metrics": "cache_ttl_metrics",
    }

    def __init__(self, app: FastAPI, *, settings: Settings) -> None:  # type: ignore[override]
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response = await call_next(request)
        ttl_attr = self._TTL_ATTR_BY_PATH.get(request.url.path)
        if ttl_attr is None:
            return response
        ttl = getattr(self._settings, ttl_attr, None)
        if ttl is None:
            return response
        if response.status_code >= 400:
            return response

        serialized = getattr(request.state, "serialized_payload", None)
        etag = getattr(request.state, "computed_etag", None)

        if serialized is not None:
            body_bytes = serialized.encode("utf-8")
        else:
            if hasattr(response, "body"):
                raw_body = response.body or b""
            else:
                chunks = [chunk async for chunk in response.body_iterator]
                raw_body = b"".join(chunks)
            try:
                canonical = json.dumps(
                    json.loads(raw_body.decode("utf-8")),
                    separators=(",", ":"),
                    sort_keys=True,
                    ensure_ascii=False,
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                body_bytes = raw_body
            else:
                body_bytes = canonical.encode("utf-8")

        if etag is None:
            etag = hashlib.sha256(body_bytes).hexdigest()

        background = getattr(response, "background", None)
        media_type = getattr(response, "media_type", None)
        status_code = response.status_code
        new_response = Response(content=body_bytes, status_code=status_code, media_type=media_type)
        for key, value in response.headers.items():
            if key.lower() == "content-length":
                continue
            new_response.headers[key] = value
        if background is not None:
            new_response.background = background
        response = new_response

        quoted_etag = f'"{etag}"'
        if_none_match = request.headers.get("if-none-match")
        if if_none_match:
            candidates = {candidate.strip() for candidate in if_none_match.split(",")}
            match_candidates = candidates & {quoted_etag, etag, f'W/{quoted_etag}'}
            if match_candidates:
                response.status_code = status.HTTP_304_NOT_MODIFIED
                response.body = b""

        response.headers["ETag"] = quoted_etag
        response.headers["Cache-Control"] = f"public, max-age={ttl}"
        response.headers["Surrogate-Control"] = f"max-age={ttl}"
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        response.headers["Expires"] = format_datetime(expires_at, usegmt=True)
        return response


app.add_middleware(CacheControlMiddleware, settings=settings)
app.add_middleware(GZipMiddleware, minimum_size=128)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Capture request metrics for Prometheus scraping."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        start = time.perf_counter()
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            REQUEST_COUNT.labels(
                method=request.method,
                path=request.url.path,
                status=str(status_code),
            ).inc()
            REQUEST_LATENCY_MS.labels(
                method=request.method,
                path=request.url.path,
            ).observe(duration_ms)


app.add_middleware(PrometheusMiddleware)


class AnnotationPayload(BaseModel):
    """Request payload for annotation updates."""

    note: str = Field(..., min_length=1, max_length=2000)
    client_timestamp: datetime


def _ensure_annotation_store() -> tuple[AnnotationStore, asyncio.Lock]:
    store = getattr(app.state, "annotation_store", None)
    lock = getattr(app.state, "annotation_lock", None)
    if store is None:
        store = AnnotationStore(Path(settings.annotations_db_path))
        app.state.annotation_store = store
    if lock is None:
        lock = asyncio.Lock()
        app.state.annotation_lock = lock
    return store, lock


def _serve_file(path: Path, *, media_type: str | None = None) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    return FileResponse(path, media_type=media_type)


@app.get("/", include_in_schema=False)
async def web_index() -> FileResponse:
    """Serve the progressive web app shell."""

    return _serve_file(INDEX_PATH, media_type="text/html")


@app.get("/manifest.webmanifest", include_in_schema=False)
async def web_manifest() -> FileResponse:
    """Expose the web manifest for installability."""

    return _serve_file(MANIFEST_PATH, media_type="application/manifest+json")


@app.get("/service-worker.js", include_in_schema=False)
async def web_service_worker() -> FileResponse:
    """Serve the service worker from the application root."""

    return _serve_file(SERVICE_WORKER_PATH, media_type="application/javascript")

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
    try:
        await FastAPICache.clear()
    except AssertionError:
        pass
    app.state.cache_health = "initializing"
    app.state.cache_health_detail = None
    client = redis.from_url(settings.redis_url, encoding="utf8", decode_responses=True)
    app.state.redis_client = client
    app.state.redis_last_error: str | None = None
    try:
        await client.ping()
    except (RedisConnectionError, OSError, TimeoutError) as exc:
        logger.warning(
            "Redis backend unavailable; falling back to in-memory cache", exc_info=exc
        )
        FastAPICache.init(InMemoryBackend(), prefix="hv-cache-fallback")
        app.state.cache_health = "degraded"
        app.state.cache_health_detail = "redis_unreachable"
        app.state.redis_last_error = str(exc)
        app.state.redis_client = None
        try:
            await client.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
    else:
        FastAPICache.init(RedisBackend(client), prefix="hv-cache")
        app.state.cache_health = "healthy"
    app.state.cache_refresh_task = asyncio.create_task(
        schedule_cache_refresh(
            interval=settings.interval,
            lookback_days=settings.lookback_days,
            delay=settings.cache_refresh_interval,
        )
    )
    try:
        _ensure_annotation_store()
    except Exception as exc:  # pragma: no cover - startup should not fail on annotations
        logger.warning("Annotation store unavailable", exc_info=exc)
        app.state.annotation_store = None


@app.on_event("shutdown")
async def on_shutdown() -> None:
    task = getattr(app.state, "cache_refresh_task", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Cache refresh task raised during shutdown", exc_info=exc
            )
    try:
        await FastAPICache.clear()
    except AssertionError:
        pass
    app.state.cache_health = "stopped"
    client = getattr(app.state, "redis_client", None)
    if client is not None:
        try:
            await client.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
    app.state.redis_client = None
    app.state.redis_last_error = None


async def _check_redis_health() -> tuple[bool, dict[str, Any]]:
    """Probe the configured Redis backend and return structured status."""

    client = getattr(app.state, "redis_client", None)
    last_error = getattr(app.state, "redis_last_error", None)
    if client is None:
        detail = "redis_client_unavailable"
        return False, {"status": "down", "detail": detail, "last_error": last_error}
    try:
        await client.ping()
    except (RedisConnectionError, OSError, TimeoutError) as exc:
        app.state.redis_last_error = str(exc)
        return False, {
            "status": "down",
            "detail": "redis_unreachable",
            "last_error": str(exc),
        }
    else:
        app.state.redis_last_error = None
        return True, {"status": "up", "detail": None, "last_error": None}


def _check_cache_refresh_task() -> tuple[bool, dict[str, Any]]:
    """Return health information for the cache refresh background task."""

    task = getattr(app.state, "cache_refresh_task", None)
    if task is None:
        return False, {"status": "missing", "detail": "cache_refresh_task_missing"}
    if task.cancelled():
        return False, {"status": "cancelled", "detail": "cache_refresh_task_cancelled"}
    if task.done():
        try:
            exc = task.exception()
        except asyncio.CancelledError:  # pragma: no cover - defensive
            return False, {"status": "cancelled", "detail": "cache_refresh_task_cancelled"}
        if exc is not None:
            return False, {
                "status": "error",
                "detail": f"cache_refresh_task_failed:{exc.__class__.__name__}",
            }
        return True, {"status": "completed", "detail": None}
    return True, {"status": "running", "detail": None}


@app.get("/healthz", tags=["operations"], response_class=JSONResponse)
async def healthz() -> JSONResponse:
    """Liveness endpoint reporting process and background worker health."""

    redis_ok, redis_report = await _check_redis_health()
    task_ok, task_report = _check_cache_refresh_task()
    cache_state = getattr(app.state, "cache_health", "unknown")
    overall_ok = task_ok and cache_state != "stopped"
    status_code = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    body = {
        "status": "ok" if overall_ok else "error",
        "redis": redis_report | {"cache_state": cache_state},
        "cache_refresh_task": task_report,
    }
    if not redis_ok:
        body["redis"]["detail"] = redis_report.get("detail", "redis_unreachable")
    return JSONResponse(status_code=status_code, content=body)


@app.get("/readyz", tags=["operations"], response_class=JSONResponse)
async def readyz() -> JSONResponse:
    """Readiness endpoint ensuring dependencies are reachable."""

    redis_ok, redis_report = await _check_redis_health()
    task_ok, task_report = _check_cache_refresh_task()
    cache_state = getattr(app.state, "cache_health", "unknown")
    redis_required = settings.require_redis_for_readyz
    if redis_required:
        cache_ready = cache_state == "healthy"
    else:
        cache_ready = cache_state in {"healthy", "degraded"}
    overall_ok = task_ok and cache_ready and (redis_ok or not redis_required)
    status_code = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    body = {
        "status": "ok" if overall_ok else "error",
        "redis": redis_report | {"cache_state": cache_state},
        "cache_refresh_task": task_report,
    }
    if not overall_ok:
        body["redis"]["detail"] = redis_report.get("detail", "redis_unreachable")
    return JSONResponse(status_code=status_code, content=body)


@app.get("/metrics/prometheus", include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics for scraping."""

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/annotations", tags=["annotations"])
async def list_annotations(request: Request) -> Any:
    """Return all stored annotations."""

    try:
        store, lock = _ensure_annotation_store()
    except Exception as exc:
        error = wrap_error(
            exc,
            CacheError,
            message="Failed to initialize annotation store",
        )
        _handle_error(error, event="annotations_store_init_failed", endpoint="annotations")
    async with lock:
        try:
            records = await asyncio.to_thread(store.list_annotations)
        except Exception as exc:
            error = wrap_error(
                exc,
                CacheError,
                message="Failed to load annotations",
            )
            _handle_error(error, event="annotations_read_failed", endpoint="annotations")
    return _prepare_response_payload(request, [record.__dict__ for record in records])


@app.get("/annotations/history/{ticker}", tags=["annotations"])
async def annotation_history(request: Request, ticker: str) -> Any:
    """Return the audit trail for a single ticker."""

    try:
        sanitized = sanitize_multiple_tickers(ticker, max_tickers=1)[0]
    except SanitizationError as exc:
        _handle_error(exc, event="annotations_validation", endpoint="annotations")
    try:
        store, lock = _ensure_annotation_store()
    except Exception as exc:
        error = wrap_error(
            exc,
            CacheError,
            message="Failed to initialize annotation store",
        )
        _handle_error(error, event="annotations_store_init_failed", endpoint="annotations")
    async with lock:
        try:
            entries = await asyncio.to_thread(store.load_history, sanitized)
        except Exception as exc:
            error = wrap_error(
                exc,
                CacheError,
                message="Failed to load annotation history",
            )
            _handle_error(error, event="annotations_history_failed", endpoint="annotations")
    return _prepare_response_payload(request, [record.__dict__ for record in entries])


@app.put("/annotations/{ticker}", tags=["annotations"])
async def upsert_annotation(request: Request, ticker: str, payload: AnnotationPayload) -> Any:
    """Persist a ticker annotation and record its revision history."""

    try:
        sanitized = sanitize_multiple_tickers(ticker, max_tickers=1)[0]
    except SanitizationError as exc:
        _handle_error(exc, event="annotations_validation", endpoint="annotations")

    note = payload.note.strip()
    if not note:
        error = SanitizationError("Note cannot be empty.", field="note")
        _handle_error(error, event="annotations_validation", endpoint="annotations")

    client_timestamp = payload.client_timestamp
    if client_timestamp.tzinfo is None:
        client_timestamp = client_timestamp.replace(tzinfo=timezone.utc)
    else:
        client_timestamp = client_timestamp.astimezone(timezone.utc)
    server_timestamp = datetime.now(timezone.utc)

    try:
        store, lock = _ensure_annotation_store()
    except Exception as exc:
        error = wrap_error(
            exc,
            CacheError,
            message="Failed to initialize annotation store",
        )
        _handle_error(error, event="annotations_store_init_failed", endpoint="annotations")
    async with lock:
        try:
            record = await asyncio.to_thread(
                store.upsert,
                ticker=sanitized,
                note=note,
                updated_at=server_timestamp,
                client_timestamp=client_timestamp,
            )
        except Exception as exc:
            error = wrap_error(
                exc,
                CacheError,
                message="Failed to persist annotation",
            )
            _handle_error(error, event="annotations_write_failed", endpoint="annotations")
    return _prepare_response_payload(request, record.__dict__)


def _prepare_response_payload(request: Request, payload: Any) -> Any:
    """Normalise ``payload`` and stash canonical JSON for caching middleware."""

    encoded = jsonable_encoder(payload)
    serialized = json.dumps(
        encoded,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    )
    request.state.serialized_payload = serialized
    request.state.computed_etag = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return encoded


_PRICE_COLUMN_PATTERN = re.compile(r"^[A-Za-z0-9 _\-]+$")


def _parse_column_filters(raw: str | None) -> list[str]:
    """Return a sanitized list of column selectors derived from ``raw``."""

    if raw is None:
        return []
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise SanitizationError("At least one column must be supplied.", field="columns")
    invalid = [part for part in parts if not _PRICE_COLUMN_PATTERN.fullmatch(part)]
    if invalid:
        raise SanitizationError("Column name contains invalid characters.", field="columns")
    seen: set[str] = set()
    normalized: list[str] = []
    for part in parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            normalized.append(part)
    return normalized


def _resolve_column_labels(frame: pd.DataFrame) -> dict[str, Any]:
    """Map lower-case column selectors to canonical labels for ``frame``."""

    if isinstance(frame.columns, pd.MultiIndex):
        level = frame.columns.get_level_values(0)
        ordered: list[Any] = []
        seen: set[str] = set()
        for label in level:
            lower = str(label).lower()
            if lower not in seen:
                seen.add(lower)
                ordered.append(label)
        return {str(label).lower(): label for label in ordered}
    return {str(label).lower(): label for label in frame.columns}


def _filter_price_columns(frame: pd.DataFrame, selectors: Iterable[str]) -> pd.DataFrame:
    """Return ``frame`` restricted to the requested ``selectors``."""

    selected = list(selectors)
    if not selected:
        return frame
    mapping = _resolve_column_labels(frame)
    missing = [item for item in selected if item.lower() not in mapping]
    if missing:
        joined = ", ".join(sorted(missing))
        raise SanitizationError(f"Unknown columns requested: {joined}.", field="columns")
    if isinstance(frame.columns, pd.MultiIndex):
        allowed = {mapping[item.lower()] for item in selected}
        mask = frame.columns.get_level_values(0).isin(allowed)
        return frame.loc[:, mask]
    ordered_labels = [mapping[item.lower()] for item in selected]
    return frame.loc[:, ordered_labels]


@app.get("/universe")
@cache(expire=settings.cache_ttl_universe)
def universe_endpoint(
    request: Request,
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
    payload = {"tickers": tickers, "fortune": fortune.to_dict(orient="records")}
    return _prepare_response_payload(request, payload)


@app.get("/prices")
@cache(expire=settings.cache_ttl_prices)
def prices_endpoint(
    request: Request,
    tickers: str,
    lookback_days: int | None = None,
    interval: str | None = None,
    prepost: bool | None = None,
    columns: str | None = None,
    settings: Settings = Depends(get_settings),
):
    """Return price history for ``tickers``."""

    column_filters: list[str]
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
        column_filters = _parse_column_filters(columns)
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
    try:
        filtered = _filter_price_columns(df, column_filters)
    except SanitizationError as exc:
        _handle_error(exc, event="prices_validation", endpoint="prices")
    if filtered.empty:
        payload = {"data": []}
    else:
        payload = json.loads(filtered.to_json(orient="split", date_format="iso"))
    return _prepare_response_payload(request, payload)


@app.get("/metrics")
@cache(expire=settings.cache_ttl_metrics)
def metrics_endpoint(
    request: Request,
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
    payload = result.to_dict(orient="records")
    return _prepare_response_payload(request, payload)


def main() -> None:
    """Run a development server using :mod:`uvicorn`.

    This mirrors running ``uvicorn highest_volatility.app.api:app``.
    """

    import uvicorn

    uvicorn.run("highest_volatility.app.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
