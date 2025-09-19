from __future__ import annotations

import json
from io import BytesIO

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Scope

from src.cache.store import load_cached
from highest_volatility.errors import CacheError, ErrorCode, HVError, wrap_error
from highest_volatility.logging import get_logger, log_exception
from highest_volatility.storage.ticker_cache import load_cached_fortune
from src.security.validation import (
    SanitizationError,
    sanitize_download_format,
    sanitize_interval,
    sanitize_single_ticker,
)


class SecureHeadersMiddleware(BaseHTTPMiddleware):
    """Apply baseline security headers to all responses."""

    async def dispatch(self, request: Scope, call_next):  # type: ignore[override]
        response = await call_next(request)
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload"
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        return response


app = FastAPI(title="Highest Volatility Data API")
app.add_middleware(SecureHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


logger = get_logger(__name__, component="data_api")


_STATUS_BY_CODE = {
    ErrorCode.VALIDATION: 400,
    ErrorCode.CACHE: 503,
    ErrorCode.DATA_SOURCE: 502,
}


def _handle_error(error: HVError, *, event: str, endpoint: str) -> None:
    log_exception(logger, error, event=event, context={"endpoint": endpoint})
    status = _STATUS_BY_CODE.get(error.code, 500)
    raise HTTPException(status_code=status, detail=error.user_message)


@app.get("/prices/{ticker}")
def get_prices(
    ticker: str,
    interval: str = "1d",
    fmt: str = "json",
):
    try:
        ticker = sanitize_single_ticker(ticker)
        interval = sanitize_interval(interval)
        fmt = sanitize_download_format(fmt)
    except SanitizationError as exc:  # pragma: no cover - defensive
        _handle_error(exc, event="prices_validation", endpoint="prices")

    try:
        df, _ = load_cached(ticker, interval)
    except HVError as error:  # pragma: no cover - defensive
        _handle_error(error, event="prices_cache_failure", endpoint="prices")
    except Exception as exc:  # pragma: no cover - defensive
        error = wrap_error(
            exc,
            CacheError,
            message="Failed to load cached prices",
            context={"ticker": ticker, "interval": interval},
        )
        _handle_error(error, event="prices_cache_failure", endpoint="prices")
    if df is None:
        raise HTTPException(status_code=404, detail="Ticker not found")

    if fmt == "json":
        return json.loads(df.to_json(orient="split", date_format="iso"))
    if fmt == "parquet":
        buf = BytesIO()
        try:
            df.to_parquet(buf)
        except Exception as exc:  # pragma: no cover - defensive
            error = wrap_error(
                exc,
                CacheError,
                message="Failed to serialise parquet response",
                context={"ticker": ticker, "interval": interval},
            )
            _handle_error(error, event="prices_serialisation_failure", endpoint="prices")
        filename = f"{ticker}_{interval}.parquet"
        return Response(
            buf.getvalue(),
            media_type="application/x-parquet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    raise HTTPException(status_code=400, detail="Unknown format")


@app.get("/fortune-tickers")
def get_fortune_tickers(fmt: str = "json"):
    df = load_cached_fortune()
    if df is None:
        raise HTTPException(status_code=404, detail="Fortune list not available")

    try:
        fmt = sanitize_download_format(fmt)
    except SanitizationError as exc:  # pragma: no cover - defensive
        _handle_error(exc, event="fortune_validation", endpoint="fortune-tickers")

    if fmt == "json":
        return {"tickers": df["ticker"].dropna().tolist()}
    if fmt == "parquet":
        buf = BytesIO()
        try:
            df.to_parquet(buf)
        except Exception as exc:  # pragma: no cover - defensive
            error = wrap_error(
                exc,
                CacheError,
                message="Failed to serialise fortune parquet",
                context={"rows": len(df)},
            )
            _handle_error(error, event="fortune_serialisation_failure", endpoint="fortune-tickers")
        return Response(
            buf.getvalue(),
            media_type="application/x-parquet",
            headers={"Content-Disposition": 'attachment; filename="fortune_tickers.parquet"'},
        )
    raise HTTPException(status_code=400, detail="Unknown format")
