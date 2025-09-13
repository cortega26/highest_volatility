"""FastAPI application exposing Highest Volatility functionality.

Run with:
    uvicorn highest_volatility.app.api:app --reload
or:
    python -m highest_volatility.app.api

Environment variables prefixed with ``HV_`` (e.g. ``HV_LOOKBACK_DAYS``)
can override default configuration values.
"""

from __future__ import annotations

import json

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseSettings

from highest_volatility.app.cli import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_DAYS,
    DEFAULT_TOP_N,
)
from highest_volatility.compute.metrics import METRIC_REGISTRY
from highest_volatility.ingest.prices import download_price_history
from highest_volatility.universe import build_universe


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

    class Config:
        env_prefix = "HV_"


def get_settings() -> Settings:
    return Settings()


app = FastAPI(title="Highest Volatility API")


@app.get("/universe")
def universe_endpoint(
    top_n: int | None = None,
    settings: Settings = Depends(get_settings),
):
    """Return a validated ticker universe."""

    limit = top_n or settings.top_n
    tickers, fortune = build_universe(limit, validate=True)
    return {"tickers": tickers, "fortune": fortune.to_dict(orient="records")}


@app.get("/prices")
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
