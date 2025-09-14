from __future__ import annotations

import json
from io import BytesIO

from fastapi import FastAPI, HTTPException, Response

from cache.store import load_cached
from highest_volatility.storage.ticker_cache import load_cached_fortune

app = FastAPI(title="Highest Volatility Data API")


@app.get("/prices/{ticker}")
def get_prices(
    ticker: str,
    interval: str = "1d",
    fmt: str = "json",
):
    ticker = ticker.upper()
    df, _ = load_cached(ticker, interval)
    if df is None:
        raise HTTPException(status_code=404, detail="Ticker not found")

    if fmt == "json":
        return json.loads(df.to_json(orient="split", date_format="iso"))
    if fmt == "parquet":
        buf = BytesIO()
        df.to_parquet(buf)
        return Response(
            buf.getvalue(),
            media_type="application/x-parquet",
            headers={"Content-Disposition": f"attachment; filename={ticker}_{interval}.parquet"},
        )
    raise HTTPException(status_code=400, detail="Unknown format")


@app.get("/fortune-tickers")
def get_fortune_tickers(fmt: str = "json"):
    df = load_cached_fortune()
    if df is None:
        raise HTTPException(status_code=404, detail="Fortune list not available")

    if fmt == "json":
        return {"tickers": df["ticker"].dropna().tolist()}
    if fmt == "parquet":
        buf = BytesIO()
        df.to_parquet(buf)
        return Response(
            buf.getvalue(),
            media_type="application/x-parquet",
            headers={"Content-Disposition": "attachment; filename=fortune_tickers.parquet"},
        )
    raise HTTPException(status_code=400, detail="Unknown format")
