"""On-disk cache for price data using Parquet and JSON manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import json
import os
from io import BytesIO

import pandas as pd
import requests  # type: ignore[import]
from highest_volatility.pipeline import validate_cache
from src.security.validation import (
    SanitizationError,
    sanitize_interval,
    sanitize_single_ticker,
)

# Default on-disk cache root. Use project-visible folder as requested.
CACHE_ROOT = Path("cache/prices")


@dataclass
class Manifest:
    ticker: str
    interval: str
    start: str
    end: str
    rows: int
    source: str
    version: int
    updated_at: str


def _paths(ticker: str, interval: str) -> Tuple[Path, Path]:
    base = CACHE_ROOT / interval
    return base / f"{ticker}.parquet", base / f"{ticker}.json"


def _hydrate_from_api(ticker: str, interval: str) -> None:
    """Attempt to download cached data from the public API.

    The API base URL is taken from the ``HV_API_BASE_URL`` environment
    variable. If unavailable or any step fails, the function silently returns.
    """

    base_url = os.getenv("HV_API_BASE_URL")
    if not base_url:
        return
    url = f"{base_url.rstrip('/')}/prices/{ticker}?interval={interval}&fmt=parquet"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return
        df = pd.read_parquet(BytesIO(r.content))
        save_cache(ticker, interval, df, source="api")
    except Exception:
        return


def load_cached(ticker: str, interval: str) -> Tuple[Optional[pd.DataFrame], Optional[Manifest]]:
    """Load cached prices and manifest for ``ticker``/``interval``."""

    try:
        ticker = sanitize_single_ticker(ticker)
        interval = sanitize_interval(interval)
    except SanitizationError as exc:
        raise ValueError(f"Invalid cache lookup: {exc}") from exc

    parquet_path, manifest_path = _paths(ticker, interval)
    if not parquet_path.exists() or not manifest_path.exists():
        if os.getenv("GITHUB_ACTIONS") != "true":
            _hydrate_from_api(ticker, interval)
        if not parquet_path.exists() or not manifest_path.exists():
            return None, None

    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    manifest_data = json.loads(manifest_path.read_text())
    manifest = Manifest(**manifest_data)
    return df, manifest


def save_cache(
    ticker: str,
    interval: str,
    df: pd.DataFrame,
    source: str,
    *,
    validate: bool = True,
) -> Manifest:
    """Persist ``df`` and manifest to disk."""

    if df.empty:
        raise ValueError("Cannot cache empty DataFrame")

    try:
        ticker = sanitize_single_ticker(ticker)
        interval = sanitize_interval(interval)
    except SanitizationError as exc:
        raise ValueError(f"Invalid cache write: {exc}") from exc

    df = df.sort_index()
    manifest = Manifest(
        ticker=ticker,
        interval=interval,
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        rows=len(df),
        source=source,
        version=1,
        updated_at=
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    if validate:
        validate_cache(df, manifest)

    parquet_path, manifest_path = _paths(ticker, interval)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_parquet = parquet_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_parquet)
    tmp_parquet.replace(parquet_path)

    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(asdict(manifest)))
    tmp_manifest.replace(manifest_path)
    return manifest
