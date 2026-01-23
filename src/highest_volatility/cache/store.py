"""On-disk cache for price data using Parquet and JSON manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple
import json
import os

import pandas as pd
import requests  # type: ignore[import]
from highest_volatility.pipeline import validate_cache
from highest_volatility.config.paths import expand_env_path

# Default on-disk cache root. Use project-visible folder unless overridden.
def _resolve_cache_root() -> Path:
    env_root = os.getenv("HV_CACHE_ROOT")
    if env_root:
        return expand_env_path(env_root, field="HV_CACHE_ROOT")
    return Path("cache/prices")

CACHE_ROOT = _resolve_cache_root()

# Increment when cache schema/content changes invalidate on-disk data.
CACHE_VERSION = 2


def _security_validation():
    from highest_volatility.security import validation as security_validation

    return security_validation


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


def _frame_from_split_payload(payload: object) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()
    data = payload.get("data")
    if not data:
        return pd.DataFrame()
    columns = payload.get("columns", [])
    index = payload.get("index", [])
    if not isinstance(columns, list) or not isinstance(index, list):
        return pd.DataFrame()
    if columns and isinstance(columns[0], (list, tuple)):
        columns = pd.MultiIndex.from_tuples([tuple(col) for col in columns])
    frame = pd.DataFrame(data=data, columns=columns)
    if index:
        frame.index = pd.to_datetime(index)
    return frame


def _extract_single_ticker_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame.copy()
    ticker_key = str(ticker)
    level1 = frame.columns.get_level_values(1).astype(str)
    if (level1 == ticker_key).any():
        selected = frame.loc[:, level1 == ticker_key].copy()
        selected.columns = selected.columns.droplevel(1)
        return selected
    level0 = frame.columns.get_level_values(0).astype(str)
    if (level0 == ticker_key).any():
        selected = frame.loc[:, level0 == ticker_key].copy()
        selected.columns = selected.columns.droplevel(0)
        return selected
    return pd.DataFrame()


def _hydrate_from_api(ticker: str, interval: str) -> None:
    """Attempt to download cached data from the public API.

    The API base URL is taken from the ``HV_API_BASE_URL`` environment
    variable. If unavailable or any step fails, the function silently returns.
    """

    base_url = os.getenv("HV_API_BASE_URL")
    if not base_url:
        return
    url = f"{base_url.rstrip('/')}/prices"
    try:
        response = requests.get(
            url,
            params={"tickers": ticker, "interval": interval},
            timeout=10,
        )
        if response.status_code != 200:
            return
        payload = response.json()
        frame = _frame_from_split_payload(payload)
        if frame.empty:
            return
        frame = _extract_single_ticker_frame(frame, ticker)
        if frame.empty:
            return
        save_cache(ticker, interval, frame, source="api")
    except Exception:
        return


def load_cached(ticker: str, interval: str) -> Tuple[Optional[pd.DataFrame], Optional[Manifest]]:
    """Load cached prices and manifest for ``ticker``/``interval``."""

    security_validation = _security_validation()

    try:
        ticker = security_validation.sanitize_single_ticker(ticker)
        interval = security_validation.sanitize_interval(interval)
    except security_validation.SanitizationError as exc:
        raise ValueError(f"Invalid cache lookup: {exc}") from exc

    parquet_path, manifest_path = _paths(ticker, interval)
    if not parquet_path.exists() or not manifest_path.exists():
        if os.getenv("GITHUB_ACTIONS") != "true":
            _hydrate_from_api(ticker, interval)
        if not parquet_path.exists() or not manifest_path.exists():
            return None, None

    try:
        manifest_data = json.loads(manifest_path.read_text())
        manifest = Manifest(**manifest_data)
    except (json.JSONDecodeError, TypeError, ValueError):
        for path in (parquet_path, manifest_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return None, None

    if manifest.version < CACHE_VERSION:
        for path in (parquet_path, manifest_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return None, None

    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df, manifest


def save_cache(
    ticker: str,
    interval: str,
    df: pd.DataFrame,
    source: str,
    *,
    validate: bool = True,
    allowed_gaps: Optional[Iterable[str | pd.Timestamp]] = None,
) -> Manifest:
    """Persist ``df`` and manifest to disk."""

    if df.empty:
        raise ValueError("Cannot cache empty DataFrame")

    security_validation = _security_validation()

    try:
        ticker = security_validation.sanitize_single_ticker(ticker)
        interval = security_validation.sanitize_interval(interval)
    except security_validation.SanitizationError as exc:
        raise ValueError(f"Invalid cache write: {exc}") from exc

    df = df.sort_index()
    manifest = Manifest(
        ticker=ticker,
        interval=interval,
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        rows=len(df),
        source=source,
        version=CACHE_VERSION,
        updated_at=
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    if validate:
        validate_cache(df, manifest, allowed_gaps=allowed_gaps)

    parquet_path, manifest_path = _paths(ticker, interval)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_parquet = parquet_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_parquet)
    tmp_parquet.replace(parquet_path)

    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(asdict(manifest)))
    tmp_manifest.replace(manifest_path)
    return manifest
