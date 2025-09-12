"""On-disk cache for price data using Parquet and JSON manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import json

import pandas as pd

CACHE_ROOT = Path(".cache/prices")


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


def load_cached(ticker: str, interval: str) -> Tuple[Optional[pd.DataFrame], Optional[Manifest]]:
    """Load cached prices and manifest for ``ticker``/``interval``."""

    parquet_path, manifest_path = _paths(ticker, interval)
    if not parquet_path.exists() or not manifest_path.exists():
        return None, None

    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    manifest_data = json.loads(manifest_path.read_text())
    manifest = Manifest(**manifest_data)
    return df, manifest


def save_cache(ticker: str, interval: str, df: pd.DataFrame, source: str) -> Manifest:
    """Persist ``df`` and manifest to disk."""

    if df.empty:
        raise ValueError("Cannot cache empty DataFrame")

    df = df.sort_index()
    parquet_path, manifest_path = _paths(ticker, interval)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_parquet = parquet_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_parquet)
    tmp_parquet.replace(parquet_path)

    manifest = Manifest(
        ticker=ticker,
        interval=interval,
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        rows=len(df),
        source=source,
        version=1,
        updated_at=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    )
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(asdict(manifest)))
    tmp_manifest.replace(manifest_path)
    return manifest
