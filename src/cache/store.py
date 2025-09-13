"""On-disk cache for price data using Parquet and JSON manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import json
import os
import subprocess

import pandas as pd
import requests

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


def _hydrate_from_repo(ticker: str, interval: str, parquet_path: Path, manifest_path: Path) -> None:
    """Attempt to download cached data from the GitHub repository.

    This is used for local runs to hydrate the on-disk cache before falling
    back to fresh downloads.  It is a best-effort helper and silently returns
    if any step fails or if the files are unavailable upstream.
    """

    try:
        remote_url = (
            subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return

    if remote_url.endswith(".git"):
        remote_url = remote_url[:-4]
    if remote_url.startswith("git@github.com:"):
        owner_repo = remote_url.split("git@github.com:", 1)[1]
    elif remote_url.startswith("https://github.com/"):
        owner_repo = remote_url.split("https://github.com/", 1)[1]
    else:
        return

    owner_repo = owner_repo.strip("/")
    branch = os.getenv("CACHE_REPO_BRANCH", "main")
    base = (
        f"https://raw.githubusercontent.com/{owner_repo}/{branch}/.cache/prices/{interval}/{ticker}"
    )

    targets = [
        (parquet_path, base + ".parquet", "b"),
        (manifest_path, base + ".json", "t"),
    ]
    for path, url, mode in targets:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "b":
                path.write_bytes(r.content)
            else:
                path.write_text(r.text)
        except Exception:
            return


def load_cached(ticker: str, interval: str) -> Tuple[Optional[pd.DataFrame], Optional[Manifest]]:
    """Load cached prices and manifest for ``ticker``/``interval``."""

    parquet_path, manifest_path = _paths(ticker, interval)
    if not parquet_path.exists() or not manifest_path.exists():
        if os.getenv("GITHUB_ACTIONS") != "true":
            _hydrate_from_repo(ticker, interval, parquet_path, manifest_path)
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
        updated_at=
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(asdict(manifest)))
    tmp_manifest.replace(manifest_path)
    return manifest
