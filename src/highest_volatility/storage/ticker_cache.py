"""Simple on-disk cache for the Fortune 500 tickers list.

Stores a CSV under ``.cache/tickers/fortune_500.csv`` with columns
``rank``, ``company``, ``ticker``.  The file modification time is used
as freshness indicator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


CACHE_PATH = Path(".cache") / "tickers" / "fortune_500.csv"
REPO_FALLBACK = Path("fortune500_tickers.csv")


@dataclass
class CacheInfo:
    path: Path
    modified: datetime
    age_days: float


def _info(path: Path) -> Optional[CacheInfo]:
    try:
        st = path.stat()
        mtime = datetime.fromtimestamp(st.st_mtime)
        age = (datetime.now() - mtime).total_seconds() / 86400.0
        return CacheInfo(path=path, modified=mtime, age_days=age)
    except Exception:
        return None


def load_cached_fortune(max_age_days: int = 30, *, min_rows: int = 100) -> Optional[pd.DataFrame]:
    """Return cached Fortune list if present and fresh enough.

    Parameters
    ----------
    max_age_days:
        Maximum age for the cache to be considered fresh.
    min_rows:
        Minimum number of rows expected (guards against truncated files).
    """

    # Try dedicated cache path first
    candidates = []
    ci = _info(CACHE_PATH)
    if ci is not None and ci.age_days <= max_age_days:
        candidates.append(ci.path)
    # Fall back to a repo-tracked file if present
    ci_repo = _info(REPO_FALLBACK)
    if ci_repo is not None and REPO_FALLBACK.exists() and REPO_FALLBACK not in candidates:
        candidates.append(REPO_FALLBACK)
    if not candidates:
        return None
    # Load the first viable candidate
    try:
        df = pd.read_csv(candidates[0])
        cols = {c.lower() for c in df.columns}
        if not {"rank", "company", "ticker"}.issubset(cols):
            return None
        if len(df) < min_rows:
            return None
        # Normalize columns
        df = df.rename(columns=str.lower)
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
        return df.dropna(subset=["company", "ticker"]).reset_index(drop=True)
    except Exception:
        return None


def save_cached_fortune(df: pd.DataFrame) -> None:
    """Persist the Fortune list to ``CACHE_PATH``.

    Parent directories are created as needed.
    """

    path = CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


__all__ = ["load_cached_fortune", "save_cached_fortune", "CACHE_PATH"]
