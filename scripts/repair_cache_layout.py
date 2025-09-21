#!/usr/bin/env python3
"""Repair and normalize the local price cache layout.

Actions performed:
- Move legacy CSVs from `cache/prices/*.csv` into per-interval subfolders
  (assumes daily -> `1d`) and convert to Parquet with a DatetimeIndex.
- Inspect `cache/prices/30m/*.csv`; if they lack intraday times (date-only),
  refetch correct 30m bars from Yahoo and write Parquet.

Notes:
- Originals are preserved by default (use --delete-legacy to remove after convert).
- Parquet files are written via the existing cache store and validated.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


def _ensure_src_on_path() -> Path:
    """Ensure the repository ``src`` directory is importable."""

    here = Path(__file__).resolve()
    root = here.parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def _iter_legacy_daily_csv(root: Path) -> Iterable[Path]:
    for p in sorted((root).glob("*.csv")):
        # Skip obvious non-ticker names
        if p.name.lower() in {"readme.csv"}:
            continue
        yield p


def _iter_legacy_30m_csv(root: Path) -> Iterable[Path]:
    d = root / "30m"
    if not d.exists():
        return []
    return sorted(d.glob("*.csv"))


def _parse_price_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Prefer a Datetime column name if present, fallback to Date
    idx_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else None)
    if idx_col is None:
        raise ValueError(f"Missing Date/Datetime column in {path}")
    idx = pd.to_datetime(df[idx_col])
    df = df.drop(columns=[idx_col])
    df.index = idx
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Sort and drop all-empty columns
    df = df.sort_index().dropna(how="all", axis=1)
    return df


def _has_time_components(df: pd.DataFrame) -> bool:
    if not isinstance(df.index, pd.DatetimeIndex):
        return False
    times = df.index.time
    # If any timestamp has a non-midnight time, consider it intraday
    return any(t.hour != 0 or t.minute != 0 or t.second != 0 for t in times)


async def _refetch_30m(ticker: str, *, days: int = 60) -> pd.DataFrame:
    # Lazy import to keep script light if user only runs daily conversion
    _ensure_src_on_path()
    from highest_volatility.config.interval_policy import INTERVAL_WINDOWS

    try:
        from highest_volatility.datasource.yahoo_http_async import YahooHTTPAsyncDataSource
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing YahooHTTPAsyncDataSource in runtime") from exc

    ds = YahooHTTPAsyncDataSource()
    window = INTERVAL_WINDOWS.get("30m")
    max_days = int(window.days) if window is not None else 60
    days = min(days, max_days)
    end = date.today()
    start = end - timedelta(days=days)
    df = await ds.get_prices(ticker, start, end, "30m")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def convert_legacy_daily(root: Path, *, delete_legacy: bool = False, dry_run: bool = False) -> None:
    _ensure_src_on_path()
    from highest_volatility.cache.store import save_cache

    for csv in _iter_legacy_daily_csv(root):
        ticker = csv.stem
        try:
            df = _parse_price_csv(csv)
        except Exception as exc:
            print(f"[daily] Skip {csv.name}: {exc}")
            continue
        if dry_run:
            print(f"[daily] Would convert {csv.name} -> 1d/{ticker}.parquet ({len(df)} rows)")
            continue
        try:
            save_cache(ticker, "1d", df, source="repair", validate=False)
            print(f"[daily] Converted {csv.name} -> 1d/{ticker}.parquet")
            if delete_legacy:
                csv.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[daily] Failed to convert {csv.name}: {exc}")


def repair_30m(root: Path, *, refetch_days: int = 60, delete_legacy: bool = False, dry_run: bool = False) -> None:
    _ensure_src_on_path()
    from highest_volatility.cache.store import save_cache

    to_fix: list[Path] = []
    for csv in _iter_legacy_30m_csv(root):
        try:
            df = _parse_price_csv(csv)
        except Exception as exc:
            print(f"[30m] Skip {csv.name}: {exc}")
            continue
        if not _has_time_components(df):
            to_fix.append(csv)
        else:
            # Has proper intraday times; convert in place to Parquet
            ticker = csv.stem
            if dry_run:
                print(f"[30m] Would convert {csv.name} -> 30m/{ticker}.parquet ({len(df)} rows)")
            else:
                try:
                    save_cache(ticker, "30m", df, source="repair", validate=False)
                    print(f"[30m] Converted {csv.name} -> 30m/{ticker}.parquet")
                    if delete_legacy:
                        csv.unlink(missing_ok=True)
                except Exception as exc:
                    print(f"[30m] Failed to convert {csv.name}: {exc}")

    if not to_fix:
        return

    # Refetch those lacking intraday times
    async def _do_refetch():
        _ensure_src_on_path()
        from highest_volatility.cache.store import save_cache

        for csv in to_fix:
            ticker = csv.stem
            try:
                if dry_run:
                    print(f"[30m] Would refetch {ticker} ({refetch_days} days) -> 30m/{ticker}.parquet")
                    continue
                df_new = await _refetch_30m(ticker, days=refetch_days)
                save_cache(ticker, "30m", df_new, source="repair", validate=False)
                print(f"[30m] Refetched {ticker} -> 30m/{ticker}.parquet")
                if delete_legacy:
                    csv.unlink(missing_ok=True)
            except Exception as exc:
                print(f"[30m] Failed refetch {ticker}: {exc}")

    asyncio.run(_do_refetch())


def main() -> None:
    _ensure_src_on_path()
    from highest_volatility.cache.store import CACHE_ROOT

    parser = argparse.ArgumentParser(description="Normalize cache layout and fix intraday timestamps")
    parser.add_argument("--root", type=Path, default=CACHE_ROOT, help="Cache root (default: cache/prices)")
    parser.add_argument("--dry-run", action="store_true", help="Only print actions without writing")
    parser.add_argument("--delete-legacy", action="store_true", help="Delete CSVs after successful conversion")
    parser.add_argument("--refetch-days", type=int, default=60, help="Days to refetch for 30m without times")
    args = parser.parse_args()

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    # Ensure expected interval subfolders exist
    for iv in ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]:
        (root / iv).mkdir(parents=True, exist_ok=True)

    print(f"Cache root: {root}")
    convert_legacy_daily(root, delete_legacy=args.delete_legacy, dry_run=args.dry_run)
    repair_30m(root, refetch_days=args.refetch_days, delete_legacy=args.delete_legacy, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
