#!/usr/bin/env python3
"""Backfill cached prices for multiple intervals with proper window limits.

Uses the async HTTP datasource and the existing AsyncPriceFetcher which
automatically respects interval backfill windows via config.interval_policy.

Examples:
  python scripts/backfill_all_intervals.py --tickers AAPL MSFT NVDA
  python scripts/backfill_all_intervals.py --tickers-file tickers.txt --intervals 1m,5m,15m,30m,1h,1d
  python scripts/backfill_all_intervals.py --concurrency 16 --throttle 0.1
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List


def _ensure_src_on_path() -> Path:
    """Make sure the repository ``src`` directory is importable."""

    here = Path(__file__).resolve()
    root = here.parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def _read_tickers_from_file(p: Path) -> List[str]:
    lines = [ln.strip() for ln in p.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _discover_tickers() -> List[str]:
    """Discover tickers from existing cache CSVs/Parquet as a fallback."""

    _ensure_src_on_path()
    from src.cache.store import CACHE_ROOT  # Local import to avoid lint errors

    root = CACHE_ROOT
    out: set[str] = set()
    # Legacy CSVs at root
    for p in (root).glob("*.csv"):
        out.add(p.stem)
    # Existing parquet caches under any interval
    if root.exists():
        for sub in root.iterdir():
            if sub.is_dir():
                for pq in sub.glob("*.parquet"):
                    out.add(pq.stem)
    return sorted(out)


async def _backfill_interval(tickers: List[str], interval: str, *, concurrency: int, throttle: float) -> None:
    _ensure_src_on_path()
    from src.datasource.yahoo_http_async import YahooHTTPAsyncDataSource
    from src.ingest.async_fetch_prices import AsyncPriceFetcher
    from src.ingest.fetch_async import fetch_many_async

    ds = YahooHTTPAsyncDataSource()
    fetcher = AsyncPriceFetcher(ds, source_name="yahoo-http", throttle=throttle)
    await fetch_many_async(fetcher, tickers, interval, max_concurrency=concurrency)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill cached prices across intervals")
    parser.add_argument("--tickers", nargs="*", help="Ticker symbols (space-separated)")
    parser.add_argument("--tickers-file", type=Path, help="Optional file with one ticker per line")
    parser.add_argument(
        "--intervals",
        default="1m,5m,15m,30m,1h,1d",
        help="Comma-separated intervals to backfill (default: 1m..1d)",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent tickers")
    parser.add_argument("--throttle", type=float, default=0.2, help="Per-request throttle seconds")
    args = parser.parse_args()

    tickers: List[str] = []
    if args.tickers_file:
        tickers.extend(_read_tickers_from_file(args.tickers_file))
    if args.tickers:
        tickers.extend(args.tickers)
    if not tickers:
        tickers = _discover_tickers()
    if not tickers:
        raise SystemExit("No tickers provided or discovered from cache.")
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]

    print(f"Tickers: {len(tickers)}; Intervals: {', '.join(intervals)}")
    for iv in intervals:
        print(f"Backfilling interval: {iv}…")
        asyncio.run(_backfill_interval(tickers, iv, concurrency=args.concurrency, throttle=args.throttle))
    print("Done.")


if __name__ == "__main__":
    main()

