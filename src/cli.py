"""Command line interface for price ingestion."""

from __future__ import annotations

import argparse
import random
from datetime import date

import pandas as pd

import asyncio

from src.cache.store import load_cached
from src.config.interval_policy import full_backfill_start
from src.datasource.yahoo import YahooDataSource
from src.datasource.yahoo_async import YahooAsyncDataSource
from src.ingest.fetch_async import fetch_many_async
from src.ingest.async_fetch_prices import AsyncPriceFetcher


def _compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if a.index.min() != b.index.min():
        return False
    if a.index.max() != b.index.max():
        return False
    if len(a) != len(b):
        return False
    cols = [c for c in ["Adj Close", "Close"] if c in a.columns and c in b.columns]
    if cols:
        if not a[cols].tail(10).equals(b[cols].tail(10)):
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Price ingestion utility")
    parser.add_argument(
        "--interval",
        default="1d",
        choices=["1m", "5m", "10m", "15m", "30m", "60m", "1h", "1d", "1wk", "1mo"],
    )
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--integrity", choices=["none","sample","full"], default="none")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--throttle", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    datasource = YahooAsyncDataSource()
    source_name = "yahoo"

    fetcher = AsyncPriceFetcher(datasource, source_name=source_name, throttle=args.throttle)
    asyncio.run(
        fetch_many_async(
            fetcher,
            args.tickers,
            args.interval,
            force_refresh=args.force_refresh,
            max_concurrency=args.workers,
        )
    )

    if args.integrity != "none":
        to_check = list(args.tickers)
        if args.integrity == "sample" and len(to_check) > 5:
            to_check = random.sample(to_check, 5)
        mismatches = []
        for t in to_check:
            cached_df, _ = load_cached(t, args.interval)
            if cached_df is None:
                mismatches.append(t)
                continue
            # Use synchronous datasource for integrity checks to keep implementation simple
            sync_ds = YahooDataSource()
            fresh_df = sync_ds.get_prices(t, full_backfill_start(args.interval), date.today(), args.interval)
            if not _compare_frames(cached_df, fresh_df):
                mismatches.append(t)
        if mismatches:
            print("Integrity check failed for: " + ", ".join(sorted(mismatches)))
            return 1
        else:
            mode = "sample" if args.integrity == "sample" else "full"
            print(f"Integrity OK ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
