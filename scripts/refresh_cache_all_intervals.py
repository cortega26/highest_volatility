#!/usr/bin/env python3
"""Refresh cached prices across multiple intervals and record misses."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from highest_volatility.app.cli import INTERVAL_CHOICES
from highest_volatility.cache.store import load_cached
from highest_volatility.ingest.prices import download_price_history
from highest_volatility.ingest.tickers import normalize_ticker
from highest_volatility.storage.ticker_cache import load_cached_fortune
from highest_volatility.universe import build_universe


@dataclass(frozen=True)
class FailureRecord:
    timestamp: str
    interval: str
    ticker: str
    reason: str


def _log(message: str) -> None:
    print(message, flush=True)


def _resolve_intervals(raw: str | None) -> list[str]:
    if not raw:
        return list(INTERVAL_CHOICES)
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(part)
    return unique


def _interval_is_intraday(interval: str) -> bool:
    return interval.lower().endswith(("m", "h"))


def _load_tickers_from_file(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").splitlines()
    tokens: list[str] = []
    for line in raw:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for part in stripped.replace(",", " ").split():
            tokens.append(normalize_ticker(part))
    return list(dict.fromkeys(tokens))


def _collect_missing(
    tickers: Sequence[str],
    interval: str,
    *,
    load_cached_fn=load_cached,
) -> list[FailureRecord]:
    failures: list[FailureRecord] = []
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for ticker in tickers:
        try:
            cached_df, manifest = load_cached_fn(ticker, interval)
        except Exception as exc:
            failures.append(
                FailureRecord(
                    timestamp=timestamp,
                    interval=interval,
                    ticker=ticker,
                    reason=f"load_error:{exc.__class__.__name__}",
                )
            )
            continue
        if cached_df is None or cached_df.empty or manifest is None:
            failures.append(
                FailureRecord(
                    timestamp=timestamp,
                    interval=interval,
                    ticker=ticker,
                    reason="missing_cache",
                )
            )
    return failures


def refresh_intervals(
    tickers: Sequence[str],
    intervals: Sequence[str],
    *,
    lookback_days_daily: int,
    lookback_days_intraday: int,
    prepost: bool,
    force_refresh: bool,
    max_workers: int,
    chunk_sleep: float,
    max_retries: int,
    verbose: bool = True,
    download_fn=download_price_history,
    load_cached_fn=load_cached,
) -> list[FailureRecord]:
    all_failures: list[FailureRecord] = []
    for interval in intervals:
        lookback_days = (
            lookback_days_intraday if _interval_is_intraday(interval) else lookback_days_daily
        )
        if verbose:
            _log(
                f"[{interval}] Refreshing {len(tickers)} tickers "
                f"(lookback_days={lookback_days}, workers={max_workers})..."
            )
        download_fn(
            list(tickers),
            lookback_days,
            interval=interval,
            prepost=prepost,
            use_cache=True,
            force_refresh=force_refresh,
            max_workers=max_workers,
            chunk_sleep=chunk_sleep,
            max_retries=max_retries,
            matrix_mode="batch",
        )
        interval_failures = _collect_missing(tickers, interval, load_cached_fn=load_cached_fn)
        if verbose:
            _log(f"[{interval}] Missing after refresh: {len(interval_failures)}")
        all_failures.extend(interval_failures)
    return all_failures


def _write_failures(path: Path, failures: Iterable[FailureRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(failures)
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(["timestamp", "interval", "ticker", "reason"])
        for record in rows:
            writer.writerow(
                [record.timestamp, record.interval, record.ticker, record.reason]
            )


def _load_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        tickers = list(dict.fromkeys(normalize_ticker(ticker) for ticker in args.tickers))
        _log(f"Using {len(tickers)} tickers from --tickers.")
        return tickers
    if args.tickers_file:
        tickers = _load_tickers_from_file(Path(args.tickers_file))
        _log(f"Using {len(tickers)} tickers from {args.tickers_file}.")
        return tickers
    cached = load_cached_fortune(max_age_days=args.ticker_cache_days, min_rows=min(100, args.top_n))
    if cached is None:
        _log(
            "No cached Fortune list found; scraping via Selenium (Chromium). "
            "Pass --tickers or --tickers-file to skip scraping."
        )
    else:
        _log(f"Using cached Fortune list ({len(cached)} rows).")
    tickers, _fortune = build_universe(
        args.top_n,
        validate=args.validate_universe,
        ticker_cache_days=args.ticker_cache_days,
    )
    _log(f"Universe tickers selected: {len(tickers)}")
    return list(tickers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh cached prices across intervals.")
    parser.add_argument(
        "--intervals",
        default=",".join(INTERVAL_CHOICES),
        help="Comma-separated intervals to refresh.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=500,
        help="Universe size when tickers are not supplied.",
    )
    parser.add_argument(
        "--ticker-cache-days",
        type=int,
        default=30,
        help="Use cached Fortune list if not older than this many days.",
    )
    parser.add_argument(
        "--validate-universe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Validate tickers via Selenium when building the universe.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Optional explicit tickers to refresh.",
    )
    parser.add_argument(
        "--tickers-file",
        help="Optional file containing tickers (comma or newline separated).",
    )
    parser.add_argument(
        "--lookback-days-daily",
        type=int,
        default=365,
        help="Lookback window for daily/weekly intervals.",
    )
    parser.add_argument(
        "--lookback-days-intraday",
        type=int,
        default=30,
        help="Lookback window for intraday intervals.",
    )
    parser.add_argument(
        "--prepost",
        action="store_true",
        help="Include pre/post-market data for intraday intervals.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force-refresh all tickers instead of incremental cache updates.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max parallel workers for each interval refresh.",
    )
    parser.add_argument(
        "--chunk-sleep",
        type=float,
        default=0.5,
        help="Sleep between chunk batches to avoid rate limiting.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts per download.",
    )
    parser.add_argument(
        "--failures-csv",
        default="cache/refresh_failures.csv",
        help="CSV path to append failed refresh entries.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    intervals = _resolve_intervals(args.intervals)
    _log(f"Intervals: {', '.join(intervals)}")
    tickers = _load_tickers(args)
    if not tickers:
        print("No tickers provided or discovered; nothing to refresh.")
        return 0

    failures = refresh_intervals(
        tickers,
        intervals,
        lookback_days_daily=args.lookback_days_daily,
        lookback_days_intraday=args.lookback_days_intraday,
        prepost=args.prepost,
        force_refresh=args.force_refresh,
        max_workers=args.max_workers,
        chunk_sleep=args.chunk_sleep,
        max_retries=args.max_retries,
    )
    _write_failures(Path(args.failures_csv), failures)
    if failures:
        print(f"Refresh completed with {len(failures)} missing entries.")
        return 2
    print("Refresh completed with no missing entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
