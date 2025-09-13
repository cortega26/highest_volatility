"""Command line interface for the Highest Volatility package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional
import time

import pandas as pd

from highest_volatility.compute.metrics import METRIC_REGISTRY, load_plugins
from highest_volatility.ingest.prices import download_price_history
from highest_volatility.universe import build_universe
from highest_volatility.storage.csv_store import save_csv
from highest_volatility.storage.sqlite_store import save_sqlite


DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_TOP_N = 100
DEFAULT_PRINT_TOP = 5
DEFAULT_MIN_DAYS = 126
INTERVAL_CHOICES = ["1d", "60m", "30m", "15m", "5m", "1m"]
# Load any third-party metric plugins before building choices
load_plugins()
METRIC_CHOICES = sorted(METRIC_REGISTRY.keys())


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Find most volatile Fortune 100 stocks"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Number of days of price history to use",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="How many Fortune companies to analyse",
    )
    parser.add_argument(
        "--print-top",
        type=int,
        default=DEFAULT_PRINT_TOP,
        help="How many results to print",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=DEFAULT_MIN_DAYS,
        help="Minimum data points required for a ticker",
    )
    parser.add_argument(
        "--interval",
        choices=INTERVAL_CHOICES,
        default="1d",
        help="Price interval (daily or intraday)",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default="cc_vol",
        help=(
            "Metric to rank by (e.g. cc_vol, sharpe_ratio, max_drawdown, "
            "var, sortino)"
        ),
    )
    parser.add_argument(
        "--prepost",
        action="store_true",
        help="Include pre/post market data for intraday intervals",
    )
    parser.add_argument(
        "--validate-universe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quickly pre-validate tickers via Yahoo (disable to speed up startup)",
    )
    parser.add_argument(
        "--tickers-cache-days",
        type=int,
        default=30,
        help="Use cached tickers if not older than this many days",
    )
    parser.add_argument(
        "--refresh-tickers",
        action="store_true",
        help="Ignore cached tickers and scrape anew",
    )
    parser.add_argument(
        "--output-csv", type=Path, help="Optional path to write the full results as CSV"
    )
    parser.add_argument(
        "--output-sqlite", type=Path, help="Optional path to write the full results to SQLite"
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use on-disk quote cache to speed up repeated runs",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="Print per-step timings to help validate caching effectiveness",
    )
    parser.add_argument(
        "--force-refresh-prices",
        action="store_true",
        help="Force-refresh prices from Yahoo (ignore cache for this run)",
    )
    parser.add_argument(
        "--price-fetch-workers",
        type=int,
        default=8,
        help="Max parallel workers for price fetching",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for price downloads",
    )
    parser.add_argument(
        "--async-fetch",
        action="store_true",
        help="Fetch prices asynchronously via HTTP API",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the CLI."""

    args = parse_args(argv)

    timings: dict[str, float] = {}

    print("[1/4] Building universe (Selenium)…", flush=True)
    t0 = time.perf_counter()
    tickers, fortune = build_universe(
        args.top_n,
        validate=args.validate_universe,
        use_ticker_cache=not args.refresh_tickers,
        ticker_cache_days=args.tickers_cache_days,
    )
    timings["build_universe"] = time.perf_counter() - t0
    print(f"      Universe size: {len(tickers)} tickers", flush=True)
    if len(tickers) < args.print_top:
        raise SystemExit(
            f"Universe too small: got {len(tickers)} tickers, but --print-top={args.print_top}. "
            "Check pagination / fetch source."
        )
    print("[2/4] Downloading price history…", flush=True)
    t0 = time.perf_counter()
    prices = download_price_history(
        tickers,
        args.lookback_days,
        interval=args.interval,
        prepost=args.prepost,
        use_cache=args.use_cache,
        force_refresh=args.force_refresh_prices,
        max_workers=args.price_fetch_workers,
        matrix_mode="async" if args.async_fetch else "batch",
        max_retries=args.max_retries,
    )
    timings["download_prices"] = time.perf_counter() - t0
    if isinstance(prices.columns, pd.MultiIndex):
        if "Adj Close" in prices.columns.get_level_values(0):
            close = prices["Adj Close"]
        else:
            close = prices["Close"]
    else:
        close = (
            prices["Adj Close"] if "Adj Close" in prices.columns else prices["Close"]
        )

    # Sanitize price matrix to prevent duplicate/degenerate series from skewing metrics
    def _sanitize_close(df: pd.DataFrame, min_days: int) -> tuple[pd.DataFrame, list[str], list[str]]:
        import hashlib
        dropped_short: list[str] = []
        dropped_dupe: list[str] = []
        # Drop duplicate-named columns to avoid ambiguous selections
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        # Drop columns with insufficient data (count non-NaN strictly per column)
        keep_cols: list[str] = []
        for c in df.columns:
            if int(df[c].notna().sum()) >= min_days:
                keep_cols.append(c)
            else:
                dropped_short.append(c)
        df2 = df[keep_cols].copy()
        # Drop exact-duplicate recent tails
        seen: dict[str, str] = {}
        tail_n = min(250, max(60, min_days))
        for c in list(df2.columns):
            s = df2[c].tail(tail_n).astype(float).ffill().bfill()
            if s.dropna().empty:
                df2 = df2.drop(columns=[c])
                continue
            sig = hashlib.sha1(s.to_numpy().tobytes()).hexdigest()
            if sig in seen:
                dropped_dupe.append(c)
                df2 = df2.drop(columns=[c])
            else:
                seen[sig] = c
        return df2, dropped_short, dropped_dupe

    close, dropped_short, dropped_dupe = _sanitize_close(close, args.min_days)
    if dropped_short or dropped_dupe:
        print(
            f"      Sanitized price matrix: dropped {len(dropped_short)} short and {len(dropped_dupe)} duplicate series",
            flush=True,
        )
    # Filter raw prices (OHLC) to the remaining tickers for downstream metrics
    if isinstance(prices.columns, pd.MultiIndex):
        # Keep only tickers present after sanitization
        keep = [c for c in close.columns]
        prices = prices.loc[:, prices.columns.get_level_values(1).isin(keep)]
        # Ensure expected order isn't required later
    tickers = [t for t in tickers if t in close.columns]
    print("[3/4] Computing metrics…", flush=True)
    t0 = time.perf_counter()
    metric_func = METRIC_REGISTRY[args.metric]
    metric_df = metric_func(
        prices,
        tickers=tickers,
        close=close,
        min_periods=args.min_days,
        interval=args.interval,
    )
    metric_df = metric_df.drop_duplicates(subset=["ticker"], keep="first")
    fortune = fortune.drop_duplicates(subset=["ticker"], keep="first")
    metrics = metric_df.set_index("ticker")
    metrics = metrics.dropna(subset=[args.metric]).sort_values(args.metric, ascending=False)
    result = metrics.join(fortune.set_index("ticker")["company"]).join(
        fortune.set_index("ticker")["rank"]
    )
    result = result[~result.index.duplicated(keep="first")]
    timings["compute_metrics"] = time.perf_counter() - t0

    print(f"[4/4] Top {args.print_top} rows:")
    t0 = time.perf_counter()
    # Reorder columns to show identifiers first
    if "company" in result.columns and "rank" in result.columns:
        id_cols = ["rank", "company"]
        metric_cols = [c for c in result.columns if c not in id_cols]
        result = result[id_cols + metric_cols]
    print(result.head(args.print_top).to_string())
    timings["print_top"] = time.perf_counter() - t0

    if args.timings:
        # Summarize timings
        def fmt(x: float) -> str:
            return f"{x:.2f}s"

        total = sum(timings.values())
        print(
            "\nTimings: "
            f"build_universe={fmt(timings.get('build_universe', 0.0))}, "
            f"download_prices={fmt(timings.get('download_prices', 0.0))}, "
            f"compute_metrics={fmt(timings.get('compute_metrics', 0.0))}, "
            f"print_top={fmt(timings.get('print_top', 0.0))}, "
            f"total={fmt(total)}"
        )

    if args.output_csv:
        save_csv(result.reset_index(), args.output_csv)
    if args.output_sqlite:
        save_sqlite(result.reset_index(), args.output_sqlite)
