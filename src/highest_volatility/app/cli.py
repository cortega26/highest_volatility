"""Command line interface for the Highest Volatility package."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from highest_volatility.app.sanitization import sanitize_close
from highest_volatility.compute.metrics import (
    METRIC_REGISTRY,
    load_plugins,
    metric_display_name,
)
from highest_volatility.ingest.prices import download_price_history
from highest_volatility.storage.csv_store import save_csv
from highest_volatility.storage.sqlite_store import save_sqlite
from highest_volatility.universe import build_universe


DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_TOP_N = 100
DEFAULT_PRINT_TOP = 5
DEFAULT_MIN_DAYS = 126
INTERVAL_CHOICES = ["1d", "1h", "30m", "15m", "5m", "1m"]
# Load any third-party metric plugins before building choices
load_plugins()
METRIC_CHOICES = sorted(METRIC_REGISTRY.keys())

_METRIC_HELP_EXAMPLES = ["cc_vol", "parkinson_vol", "max_drawdown"]


def _format_metric_examples() -> str:
    examples = [
        f"{key} ({metric_display_name(key)})"
        for key in _METRIC_HELP_EXAMPLES
        if key in METRIC_REGISTRY
    ]
    if examples:
        return ", ".join(examples)
    return ", ".join(METRIC_CHOICES[:3])


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Find most volatile Fortune 100 stocks",
        allow_abbrev=False,
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
        help=f"Metric to rank by (e.g. {_format_metric_examples()})",
    )
    parser.add_argument(
        "--prepost",
        action="store_true",
        help="Include pre/post market data for intraday intervals",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Ticker symbols to process",
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
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    return build_parser().parse_args(argv)


@dataclass(frozen=True)
class BuildUniverseResult:
    """Container describing the outcome of the universe build step."""

    tickers: list[str]
    fortune: pd.DataFrame
    duration: float


@dataclass(frozen=True)
class DownloadPricesResult:
    """Container describing the outcome of the price download step."""

    prices: pd.DataFrame
    close: pd.DataFrame
    tickers: list[str]
    dropped_short: list[str]
    dropped_duplicate: list[str]
    duration: float


@dataclass(frozen=True)
class ComputeMetricsResult:
    """Container describing the computed metrics output."""

    result: pd.DataFrame
    duration: float


@dataclass(frozen=True)
class RenderOutputResult:
    """Container describing the rendering/exporting step."""

    duration: float


def _extract_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Extract the close or adjusted close slice from raw prices."""

    if prices.columns.empty:
        return pd.DataFrame(index=prices.index)

    if isinstance(prices.columns, pd.MultiIndex):
        if "Adj Close" in prices.columns.get_level_values(0):
            return prices["Adj Close"]
        return prices["Close"]
    return prices["Adj Close"] if "Adj Close" in prices.columns else prices["Close"]


def _build_universe_step(args: argparse.Namespace) -> BuildUniverseResult:
    """Retrieve the Fortune universe and measure execution time."""

    start = time.perf_counter()
    tickers, fortune = build_universe(
        args.top_n,
        validate=args.validate_universe,
        use_ticker_cache=not args.refresh_tickers,
        ticker_cache_days=args.tickers_cache_days,
    )
    duration = time.perf_counter() - start
    return BuildUniverseResult(tickers=tickers, fortune=fortune, duration=duration)


def _download_prices_step(
    args: argparse.Namespace, tickers: list[str]
) -> DownloadPricesResult:
    """Download and sanitize price history for the provided tickers."""

    start = time.perf_counter()
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
    duration = time.perf_counter() - start

    if prices.empty:
        empty_close = pd.DataFrame(index=prices.index)
        return DownloadPricesResult(
            prices=prices,
            close=empty_close,
            tickers=[],
            dropped_short=[],
            dropped_duplicate=[],
            duration=duration,
        )

    close = _extract_close(prices)
    close, dropped_short, dropped_dupe = sanitize_close(close, args.min_days)

    # Align raw prices to sanitized tickers for downstream metrics
    keep_tickers = list(close.columns)
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices.loc[:, prices.columns.get_level_values(1).isin(keep_tickers)]
    else:
        prices = prices.loc[:, keep_tickers]

    sanitized_tickers = [ticker for ticker in tickers if ticker in keep_tickers]
    return DownloadPricesResult(
        prices=prices,
        close=close,
        tickers=sanitized_tickers,
        dropped_short=dropped_short,
        dropped_duplicate=dropped_dupe,
        duration=duration,
    )


def _compute_metrics_step(
    args: argparse.Namespace,
    prices_result: DownloadPricesResult,
    fortune: pd.DataFrame,
) -> ComputeMetricsResult:
    """Compute the ranking metrics for the sanitized universe."""

    start = time.perf_counter()
    metric_func = METRIC_REGISTRY[args.metric]
    if prices_result.tickers:
        metric_df = metric_func(
            prices_result.prices,
            tickers=prices_result.tickers,
            close=prices_result.close,
            min_periods=args.min_days,
            interval=args.interval,
        )
    else:
        metric_df = pd.DataFrame(columns=["ticker", args.metric])
    metric_df = metric_df.drop_duplicates(subset=["ticker"], keep="first")
    fortune = fortune.drop_duplicates(subset=["ticker"], keep="first")
    metrics = metric_df.set_index("ticker")
    metrics = metrics.dropna(subset=[args.metric]).sort_values(args.metric, ascending=False)
    result = metrics.join(fortune.set_index("ticker")["company"]).join(
        fortune.set_index("ticker")["rank"]
    )
    result = result[~result.index.duplicated(keep="first")]
    duration = time.perf_counter() - start
    return ComputeMetricsResult(result=result, duration=duration)


def _render_output_step(
    args: argparse.Namespace, compute_result: ComputeMetricsResult
) -> RenderOutputResult:
    """Render CLI output and optional exports, capturing execution time."""

    start = time.perf_counter()
    result = compute_result.result.copy()
    if "company" in result.columns and "rank" in result.columns:
        id_cols = ["rank", "company"]
        metric_cols = [col for col in result.columns if col not in id_cols]
        result = result[id_cols + metric_cols]

    print(result.head(args.print_top).to_string())

    if args.output_csv:
        save_csv(result.reset_index(), args.output_csv)
    if args.output_sqlite:
        save_sqlite(result.reset_index(), args.output_sqlite)

    duration = time.perf_counter() - start
    return RenderOutputResult(duration=duration)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the CLI."""

    args = parse_args(argv)
    timings: dict[str, float] = {}

    if args.tickers:
        print("[1/4] Using provided tickers…", flush=True)
        normalized_tickers = list(dict.fromkeys(ticker.upper() for ticker in args.tickers))
        fortune = pd.DataFrame(
            {
                "ticker": normalized_tickers,
                "company": [None] * len(normalized_tickers),
                "rank": [None] * len(normalized_tickers),
            }
        )
        universe_result = BuildUniverseResult(
            tickers=normalized_tickers,
            fortune=fortune,
            duration=0.0,
        )
        timings["build_universe"] = universe_result.duration
    else:
        print("[1/4] Building universe (Selenium)…", flush=True)
        universe_result = _build_universe_step(args)
        timings["build_universe"] = universe_result.duration
    print(f"      Universe size: {len(universe_result.tickers)} tickers", flush=True)
    if len(universe_result.tickers) < args.print_top:
        raise SystemExit(
            f"Universe too small: got {len(universe_result.tickers)} tickers, but --print-top={args.print_top}. "
            "Check pagination / fetch source."
        )

    print("[2/4] Downloading price history…", flush=True)
    prices_result = _download_prices_step(args, universe_result.tickers)
    timings["download_prices"] = prices_result.duration
    if prices_result.dropped_short or prices_result.dropped_duplicate:
        print(
            "      Sanitized price matrix: "
            f"dropped {len(prices_result.dropped_short)} short and "
            f"{len(prices_result.dropped_duplicate)} duplicate series",
            flush=True,
        )

    print("[3/4] Computing metrics…", flush=True)
    compute_result = _compute_metrics_step(args, prices_result, universe_result.fortune)
    timings["compute_metrics"] = compute_result.duration

    print(f"[4/4] Top {args.print_top} rows:")
    render_result = _render_output_step(args, compute_result)
    timings["print_top"] = render_result.duration

    if args.timings:
        total = sum(timings.values())

        def fmt(value: float) -> str:
            return f"{value:.2f}s"

        print(
            "\nTimings: "
            f"build_universe={fmt(timings.get('build_universe', 0.0))}, "
            f"download_prices={fmt(timings.get('download_prices', 0.0))}, "
            f"compute_metrics={fmt(timings.get('compute_metrics', 0.0))}, "
            f"print_top={fmt(timings.get('print_top', 0.0))}, "
            f"total={fmt(total)}"
        )
