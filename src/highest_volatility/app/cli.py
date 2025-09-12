"""Command line interface for the Highest Volatility package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from highest_volatility.compute.metrics import annualized_volatility
from highest_volatility.ingest.prices import download_price_history
from highest_volatility.ingest.tickers import fetch_fortune_tickers
from highest_volatility.storage.csv_store import save_csv
from highest_volatility.storage.sqlite_store import save_sqlite


DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_TOP_N = 100
DEFAULT_PRINT_TOP = 5
DEFAULT_MIN_DAYS = 126


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Find most volatile Fortune 100 stocks")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Number of days of price history to use")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="How many Fortune companies to analyse")
    parser.add_argument("--print-top", type=int, default=DEFAULT_PRINT_TOP, help="How many results to print")
    parser.add_argument("--min-days", type=int, default=DEFAULT_MIN_DAYS, help="Minimum data points required for a ticker")
    parser.add_argument("--output-csv", type=Path, help="Optional path to write the full results as CSV")
    parser.add_argument("--output-sqlite", type=Path, help="Optional path to write the full results to SQLite")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the CLI."""

    args = parse_args(argv)
    fortune = fetch_fortune_tickers(top_n=args.top_n)
    tickers = fortune["ticker"].tolist()
    prices = download_price_history(tickers, args.lookback_days)
    vols = annualized_volatility(prices, min_days=args.min_days).rename(
        columns={"annualized_volatility": "volatility"}
    )
    result = fortune.set_index("ticker").join(
        vols.set_index("ticker")
    ).dropna()
    result = result.sort_values("volatility", ascending=False)

    print(result.head(args.print_top).to_string())

    if args.output_csv:
        save_csv(result.reset_index(), args.output_csv)
    if args.output_sqlite:
        save_sqlite(result.reset_index(), args.output_sqlite)
