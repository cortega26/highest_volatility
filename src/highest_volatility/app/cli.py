"""Command line interface for the Highest Volatility package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from highest_volatility.compute.metrics import (
    additional_volatility_measures,
    annualized_volatility,
)
from highest_volatility.ingest.prices import download_price_history
from highest_volatility.ingest.tickers import fetch_fortune_tickers
from highest_volatility.storage.csv_store import save_csv
from highest_volatility.storage.sqlite_store import save_sqlite


DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_TOP_N = 100
DEFAULT_PRINT_TOP = 5
DEFAULT_MIN_DAYS = 126
INTERVAL_CHOICES = ["1d", "60m", "30m", "15m", "5m", "1m"]
METRIC_CHOICES = [
    "cc_vol",
    "parkinson_vol",
    "gk_vol",
    "rs_vol",
    "yz_vol",
    "ewma_vol",
    "mad_vol",
]


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
        help="Volatility metric to rank by",
    )
    parser.add_argument(
        "--prepost",
        action="store_true",
        help="Include pre/post market data for intraday intervals",
    )
    parser.add_argument(
        "--output-csv", type=Path, help="Optional path to write the full results as CSV"
    )
    parser.add_argument(
        "--output-sqlite", type=Path, help="Optional path to write the full results to SQLite"
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the CLI."""

    args = parse_args(argv)
    fortune = fetch_fortune_tickers(top_n=args.top_n)
    tickers = fortune["ticker"].tolist()
    prices = download_price_history(
        tickers, args.lookback_days, interval=args.interval, prepost=args.prepost
    )
    if isinstance(prices.columns, pd.MultiIndex):
        if "Adj Close" in prices.columns.get_level_values(0):
            close = prices["Adj Close"]
        else:
            close = prices["Close"]
    else:
        close = (
            prices["Adj Close"] if "Adj Close" in prices.columns else prices["Close"]
        )
    vols_cc = annualized_volatility(
        close, min_periods=args.min_days, interval=args.interval
    ).rename(columns={"annualized_volatility": "cc_vol"})
    extras = additional_volatility_measures(
        prices, tickers, min_periods=args.min_days, interval=args.interval
    )
    vols = vols_cc.merge(extras, on="ticker", how="left")
    result = fortune.set_index("ticker").join(vols.set_index("ticker"))
    result = result.dropna(subset=[args.metric])
    result = result.sort_values(args.metric, ascending=False)

    print(result.head(args.print_top).to_string())

    if args.output_csv:
        save_csv(result.reset_index(), args.output_csv)
    if args.output_sqlite:
        save_sqlite(result.reset_index(), args.output_sqlite)
