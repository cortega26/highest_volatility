"""Ticker retrieval utilities for the Highest Volatility package."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests


@dataclass
class FortuneTicker:
    """Representation of a single company in the Fortune list."""

    rank: int
    company: str
    ticker: str


DEFAULT_SOURCE_URL = "https://us500.com/fortune-500-companies"
FALLBACK_TICKERS = [
    FortuneTicker(1, "Apple", "AAPL"),
    FortuneTicker(2, "Microsoft", "MSFT"),
    FortuneTicker(3, "Amazon.com", "AMZN"),
    FortuneTicker(4, "Alphabet", "GOOGL"),
    FortuneTicker(5, "Meta Platforms", "META"),
]


def normalize_ticker(ticker: str) -> str:
    """Return a Yahoo Finance compatible ticker symbol."""

    t = ticker.strip().upper()
    return t.replace(".", "-")


def fetch_fortune_tickers(source_url: str = DEFAULT_SOURCE_URL, *, top_n: int = 100) -> pd.DataFrame:
    """Fetch the Fortune 500 table and return the first *top_n* rows with tickers.

    If the remote data cannot be retrieved, a small built-in list of well known
    tickers is returned instead.
    """

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(source_url, timeout=30, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        table = tables[0]
        table = table.rename(columns=str.lower)
        expected_cols = {"rank", "company", "ticker"}
        if expected_cols.issubset(set(table.columns)):
            table = table[list(expected_cols)].head(top_n)
            table["ticker"] = table["ticker"].astype(str).map(normalize_ticker)
            table = table[table["ticker"] != ""]
            return table.reset_index(drop=True)
    except Exception:
        pass

    fallback_df = pd.DataFrame([f.__dict__ for f in FALLBACK_TICKERS])
    return fallback_df.head(top_n)
