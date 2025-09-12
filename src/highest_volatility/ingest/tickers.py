"""Ticker retrieval utilities for the Highest Volatility package."""

from __future__ import annotations

from dataclasses import dataclass

import json

import pandas as pd
import requests
from bs4 import BeautifulSoup


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
    """Fetch the Fortune company list and return the first *top_n* rows.

    The primary data source is ``us500.com`` which renders its table via
    JavaScript.  To avoid introducing a full browser dependency, we parse the
    ``__NEXT_DATA__`` JSON blob embedded in the page which contains the
    information for the first 50 companies.  If anything goes wrong a small
    built-in fallback list is returned instead.
    """

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(source_url, timeout=30, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        script = soup.find("script", id="__NEXT_DATA__")
        if script and script.string:
            data = json.loads(script.string)
            companies = data["props"]["pageProps"].get("initialResults", [])
            if companies:
                table = pd.DataFrame(companies)
                table = table.rename(columns=str.lower)
                expected_cols = {"rank", "company", "ticker"}
                table = table[list(expected_cols)].head(top_n)
                table["rank"] = table["rank"].astype(int)
                table["ticker"] = table["ticker"].astype(str).map(normalize_ticker)
                table = table[table["ticker"].str.fullmatch(r"[A-Z]+[A-Z0-9.-]*")]  # drop entries without a valid ticker
                return table.reset_index(drop=True)
    except Exception:
        pass

    fallback_df = pd.DataFrame([f.__dict__ for f in FALLBACK_TICKERS])
    return fallback_df.head(top_n)
