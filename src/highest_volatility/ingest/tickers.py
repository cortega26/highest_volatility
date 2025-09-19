"""Ticker retrieval utilities for the Highest Volatility package."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from highest_volatility.sources.selenium_universe import fetch_us500_fortune_pairs


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


def _fetch_with_selenium(source_url: str, top_n: int) -> pd.DataFrame:
    """Fetch Fortune companies via the robust Selenium grid harvester.

    Uses the same implementation as the universe builder to ensure consistency.
    """

    # Fetch extra to compensate for entries without tickers
    pairs = fetch_us500_fortune_pairs(top_n * 2)
    seen: set[str] = set()
    recs: list[dict] = []
    rank = 1
    for name, ticker in pairs:
        t = normalize_ticker(ticker)
        if not t or t in seen:
            continue
        recs.append({"rank": rank, "company": name, "ticker": t})
        seen.add(t)
        rank += 1
        if len(recs) >= top_n:
            break
    df = pd.DataFrame(recs)
    return df


def fetch_fortune_tickers(
    source_url: str = DEFAULT_SOURCE_URL,
    *,
    top_n: int = 100,
) -> pd.DataFrame:
    """Fetch the Fortune company list and return the first *top_n* rows.

    The list is retrieved exclusively via Selenium Stealth.  If scraping fails
    a small built-in fallback list is returned instead.
    """

    try:
        table = _fetch_with_selenium(source_url, top_n)
        if not table.empty:
            return table.head(top_n)
    except Exception:
        pass

    fallback_df = pd.DataFrame([f.__dict__ for f in FALLBACK_TICKERS])
    return fallback_df.head(top_n)

