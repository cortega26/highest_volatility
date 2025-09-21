"""Utilities for loading cached prices."""

from __future__ import annotations

import pandas as pd

from highest_volatility.cache.store import load_cached


def load_prices_from_cache(ticker: str, interval: str = "1d") -> pd.DataFrame:
    """Return cached prices for ``ticker``.

    Raises
    ------
    FileNotFoundError
        If the cache is missing.
    """

    df, _ = load_cached(ticker, interval)
    if df is None:
        raise FileNotFoundError(f"No cache for {ticker} {interval}")
    return df
