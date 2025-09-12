"""Volatility computation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_volatility(prices: pd.DataFrame, *, min_days: int = 126) -> pd.Series:
    """Calculate annualized volatility for each column in ``prices``.

    Parameters
    ----------
    prices:
        DataFrame of adjusted close prices, columns are tickers.
    min_days:
        Minimum number of price observations required for a ticker to be included.

    Returns
    -------
    pandas.Series
        Volatility values indexed by ticker symbol.
    """

    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    std_dev = log_returns.std(skipna=True)
    valid = log_returns.count() >= min_days
    vols = std_dev[valid] * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vols.sort_values(ascending=False)
