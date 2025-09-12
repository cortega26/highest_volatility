"""Price history retrieval using yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import pandas as pd
import yfinance as yf


def download_price_history(tickers: List[str], lookback_days: int) -> pd.DataFrame:
    """Download adjusted close prices for ``tickers``.

    Parameters
    ----------
    tickers:
        List of ticker symbols compatible with Yahoo Finance.
    lookback_days:
        Number of calendar days of history to request.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date with one column per ticker.
    """

    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days * 2)
    ticker_str = " ".join(tickers)
    data = yf.download(ticker_str, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        data = data.to_frame(name="Close")
    return data.dropna(how="all")
