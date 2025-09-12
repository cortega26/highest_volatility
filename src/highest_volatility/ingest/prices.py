"""Price history retrieval using yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import pandas as pd
import yfinance as yf


def download_price_history(
    tickers: List[str],
    lookback_days: int,
    *,
    interval: str = "1d",
    prepost: bool = False,
) -> pd.DataFrame:
    """Download price history for ``tickers``.

    Parameters
    ----------
    tickers:
        List of ticker symbols compatible with Yahoo Finance.
    lookback_days:
        Number of calendar days of history to request.
    interval:
        Data interval supported by Yahoo Finance (e.g. ``1d``, ``60m``, ``15m``).
    prepost:
        Include pre/post market data for intraday intervals.

    Returns
    -------
    pandas.DataFrame
        Raw DataFrame as returned by :func:`yfinance.download`.
    """

    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days * 2)
    ticker_str = " ".join(tickers)
    data = yf.download(
        ticker_str,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=False,
        prepost=prepost,
    )
    return data.dropna(how="all")
