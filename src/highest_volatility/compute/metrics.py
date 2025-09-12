from __future__ import annotations

"""Pure functions for analysing price volatility.

The helpers here operate on DataFrames of adjusted close prices where the
index contains dates and columns are ticker symbols.  All functions return
*tidy* DataFrames so that downstream code can work with the results in a
consistent manner.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns.

    Parameters
    ----------
    prices:
        DataFrame of adjusted close prices indexed by date and with
        tickers as columns.

    Returns
    -------
    DataFrame
        Tidy DataFrame with columns ``date``, ``ticker`` and ``daily_return``.

    Examples
    --------
    >>> import pandas as pd
    >>> prices = pd.DataFrame({'A': [1, 2, 1.5]},
    ...                       index=pd.date_range('2020-01-01', periods=3))
    >>> expected = pd.DataFrame(
    ...     {
    ...         'date': pd.to_datetime(['2020-01-02', '2020-01-03']),
    ...         'ticker': ['A', 'A'],
    ...         'daily_return': [1.0, -0.25],
    ...     }
    ... )
    >>> pd.testing.assert_frame_equal(daily_returns(prices), expected)
    """

    returns = prices.pct_change().iloc[1:]
    return (
        returns.stack()
        .rename("daily_return")
        .rename_axis(index=["date", "ticker"])
        .reset_index()
    )


def annualized_volatility(
    prices: pd.DataFrame, *, min_days: int | None = None
) -> pd.DataFrame:
    """Calculate annualized volatility for each ticker.

    Parameters
    ----------
    prices:
        Adjusted close prices with dates as index and tickers as columns.
    min_days:
        Optional minimum number of observations required for a ticker to
        be included in the result.

    Returns
    -------
    DataFrame
        Columns ``ticker`` and ``annualized_volatility`` sorted in
        descending order.

    Examples
    --------
    >>> import pandas as pd
    >>> prices = pd.DataFrame({'A': [1, 2, 1.5]},
    ...                       index=pd.date_range('2020-01-01', periods=3))
    >>> annualized_volatility(prices).round(2)
      ticker  annualized_volatility
    0      A                  11.01
    """

    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    std_dev = log_returns.std(skipna=True)
    if min_days is not None:
        valid = log_returns.count() >= min_days
        std_dev = std_dev[valid]
    vols = std_dev * np.sqrt(TRADING_DAYS_PER_YEAR)
    return (
        vols.sort_values(ascending=False)
        .rename("annualized_volatility")
        .rename_axis("ticker")
        .reset_index()
    )


def max_drawdown(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute the maximum drawdown for each ticker.

    Parameters
    ----------
    prices:
        Adjusted close prices.

    Returns
    -------
    DataFrame
        Columns ``ticker`` and ``max_drawdown`` (as negative percentages).

    Examples
    --------
    >>> import pandas as pd
    >>> prices = pd.DataFrame({'A': [1, 2, 1.5]},
    ...                       index=pd.date_range('2020-01-01', periods=3))
    >>> max_drawdown(prices).round(2)
      ticker  max_drawdown
    0      A         -0.25
    """

    drawdown = prices / prices.cummax() - 1
    mdd = drawdown.min()
    return (
        mdd.rename("max_drawdown").rename_axis("ticker").reset_index()
    )


def rolling_volatility(
    prices: pd.DataFrame, *, windows: tuple[int, ...] = (30, 60, 90)
) -> pd.DataFrame:
    """Rolling annualized volatility for multiple windows.

    Parameters
    ----------
    prices:
        Adjusted close prices.
    windows:
        Window sizes in days for which to compute volatility.

    Returns
    -------
    DataFrame
        Tidy DataFrame with columns ``date``, ``ticker``, ``window`` and
        ``rolling_volatility``.  Rows containing insufficient data for a
        given window are omitted.

    Examples
    --------
    >>> import pandas as pd
    >>> prices = pd.DataFrame({'A': [1, 2, 1.5]},
    ...                       index=pd.date_range('2020-01-01', periods=3))
    >>> rolling_volatility(prices, windows=(2,)).round(2)
            date ticker  window  rolling_volatility
    0 2020-01-03      A       2               11.01
    """

    returns = np.log(prices / prices.shift(1))
    frames = []
    for window in windows:
        vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        df = (
            vol.stack()
            .rename("rolling_volatility")
            .rename_axis(index=["date", "ticker"])
            .reset_index()
        )
        df["window"] = window
        df = df[["date", "ticker", "window", "rolling_volatility"]]
        frames.append(df.dropna(subset=["rolling_volatility"]))
    return pd.concat(frames, ignore_index=True)


def sharpe_ratio(prices: pd.DataFrame, *, risk_free: float = 0.0) -> pd.DataFrame:
    """Compute the annualized Sharpe ratio for each ticker.

    Parameters
    ----------
    prices:
        Adjusted close prices.
    risk_free:
        Annual risk-free rate expressed as a decimal (e.g. ``0.02`` for
        2%).

    Returns
    -------
    DataFrame
        Columns ``ticker`` and ``sharpe_ratio``.

    Examples
    --------
    >>> import pandas as pd
    >>> prices = pd.DataFrame({'A': [1, 2, 1.5]},
    ...                       index=pd.date_range('2020-01-01', periods=3))
    >>> sharpe_ratio(prices).round(2).to_dict()
    {'ticker': {0: 'A'}, 'sharpe_ratio': {0: 6.73}}
    """

    returns = prices.pct_change().dropna()
    excess = returns - risk_free / TRADING_DAYS_PER_YEAR
    mean = excess.mean()
    std = returns.std()
    sharpe = mean / std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sharpe.rename("sharpe_ratio").rename_axis("ticker").reset_index()


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "daily_returns",
    "annualized_volatility",
    "max_drawdown",
    "rolling_volatility",
    "sharpe_ratio",
]
