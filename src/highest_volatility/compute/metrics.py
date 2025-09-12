from __future__ import annotations

"""Pure functions for analysing price volatility.

The helpers here operate on DataFrames of adjusted close prices where the
index contains dates and columns are ticker symbols.  All functions return
*tidy* DataFrames so that downstream code can work with the results in a
consistent manner.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

TRADING_DAYS_PER_YEAR = 252
TRADING_MINUTES_PER_DAY = 390


def periods_per_year(interval: str) -> float:
    """Return the number of observation periods in a year for a given interval."""

    if interval.endswith("m"):
        minutes = int(interval[:-1])
        return TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY / minutes
    return TRADING_DAYS_PER_YEAR


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

    returns = prices.pct_change(fill_method=None).iloc[1:]
    return (
        returns.stack()
        .rename("daily_return")
        .rename_axis(index=["date", "ticker"])
        .reset_index()
    )


def annualized_volatility(
    prices: pd.DataFrame,
    *,
    min_periods: int | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Calculate annualized volatility for each ticker.

    Parameters
    ----------
    prices:
        Adjusted close prices with dates as index and tickers as columns.
    min_periods:
        Optional minimum number of observations required for a ticker to be
        included in the result.
    interval:
        Interval used when fetching the data.  Determines the annualisation
        factor.

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
    if min_periods is not None:
        valid = log_returns.count() >= min_periods
        std_dev = std_dev[valid]
    vols = std_dev * np.sqrt(periods_per_year(interval))
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

    returns = prices.pct_change(fill_method=None).dropna()
    excess = returns - risk_free / TRADING_DAYS_PER_YEAR
    mean = excess.mean()
    std = returns.std()
    sharpe = mean / std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sharpe.rename("sharpe_ratio").rename_axis("ticker").reset_index()


def additional_volatility_measures(
    raw: pd.DataFrame,
    tickers: List[str],
    *,
    min_periods: int = 2,
    interval: str = "1d",
    ewma_lambda: float = 0.94,
) -> pd.DataFrame:
    """Compute a selection of volatility estimators for each ticker.

    Parameters
    ----------
    raw:
        DataFrame as returned by :func:`yfinance.download` with OHLC data.
    tickers:
        Tickers for which to compute the metrics.
    min_periods:
        Minimum number of observations required to compute a metric.
    interval:
        Interval used when fetching the data.  Determines the annualisation
        factor.
    ewma_lambda:
        Smoothing factor for the EWMA estimator.

    Returns
    -------
    DataFrame
        Columns ``ticker`` and any of ``parkinson_vol``, ``gk_vol``,
        ``rs_vol``, ``yz_vol``, ``ewma_vol`` and ``mad_vol`` depending on the
        available data.
    """

    per_year = periods_per_year(interval)
    ln2 = np.log(2.0)
    results: List[Dict[str, float]] = []

    def _get_series(field: str, ticker: str) -> pd.Series | None:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                return raw[field][ticker].dropna()
            return raw[field].dropna()
        except Exception:
            return None

    for t in tickers:
        rec: Dict[str, float] = {"ticker": t}

        s_close = _get_series("Adj Close", t)
        if s_close is None:
            s_close = _get_series("Close", t)
        s_open = _get_series("Open", t)
        s_high = _get_series("High", t)
        s_low = _get_series("Low", t)

        frames = {
            k: v
            for k, v in {
                "close": s_close,
                "open": s_open,
                "high": s_high,
                "low": s_low,
            }.items()
            if v is not None
        }
        if not frames or "close" not in frames:
            continue

        df = pd.DataFrame(frames).dropna()
        if df.shape[0] < min_periods:
            continue

        r_cc = np.log(df["close"] / df["close"].shift(1)).dropna()

        if {"high", "low"}.issubset(df.columns):
            hl = np.log(df["high"] / df["low"]) ** 2
            var = hl.mean() / (4.0 * ln2)
            if np.isfinite(var) and var >= 0:
                rec["parkinson_vol"] = float(np.sqrt(var * per_year))

        if {"open", "high", "low", "close"}.issubset(df.columns):
            log_hl = np.log(df["high"] / df["low"]) ** 2
            log_co = np.log(df["close"] / df["open"]) ** 2
            gk_var = 0.5 * log_hl.mean() - (2.0 * ln2 - 1.0) * log_co.mean()
            if np.isfinite(gk_var) and gk_var >= 0:
                rec["gk_vol"] = float(np.sqrt(gk_var * per_year))

            term_rs = (
                np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) +
                np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"])
            )
            rs_var = term_rs.mean()
            if np.isfinite(rs_var) and rs_var >= 0:
                rec["rs_vol"] = float(np.sqrt(rs_var * per_year))

            prev_close = df["close"].shift(1)
            r_o = np.log(df["open"] / prev_close).dropna()
            r_c = np.log(df["close"] / df["open"]).dropna()
            df_rs = df.loc[r_c.index]
            term_rs_d = (
                np.log(df_rs["high"] / df_rs["close"]) * np.log(df_rs["high"] / df_rs["open"]) +
                np.log(df_rs["low"] / df_rs["close"]) * np.log(df_rs["low"] / df_rs["open"])
            )
            sigma_o2 = r_o.var(ddof=1)
            sigma_c2 = r_c.var(ddof=1)
            sigma_rs = term_rs_d.mean()
            if (
                np.isfinite(sigma_o2)
                and np.isfinite(sigma_c2)
                and np.isfinite(sigma_rs)
            ):
                k = 0.34
                yz_var = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs
                if yz_var >= 0:
                    rec["yz_vol"] = float(np.sqrt(yz_var * per_year))

        if r_cc.shape[0] >= min_periods:
            var = r_cc.var(ddof=1) if r_cc.shape[0] >= 2 else float(r_cc.iloc[-1] ** 2)
            for x in r_cc.iloc[-min_periods:]:
                var = ewma_lambda * var + (1.0 - ewma_lambda) * (x * x)
            if var >= 0:
                rec["ewma_vol"] = float(np.sqrt(var * per_year))
            mad = np.median(np.abs(r_cc - np.median(r_cc)))
            rec["mad_vol"] = float(1.4826 * mad * np.sqrt(per_year))

        results.append(rec)

    out = pd.DataFrame(results)
    if out.empty:
        return out
    cols = [
        "ticker",
        "parkinson_vol",
        "gk_vol",
        "rs_vol",
        "yz_vol",
        "ewma_vol",
        "mad_vol",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols]


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "TRADING_MINUTES_PER_DAY",
    "daily_returns",
    "annualized_volatility",
    "periods_per_year",
    "additional_volatility_measures",
    "max_drawdown",
    "rolling_volatility",
    "sharpe_ratio",
]
