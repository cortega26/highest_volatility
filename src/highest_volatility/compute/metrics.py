"""Pure functions for analysing price volatility.

The helpers here operate on DataFrames of adjusted close prices where the
index contains dates and columns are ticker symbols.  All functions return
*tidy* DataFrames so that downstream code can work with the results in a
consistent manner.
"""

import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Union, cast

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
TRADING_MINUTES_PER_DAY = 390


_INTERVAL_PATTERN = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>[a-zA-Z]+)$")


def periods_per_year(interval: str) -> float:
    """Return the annual observation count for a Yahoo-style interval string.

    Parameters
    ----------
    interval:
        Interval suffix such as ``"1m"`` (one minute) or ``"1wk"`` (one week).

    Returns
    -------
    float
        Number of observations per year implied by ``interval``.

    Raises
    ------
    ValueError
        If ``interval`` cannot be parsed or encodes a non-positive duration.
    """

    normalized = interval.strip().lower()
    match = _INTERVAL_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(f"Unrecognised interval: {interval!r}")

    value = float(match.group("value"))
    if value <= 0:
        raise ValueError("Interval magnitude must be positive")

    unit = match.group("unit")
    if unit == "m":
        minutes = value
        return TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY / minutes
    if unit == "h":
        minutes = value * 60.0
        return TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY / minutes
    if unit == "d":
        return TRADING_DAYS_PER_YEAR / value
    if unit == "wk":
        return 52.0 / value
    if unit == "mo":
        return 12.0 / value

    raise ValueError(f"Unsupported interval unit: {interval!r}")


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


def value_at_risk(prices: pd.DataFrame, *, confidence: float = 0.05) -> pd.DataFrame:
    """Compute the value at risk (VaR) for each ticker.

    Parameters
    ----------
    prices:
        Adjusted close prices.
    confidence:
        Left-tail probability used to compute the quantile.  For example,
        ``0.05`` corresponds to the 5% VaR.

    Returns
    -------
    DataFrame
        Columns ``ticker`` and ``var`` with VaR expressed as a fraction of
        the price (e.g. ``-0.03`` for a 3% loss).

    Examples
    --------
    >>> import pandas as pd
    >>> prices = pd.DataFrame({'A': [100, 110, 105, 102, 100]},
    ...                       index=pd.date_range('2020-01-01', periods=5))
    >>> value_at_risk(prices).round(2).to_dict()
    {'ticker': {0: 'A'}, 'var': {0: -0.04}}
    """

    returns = prices.pct_change(fill_method=None).dropna()
    var = returns.quantile(confidence)
    return var.rename("var").rename_axis("ticker").reset_index()


def sortino_ratio(prices: pd.DataFrame, *, risk_free: float = 0.0) -> pd.DataFrame:
    """Compute the annualized Sortino ratio for each ticker.

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
        Columns ``ticker`` and ``sortino``.

    Examples
    --------
    >>> import pandas as pd
    >>> prices = pd.DataFrame({'A': [100, 110, 105, 102, 100]},
    ...                       index=pd.date_range('2020-01-01', periods=5))
    >>> sortino_ratio(prices).round(2).to_dict()
    {'ticker': {0: 'A'}, 'sortino': {0: 1.93}}
    """

    returns = prices.pct_change(fill_method=None).dropna()
    excess = returns - risk_free / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    downside_std = downside.std()
    mean_excess = excess.mean()
    sortino = mean_excess / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sortino.rename("sortino").rename_axis("ticker").reset_index()


def additional_volatility_measures(
    raw: pd.DataFrame,
    tickers: List[str],
    *,
    min_periods: int = 2,
    interval: str = "1d",
    ewma_lambda: float = 0.94,
) -> pd.DataFrame:
    """Compute a selection of volatility estimators for each ticker.

    The implementation normalises the input frame once and materialises
    per-field ``float64`` arrays together with validity masks.  Subsequent
    ticker lookups are then simple column selections instead of repeatedly
    coercing ``Series`` objects.  This reduces the dominant cost from
    roughly :math:`O(F * T * N)` conversions (fields × tickers × periods)
    down to :math:`O(F * N)` followed by :math:`O(T * N)` arithmetic.
    """

    per_year = periods_per_year(interval)
    ln2 = np.log(2.0)
    results: List[Dict[str, Union[float, str]]] = []

    if tickers == []:
        return pd.DataFrame(
            columns=[
                "ticker",
                "parkinson_vol",
                "gk_vol",
                "rs_vol",
                "yz_vol",
                "ewma_vol",
                "mad_vol",
            ]
        )

    normalized = raw
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = set(raw.columns.get_level_values(0))
        lv1 = set(raw.columns.get_level_values(1))
        has_level0_fields = {"Open", "Close", "Adj Close", "High", "Low"}.intersection(lv0)
        if not has_level0_fields and {"Open", "Close", "Adj Close"}.intersection(lv1):
            normalized = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)
        else:
            normalized = raw.sort_index(axis=1)

    FieldArrays = tuple[np.ndarray, np.ndarray, Dict[str, int]]

    def _prepare_field(field_name: str) -> FieldArrays | None:
        if isinstance(normalized.columns, pd.MultiIndex):
            if field_name not in normalized.columns.get_level_values(0):
                return None
            frame = normalized[field_name]
        elif field_name in normalized.columns:
            frame = normalized[[field_name]]
        else:
            return None

        if isinstance(frame, pd.Series):
            series_frame = frame.to_frame()
        else:
            series_frame = frame

        numeric = series_frame.apply(pd.to_numeric, errors="coerce")
        values_2d = numeric.to_numpy(dtype=np.float64, copy=False)
        mask_2d = np.isfinite(values_2d)
        column_index_map: Dict[str, int] = {}
        for idx, col in enumerate(series_frame.columns):
            key = col if isinstance(col, str) else str(col)
            column_index_map[key] = idx
        return values_2d, mask_2d, column_index_map

    field_arrays: Dict[str, FieldArrays | None] = {
        "adj_close": _prepare_field("Adj Close"),
        "close": _prepare_field("Close"),
        "open": _prepare_field("Open"),
        "high": _prepare_field("High"),
        "low": _prepare_field("Low"),
    }

    def _column_data(field: str, ticker: str) -> tuple[np.ndarray, np.ndarray] | None:
        data = field_arrays.get(field)
        if data is None:
            return None
        values_2d, mask_2d, column_index_map = data
        column_idx = column_index_map.get(ticker)
        if column_idx is None:
            if len(column_index_map) == 1:
                column_idx = next(iter(column_index_map.values()))
            else:
                return None
        return values_2d[:, column_idx], mask_2d[:, column_idx]

    for ticker in tickers:
        close_data = _column_data("adj_close", ticker) or _column_data("close", ticker)
        if close_data is None:
            continue
        close_values = close_data[0]

        data_store: Dict[str, tuple[np.ndarray, np.ndarray]] = {"close": close_data}
        for key in ("open", "high", "low"):
            arr = _column_data(key, ticker)
            if arr is not None:
                data_store[key] = arr

        masks = [mask for _, mask in data_store.values()]
        if not masks:
            continue
        if len(masks) == 1:
            common_mask = masks[0].copy()
        else:
            common_mask = np.logical_and.reduce(masks)
        valid_count = int(common_mask.sum())
        if valid_count < min_periods:
            continue

        rec: Dict[str, Union[float, str]] = {"ticker": ticker}

        close = close_values[common_mask]
        if close.size < 2:
            r_cc = np.array([], dtype=np.float64)
        else:
            r_cc = np.diff(np.log(close))

        if "high" in data_store and "low" in data_store:
            high = data_store["high"][0][common_mask]
            low = data_store["low"][0][common_mask]
            hl = np.log(high / low) ** 2
            if hl.size >= min_periods:
                var = hl.mean() / (4.0 * ln2)
                if np.isfinite(var) and var >= 0:
                    rec["parkinson_vol"] = float(np.sqrt(var * per_year))

        if {"open", "high", "low"}.issubset(data_store):
            open_values = data_store["open"][0][common_mask]
            high = data_store["high"][0][common_mask]
            low = data_store["low"][0][common_mask]
            log_hl = np.log(high / low) ** 2
            log_co = np.log(close / open_values) ** 2
            if log_hl.size >= min_periods and log_co.size >= min_periods:
                gk_var = 0.5 * log_hl.mean() - (2.0 * ln2 - 1.0) * log_co.mean()
                if np.isfinite(gk_var) and gk_var >= 0:
                    rec["gk_vol"] = float(np.sqrt(gk_var * per_year))

                term_rs = (
                    np.log(high / close) * np.log(high / open_values)
                    + np.log(low / close) * np.log(low / open_values)
                )
                rs_var = term_rs.mean()
                if np.isfinite(rs_var) and rs_var >= 0:
                    rec["rs_vol"] = float(np.sqrt(rs_var * per_year))

                if close.size >= 2:
                    prev_close = close[:-1]
                    r_o = np.log(open_values[1:] / prev_close)
                    r_c = np.log(close / open_values)
                    if r_o.size >= 1 and r_c.size >= 1:
                        sigma_o2 = r_o.var(ddof=1)
                        sigma_c2 = r_c.var(ddof=1)
                        sigma_rs = term_rs.mean()
                        if (
                            np.isfinite(sigma_o2)
                            and np.isfinite(sigma_c2)
                            and np.isfinite(sigma_rs)
                        ):
                            k = 0.34
                            yz_var = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs
                            if yz_var >= 0:
                                rec["yz_vol"] = float(np.sqrt(yz_var * per_year))

        if r_cc.size >= min_periods:
            if r_cc.size >= 2:
                var = r_cc.var(ddof=1)
            elif r_cc.size == 1:
                var = float(r_cc[0] ** 2)
            else:
                var = float("nan")
            for x in r_cc[-min_periods:]:
                var = ewma_lambda * var + (1.0 - ewma_lambda) * (x * x)
            if np.isfinite(var) and var >= 0:
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


# ---------------------------------------------------------------------------
# Metric registry and plugin loading
# ---------------------------------------------------------------------------

# Registry mapping metric names to callables.  Each callable should accept a
# price matrix and return a tidy DataFrame containing a ``ticker`` column and a
# column named after the metric.
METRIC_REGISTRY: Dict[str, Callable[..., pd.DataFrame]] = {}

# Human-readable titles for built-in metrics.  Plugins fall back to a title-cased
# version of their key when not present in this mapping.
METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "cc_vol": "Close-to-Close Volatility",
    "parkinson_vol": "Parkinson Volatility",
    "gk_vol": "Garman-Klass Volatility",
    "rs_vol": "Rogers-Satchell Volatility",
    "yz_vol": "Yang-Zhang Volatility",
    "ewma_vol": "EWMA Volatility",
    "mad_vol": "Median Absolute Deviation Volatility",
    "sharpe_ratio": "Sharpe Ratio",
    "max_drawdown": "Maximum Drawdown",
    "var": "Value at Risk",
    "sortino": "Sortino Ratio",
}


def register_metric(name: str, func: Callable[..., pd.DataFrame]) -> None:
    """Register ``func`` under ``name`` in the metric registry."""

    METRIC_REGISTRY[name] = func


def metric_display_name(metric_key: str) -> str:
    """Return a human-readable title for ``metric_key``."""

    normalized = metric_key.strip()
    if not normalized:
        return metric_key
    return METRIC_DISPLAY_NAMES.get(normalized, normalized.replace("_", " ").title())


def _extract_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of close prices from a Yahoo-style price matrix."""

    if isinstance(prices.columns, pd.MultiIndex):
        if "Adj Close" in prices.columns.get_level_values(0):
            return prices["Adj Close"]
        return prices["Close"]
    return prices["Adj Close"] if "Adj Close" in prices.columns else prices["Close"]


def _cc_vol(
    prices: pd.DataFrame,
    *,
    min_periods: int | None = None,
    interval: str = "1d",
    **_: object,
) -> pd.DataFrame:
    close = _extract_close(prices)
    return annualized_volatility(close, min_periods=min_periods, interval=interval).rename(
        columns={"annualized_volatility": "cc_vol"}
    )


def _sharpe_ratio_metric(prices: pd.DataFrame, **_: object) -> pd.DataFrame:
    close = _extract_close(prices)
    return sharpe_ratio(close)


def _max_drawdown_metric(prices: pd.DataFrame, **_: object) -> pd.DataFrame:
    close = _extract_close(prices)
    return max_drawdown(close)


def _var_metric(prices: pd.DataFrame, **_: object) -> pd.DataFrame:
    close = _extract_close(prices)
    return value_at_risk(close)


def _sortino_metric(prices: pd.DataFrame, **_: object) -> pd.DataFrame:
    close = _extract_close(prices)
    return sortino_ratio(close)


def _extra_metric(name: str) -> Callable[..., pd.DataFrame]:
    def wrapper(
        prices: pd.DataFrame,
        tickers: List[str] | None = None,
        *,
        min_periods: int | None = None,
        interval: str = "1d",
        **_: object,
    ) -> pd.DataFrame:
        df = additional_volatility_measures(
            prices,
            tickers or [],
            min_periods=min_periods or 0,
            interval=interval,
        )
        if name in df.columns:
            return df[["ticker", name]]
        return pd.DataFrame(columns=["ticker", name])

    return wrapper


# Register built-in metrics
register_metric("cc_vol", _cc_vol)
for _name in ["parkinson_vol", "gk_vol", "rs_vol", "yz_vol", "ewma_vol", "mad_vol"]:
    register_metric(_name, _extra_metric(_name))
register_metric("sharpe_ratio", _sharpe_ratio_metric)
register_metric("max_drawdown", _max_drawdown_metric)
register_metric("var", _var_metric)
register_metric("sortino", _sortino_metric)


def load_plugins() -> None:
    """Load metric plugins via entry points or a local ``metrics/`` directory."""

    # Entry points
    try:  # pragma: no cover - Python version differences
        from importlib.metadata import entry_points
    except Exception:  # pragma: no cover - fallback for very old Python
        entry_points = None  # type: ignore

    if entry_points is not None:  # pragma: no branch
        eps_iter: Iterable
        try:
            eps_iter = entry_points(group="highest_volatility.metrics")
        except TypeError:  # pragma: no cover - Python <3.10
            all_eps = entry_points()
            if hasattr(all_eps, "select"):
                eps_iter = all_eps.select(group="highest_volatility.metrics")
            else:  # pragma: no cover - legacy API
                eps_iter = cast(
                    Iterable,
                    all_eps.get("highest_volatility.metrics", ()) or (),
                )
        for ep in eps_iter:
            try:
                func = ep.load()
            except Exception:  # pragma: no cover - plugin errors shouldn't crash
                continue
            register_metric(ep.name, func)

    # ``metrics`` directory relative to current working directory
    plugin_dir = Path.cwd() / "metrics"
    if plugin_dir.is_dir():
        for path in plugin_dir.glob("*.py"):
            if path.name.startswith("_"):
                continue
            try:
                import importlib.util

                spec = importlib.util.spec_from_file_location(path.stem, path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "METRICS"):
                        for name, func in getattr(module, "METRICS").items():
                            register_metric(name, func)
            except Exception:  # pragma: no cover - best effort
                continue


# Load plugins on import (best effort)
load_plugins()


__all__: List[str] = [
    "TRADING_DAYS_PER_YEAR",
    "TRADING_MINUTES_PER_DAY",
    "daily_returns",
    "annualized_volatility",
    "periods_per_year",
    "additional_volatility_measures",
    "max_drawdown",
    "rolling_volatility",
    "sharpe_ratio",
    "value_at_risk",
    "sortino_ratio",
    "METRIC_REGISTRY",
    "register_metric",
    "load_plugins",
]
