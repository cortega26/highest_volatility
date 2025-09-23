"""Utilities to support the Streamlit frontend for Highest Volatility."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import pandas as pd

from highest_volatility.compute.metrics import (
    METRIC_REGISTRY,
    additional_volatility_measures,
)


@dataclass(frozen=True, slots=True)
class SanitizedPrices:
    """Container describing the outcome of price sanitisation."""

    filtered: pd.DataFrame
    close_only: pd.DataFrame
    dropped_short: list[str]
    dropped_duplicate: list[str]
    dropped_empty: list[str]

    @property
    def has_close_data(self) -> bool:
        """Return ``True`` when at least one ticker survived sanitisation."""

        return not self.close_only.empty

    @property
    def available_tickers(self) -> list[str]:
        """Return the tickers present in the sanitised close-only frame."""

        return list(self.close_only.columns)

    def summarize_drops(self, *, min_days: int) -> list[str]:
        """Generate human-friendly summaries of dropped tickers."""

        messages: list[str] = []
        if self.dropped_short:
            messages.append(
                f"{len(self.dropped_short)} tickers lacked the required {min_days} observations."
            )
        if self.dropped_duplicate:
            messages.append(
                f"{len(self.dropped_duplicate)} tickers were removed due to duplicate price series."
            )
        if self.dropped_empty:
            messages.append(
                f"{len(self.dropped_empty)} tickers contained only empty values after cleaning."
            )
        return messages


def extract_close_frame(
    prices: pd.DataFrame, *, tickers: Sequence[str] | None = None
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices keyed by ticker symbol."""

    if prices.empty:
        return pd.DataFrame()

    if isinstance(prices.columns, pd.MultiIndex):
        level0 = prices.columns.get_level_values(0)
        field = "Adj Close" if "Adj Close" in level0 else "Close"
        close = prices[field]
        return close.copy()

    field = "Adj Close" if "Adj Close" in prices.columns else "Close"
    if field not in prices.columns:
        return pd.DataFrame()

    close = prices[[field]].copy()
    if tickers and len(tickers) == 1:
        close.columns = [tickers[0]]
    else:
        name = tickers[0] if tickers else field
        close.columns = [name]
    return close


def sanitize_price_matrix(
    prices: pd.DataFrame,
    *,
    min_days: int,
    tickers: Sequence[str] | None = None,
) -> SanitizedPrices:
    """Return a sanitized price matrix and metadata describing the changes."""

    close = extract_close_frame(prices, tickers=tickers)
    if close.empty:
        filtered = prices.iloc[0:0]
        return SanitizedPrices(filtered, close, [], [], [])

    close = close.loc[:, ~close.columns.duplicated(keep="first")]

    dropped_short: list[str] = []
    keep_columns: list[str] = []
    for column in close.columns:
        if int(close[column].notna().sum()) >= min_days:
            keep_columns.append(column)
        else:
            dropped_short.append(column)

    working = close[keep_columns].copy()

    dropped_empty: list[str] = []
    dropped_duplicate: list[str] = []
    tail_n = min(250, max(60, min_days))
    seen: dict[str, str] = {}
    for column in list(working.columns):
        series = working[column].tail(tail_n).astype(float).ffill().bfill()
        if series.dropna().empty:
            working = working.drop(columns=[column])
            dropped_empty.append(column)
            continue
        digest = hashlib.sha1(series.to_numpy().tobytes()).hexdigest()
        if digest in seen:
            working = working.drop(columns=[column])
            dropped_duplicate.append(column)
        else:
            seen[digest] = column

    keep_tickers = list(working.columns)
    if isinstance(prices.columns, pd.MultiIndex):
        mask = prices.columns.get_level_values(1).isin(keep_tickers)
        filtered = prices.loc[:, mask].copy()
    else:
        filtered = prices.copy()

    return SanitizedPrices(filtered, working, dropped_short, dropped_duplicate, dropped_empty)


def _join_metric_columns(base: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Join metric columns from ``frame`` onto ``base`` using the ticker index."""

    if frame is None or frame.empty or "ticker" not in frame.columns:
        return base

    cleaned = frame.drop_duplicates(subset=["ticker"], keep="first")
    if cleaned.empty:
        return base

    joined = cleaned.set_index("ticker")
    new_columns = [col for col in joined.columns if col not in base.columns]
    if not new_columns:
        return base

    return base.join(joined[new_columns], how="left")


def _augment_with_registered_metrics(
    table: pd.DataFrame,
    *,
    prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    tickers: Sequence[str],
    min_days: int,
    interval: str,
    metric_key: str,
) -> pd.DataFrame:
    """Augment ``table`` with additional metrics sourced from the registry."""

    if not len(tickers):
        return table

    augmented = table

    try:
        extra = additional_volatility_measures(
            prices,
            list(tickers),
            min_periods=min_days,
            interval=interval,
        )
    except Exception:  # pragma: no cover - best-effort enrichment
        extra = pd.DataFrame()

    augmented = _join_metric_columns(augmented, extra)

    base_kwargs = {
        "prices": prices,
        "tickers": list(tickers),
        "min_periods": min_days,
        "interval": interval,
    }
    kwargs_with_close = {**base_kwargs, "close": close_prices}

    for key, func in METRIC_REGISTRY.items():
        if key == metric_key or key in augmented.columns:
            continue

        result: pd.DataFrame | None = None
        try:
            result = func(**kwargs_with_close)
        except TypeError:
            try:
                result = func(**base_kwargs)
            except Exception:  # pragma: no cover - ignore broken metrics
                result = None
        except Exception:  # pragma: no cover - ignore broken metrics
            result = None

        augmented = _join_metric_columns(augmented, result if isinstance(result, pd.DataFrame) else pd.DataFrame())

    return augmented


def prepare_metric_table(
    prices: pd.DataFrame,
    *,
    metric_key: str,
    min_days: int,
    interval: str,
    tickers: Sequence[str],
    fortune: pd.DataFrame | None,
    close_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the requested metric and return a formatted table."""

    if metric_key not in METRIC_REGISTRY:
        raise KeyError(f"Unknown metric '{metric_key}'")

    if prices.empty:
        return pd.DataFrame()

    metric_func = METRIC_REGISTRY[metric_key]

    extra_kwargs: dict[str, pd.DataFrame] = {}
    if close_prices is not None and not close_prices.empty:
        extra_kwargs["close"] = close_prices

    metrics = metric_func(
        prices,
        tickers=list(tickers),
        min_periods=min_days,
        interval=interval,
        **extra_kwargs,
    )
    if metrics.empty:
        return metrics

    metrics = metrics.drop_duplicates(subset=["ticker"], keep="first")
    metrics = metrics.set_index("ticker")

    table = metrics
    if fortune is not None and not fortune.empty:
        fortune_clean = fortune.drop_duplicates(subset=["ticker"]).set_index("ticker")
        cols: list[str] = []
        for name in ("rank", "company"):
            if name in fortune_clean.columns:
                cols.append(name)
        if cols:
            table = fortune_clean[cols].join(table, how="inner")

    if close_prices is not None and not close_prices.empty:
        table = _augment_with_registered_metrics(
            table,
            prices=prices,
            close_prices=close_prices,
            tickers=tickers,
            min_days=min_days,
            interval=interval,
            metric_key=metric_key,
        )

    table = table.reset_index()
    if metric_key in table.columns:
        table = table.sort_values(metric_key, ascending=False)

    return table.reset_index(drop=True)


__all__ = [
    "extract_close_frame",
    "sanitize_price_matrix",
    "prepare_metric_table",
    "SanitizedPrices",
]
