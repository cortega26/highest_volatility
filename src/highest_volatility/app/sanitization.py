"""Helpers for cleaning price matrices prior to metric computation."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def sanitize_close(
    close: pd.DataFrame, min_days: int
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Drop incomplete or duplicate price series from a close-price matrix.

    Parameters
    ----------
    close:
        DataFrame of close or adjusted close prices keyed by ticker.
    min_days:
        Minimum number of non-null observations required per ticker.

    Returns
    -------
    tuple
        ``(clean_close, dropped_short, dropped_duplicate)`` where ``clean_close``
        retains valid tickers only, ``dropped_short`` lists tickers removed for
        insufficient history, and ``dropped_duplicate`` lists those removed as
        duplicate series (matching recent tails).
    """

    import hashlib

    dropped_short: List[str] = []
    dropped_dupe: List[str] = []

    clean = close.loc[:, ~close.columns.duplicated(keep="first")]

    keep_columns: List[str] = []
    for column in clean.columns:
        if int(clean[column].notna().sum()) >= min_days:
            keep_columns.append(column)
        else:
            dropped_short.append(column)

    clean = clean[keep_columns].copy()

    seen: dict[str, str] = {}
    tail_n = min(250, max(60, min_days))
    for column in list(clean.columns):
        series = clean[column].tail(tail_n).astype(float).ffill().bfill()
        if series.dropna().empty:
            clean = clean.drop(columns=[column])
            continue
        signature = hashlib.sha1(series.to_numpy().tobytes()).hexdigest()
        if signature in seen:
            dropped_dupe.append(column)
            clean = clean.drop(columns=[column])
        else:
            seen[signature] = column

    return clean, dropped_short, dropped_dupe
