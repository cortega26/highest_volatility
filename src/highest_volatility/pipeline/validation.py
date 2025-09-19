"""Validation utilities for cached price data."""

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from cache.store import Manifest

FREQ_MAP = {
    "1m": "1min",
    "5m": "5min",
    "10m": "10min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
    "1h": "60min",
    "1wk": "W",
    "1mo": "M",
}

DELTA_MAP = {
    "1m": pd.Timedelta("1min"),
    "5m": pd.Timedelta("5min"),
    "10m": pd.Timedelta("10min"),
    "15m": pd.Timedelta("15min"),
    "30m": pd.Timedelta("30min"),
    "60m": pd.Timedelta("60min"),
    "1h": pd.Timedelta("60min"),
    "1d": pd.Timedelta("1D"),
}


def validate_cache(df: pd.DataFrame, manifest: "Manifest") -> None:
    """Validate cached DataFrame against its manifest.

    Parameters
    ----------
    df:
        Price data to validate.
    manifest:
        Manifest describing the cached data.

    Raises
    ------
    ValueError
        If any validation check fails.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Index must be unique and sorted")
    if df.isna().any().any():
        raise ValueError("Data contains NaNs")

    start = pd.to_datetime(manifest.start)
    end = pd.to_datetime(manifest.end)
    if df.index.min().date() != start.date() or df.index.max().date() != end.date():
        raise ValueError("Index range does not match manifest")
    if len(df) != manifest.rows:
        raise ValueError("Row count mismatch with manifest")

    try:
        freq = pd.infer_freq(df.index) if len(df.index) >= 3 else None
    except ValueError:
        freq = None
    freq = freq or FREQ_MAP.get(manifest.interval)
    if freq:
        if (manifest.interval.endswith("m") and manifest.interval != "1mo") or freq.endswith("T") or freq.endswith("min"):
            for _, group in df.groupby(df.index.date):
                expected = pd.date_range(group.index.min(), group.index.max(), freq=freq)
                if len(expected) != len(group):
                    raise ValueError("Gap detected in index")
        else:
            expected = pd.date_range(df.index.min(), df.index.max(), freq=freq)
            if len(expected) != len(df):
                raise ValueError("Gap detected in index")
    else:
        delta = DELTA_MAP.get(manifest.interval)
        if delta is not None and not df.index.to_series().diff().dropna().le(delta).all():
            raise ValueError("Gap detected in index")
