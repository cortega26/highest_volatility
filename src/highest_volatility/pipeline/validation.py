"""Validation utilities for cached price data."""

from functools import lru_cache
from typing import TYPE_CHECKING

import pandas as pd

try:  # pragma: no cover - optional dependency for trading calendars
    import pandas_market_calendars as mcal
except ImportError:  # pragma: no cover - fallback to plain business days
    mcal = None

from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    GoodFriday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)


class _FallbackNYSEHolidayCalendar(AbstractHolidayCalendar):
    """Approximation of NYSE full-day holidays when ``pandas-market-calendars`` is unavailable."""

    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            observance=nearest_workday,
            start_date="2021-06-19",
        ),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("ChristmasDay", month=12, day=25, observance=nearest_workday),
    ]


_FALLBACK_CALENDAR = _FallbackNYSEHolidayCalendar()

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from src.cache.store import Manifest

FREQ_MAP = {
    "1m": "1min",
    "5m": "5min",
    "10m": "10min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
    "1h": "60min",
    "1d": "B",
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


def _is_daily_interval(interval: str) -> bool:
    """Return ``True`` when the manifest interval represents daily data."""

    return interval.lower().endswith("d")


@lru_cache(maxsize=1)
def _get_trading_calendar():
    """Return the XNYS trading calendar if available, otherwise ``None``."""

    if mcal is None:  # pragma: no cover - handled in fallback path
        return None
    for calendar_name in ("XNYS", "NYSE"):
        try:
            return mcal.get_calendar(calendar_name)
        except Exception:  # pragma: no cover - other names tried before fallback
            continue
    return None


def _expected_trading_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.Index:
    """Compute the ordered trading dates between ``start`` and ``end``."""

    calendar = _get_trading_calendar()
    if calendar is None:
        holidays = _FALLBACK_CALENDAR.holidays(
            start=start.normalize(),
            end=end.normalize(),
        )
        holidays = set(pd.DatetimeIndex(holidays).date)
        sessions = pd.bdate_range(start.normalize(), end.normalize(), tz=start.tz)
        return pd.Index(
            [session.normalize().date() for session in sessions if session.date() not in holidays]
        )

    valid_sessions = calendar.valid_days(start.date(), end.date())
    valid_sessions = pd.DatetimeIndex(valid_sessions)
    return pd.Index([session.date() for session in valid_sessions])


def _validate_business_day_index(index: pd.DatetimeIndex) -> None:
    """Ensure an index covers every observed trading day between endpoints."""

    if index.empty:
        return

    normalized = index.normalize()
    observed = pd.Index(normalized.date)
    expected = _expected_trading_dates(normalized.min(), normalized.max())
    observed_set = set(observed)
    missing_expected = [session for session in expected if session not in observed_set]
    if missing_expected:
        raise ValueError("Gap detected in index")


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
        elif _is_daily_interval(manifest.interval) or freq.upper() in {"B", "C"} or freq.upper().endswith("D"):
            _validate_business_day_index(df.index)
        else:
            expected = pd.date_range(df.index.min(), df.index.max(), freq=freq)
            if len(expected) != len(df):
                raise ValueError("Gap detected in index")
    else:
        if _is_daily_interval(manifest.interval):
            _validate_business_day_index(df.index)
        else:
            delta = DELTA_MAP.get(manifest.interval)
            if delta is not None and not df.index.to_series().diff().dropna().le(delta).all():
                raise ValueError("Gap detected in index")
