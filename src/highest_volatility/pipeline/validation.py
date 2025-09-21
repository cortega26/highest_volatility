"""Validation utilities for cached price data.

Sparse intraday gaps approved by a datasource can be passed via the
``allowed_gaps`` parameter when calling :func:`validate_cache`. Any detected
missing bars that match these timestamps are ignored during validation while
unexpected holes still trigger errors.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, TYPE_CHECKING, Optional, Set, Tuple

import pandas as pd
from pandas.tseries.frequencies import to_offset

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

from highest_volatility.logging import get_logger


logger = get_logger(__name__, component="cache_validation")

class _FallbackNYSEHolidayCalendar(AbstractHolidayCalendar):
    """Approximation of NYSE full-day holidays when ``pandas-market-calendars`` is unavailable.

    The special-closure dates mirror historical NYSE shutdowns documented at
    https://www.nyse.com/markets/hours-calendars/historical-holidays.
    """

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
        # One-off full-day closures observed by the NYSE since the mid-1980s.
        Holiday("NYSESpecialClosure1985Sep27", year=1985, month=9, day=27),
        Holiday("NYSESpecialClosure1994Apr27", year=1994, month=4, day=27),
        Holiday("NYSESpecialClosure2001Sep11", year=2001, month=9, day=11),
        Holiday("NYSESpecialClosure2001Sep12", year=2001, month=9, day=12),
        Holiday("NYSESpecialClosure2001Sep13", year=2001, month=9, day=13),
        Holiday("NYSESpecialClosure2001Sep14", year=2001, month=9, day=14),
        Holiday("NYSESpecialClosure2004Jun11", year=2004, month=6, day=11),
        Holiday("NYSESpecialClosure2007Jan02", year=2007, month=1, day=2),
        Holiday("NYSESpecialClosure2012Oct29", year=2012, month=10, day=29),
        Holiday("NYSESpecialClosure2012Oct30", year=2012, month=10, day=30),
        Holiday("NYSESpecialClosure2018Dec05", year=2018, month=12, day=5),
    ]


_FALLBACK_CALENDAR = _FallbackNYSEHolidayCalendar()

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from highest_volatility.cache.store import Manifest

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


def _infer_timedelta(freq: Optional[str], interval: str) -> Optional[pd.Timedelta]:
    """Return the expected spacing between rows for the provided interval."""

    if freq:
        try:
            offset = to_offset(freq)
        except ValueError:
            offset = None
        else:
            nanos = getattr(offset, "nanos", None)
            if nanos:
                return pd.Timedelta(nanos, unit="ns")
    return DELTA_MAP.get(interval)


def _session_bounds(
    ts: pd.Timestamp,
    calendar,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the session open/close for the provided timestamp's trading day."""

    session_date = ts.date()
    tz = ts.tz

    if calendar is not None:
        schedule = calendar.schedule(start_date=session_date, end_date=session_date)
        if schedule.empty:
            return None
        open_ts = schedule.iloc[0]["market_open"]
        close_ts = schedule.iloc[0]["market_close"]
        if tz is not None:
            open_ts = open_ts.tz_convert(tz)
            close_ts = close_ts.tz_convert(tz)
        else:
            session_tz = getattr(calendar, "tz", None)
            if session_tz is not None:
                open_ts = open_ts.tz_convert(session_tz)
                close_ts = close_ts.tz_convert(session_tz)
            open_ts = open_ts.tz_localize(None)
            close_ts = close_ts.tz_localize(None)
        return open_ts, close_ts

    open_time = pd.Timestamp(session_date).replace(hour=9, minute=30)
    close_time = pd.Timestamp(session_date).replace(hour=16, minute=0)
    if tz is not None:
        open_time = open_time.tz_localize(tz)
        close_time = close_time.tz_localize(tz)
    return open_time, close_time


def _normalize_timestamp_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    """Return a timezone-aware UTC timestamp for comparisons."""

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _validate_intraday_index(
    index: pd.DatetimeIndex,
    *,
    freq: Optional[str],
    interval: str,
    ticker: Optional[str] = None,
    allowed_gaps: Optional[Set[pd.Timestamp]] = None,
) -> None:
    """Validate that intraday data has no gaps during regular trading hours."""

    target_delta = _infer_timedelta(freq, interval)
    if target_delta is None or index.empty:
        return

    calendar = _get_trading_calendar()
    normalized = index.normalize()
    missing_timestamps: list[pd.Timestamp] = []
    for session_day in pd.unique(normalized):
        mask = normalized == session_day
        day_index = index[mask]
        session_reference = day_index[0]
        bounds = _session_bounds(session_reference, calendar)
        if bounds is None:
            continue
        session_open, session_close = bounds
        in_hours = day_index[(day_index >= session_open) & (day_index <= session_close)]
        if len(in_hours) <= 1:
            continue
        for previous, current in zip(in_hours[:-1], in_hours[1:]):
            delta = current - previous
            if delta > target_delta:
                gap_start = previous + target_delta
                gap_end = current - target_delta
                if gap_start <= gap_end:
                    missing_range = pd.date_range(gap_start, gap_end, freq=target_delta)
                    missing_timestamps.extend(missing_range)
                else:
                    missing_timestamps.append(gap_start)
    if missing_timestamps:
        normalized_missing = sorted(
            {_normalize_timestamp_utc(ts) for ts in missing_timestamps}
        )
        if allowed_gaps:
            normalized_missing = [
                ts for ts in normalized_missing if ts not in allowed_gaps
            ]
        if not normalized_missing:
            return
        missing_formatted = [ts.isoformat() for ts in normalized_missing]
        context = {k: v for k, v in {"ticker": ticker, "interval": interval}.items() if v}
        logger.error(
            {
                "event": "intraday_gap_detected",
                "missing_timestamps": missing_formatted,
            },
            context=context or None,
        )
        raise ValueError(
            f"Gap detected in index: missing intraday timestamps {missing_formatted}"
        )


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


def _validate_business_day_index(
    index: pd.DatetimeIndex,
    *,
    ticker: Optional[str] = None,
    interval: Optional[str] = None,
) -> None:
    """Ensure an index covers every observed trading day between endpoints."""

    if index.empty:
        return

    normalized = index.normalize()
    observed = pd.Index(normalized.date)
    expected = _expected_trading_dates(normalized.min(), normalized.max())
    observed_set = set(observed)
    missing_expected = [session for session in expected if session not in observed_set]
    if missing_expected:
        missing_formatted = [str(date) for date in missing_expected]
        context = {k: v for k, v in {"ticker": ticker, "interval": interval}.items() if v}
        logger.error(
            {
                "event": "business_day_gap_detected",
                "missing_trading_dates": missing_formatted,
            },
            context=context or None,
        )
        raise ValueError(
            f"Gap detected in index: missing trading dates {missing_formatted}"
        )


def validate_cache(
    df: pd.DataFrame,
    manifest: "Manifest",
    *,
    allowed_gaps: Optional[Iterable[str | pd.Timestamp]] = None,
) -> None:
    """Validate cached DataFrame against its manifest.

    Parameters
    ----------
    df:
        Price data to validate.
    manifest:
        Manifest describing the cached data.
    allowed_gaps:
        Optional iterable of timestamps sanctioned by the datasource. Any
        detected intraday gaps that match these values will be ignored.

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

    allowed_gap_set: Optional[Set[pd.Timestamp]] = None
    if allowed_gaps:
        normalized: Set[pd.Timestamp] = set()
        for gap in allowed_gaps:
            if gap is None:
                continue
            normalized.add(_normalize_timestamp_utc(gap))
        if normalized:
            allowed_gap_set = normalized

    try:
        freq = pd.infer_freq(df.index) if len(df.index) >= 3 else None
    except ValueError:
        freq = None
    freq = freq or FREQ_MAP.get(manifest.interval)
    if freq:
        normalized_freq = freq.upper()
        if (
            manifest.interval.lower().endswith(("m", "h"))
            or freq.endswith("T")
            or freq.endswith("min")
            or normalized_freq.endswith("H")
        ):
            _validate_intraday_index(
                df.index,
                freq=freq,
                interval=manifest.interval,
                ticker=getattr(manifest, "ticker", None),
                allowed_gaps=allowed_gap_set,
            )
        elif _is_daily_interval(manifest.interval) or normalized_freq in {"B", "C"} or normalized_freq.endswith("D"):
            _validate_business_day_index(
                df.index,
                ticker=getattr(manifest, "ticker", None),
                interval=manifest.interval,
            )
        else:
            expected = pd.date_range(df.index.min(), df.index.max(), freq=freq)
            if len(expected) != len(df):
                raise ValueError("Gap detected in index")
    else:
        if _is_daily_interval(manifest.interval):
            _validate_business_day_index(
                df.index,
                ticker=getattr(manifest, "ticker", None),
                interval=manifest.interval,
            )
        elif manifest.interval.lower().endswith(("m", "h")):
            _validate_intraday_index(
                df.index,
                freq=None,
                interval=manifest.interval,
                ticker=getattr(manifest, "ticker", None),
                allowed_gaps=allowed_gap_set,
            )
        else:
            delta = DELTA_MAP.get(manifest.interval)
            if delta is not None and not df.index.to_series().diff().dropna().le(delta).all():
                raise ValueError("Gap detected in index")

