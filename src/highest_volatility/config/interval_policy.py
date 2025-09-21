"""Backfill window policy for supported intervals."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict

# Mapping of interval -> timedelta window for full backfill
INTERVAL_WINDOWS: Dict[str, timedelta] = {
    "1m": timedelta(days=7),
    "5m": timedelta(days=60),
    "15m": timedelta(days=60),
    "30m": timedelta(days=60),
    "60m": timedelta(days=730),  # alias for "1h"
    "1h": timedelta(days=730),  # ~2 years
    "1d": timedelta(days=365 * 35),
    "1wk": timedelta(days=365 * 40),
    "1mo": timedelta(days=365 * 40),
}


def full_backfill_start(interval: str, today: date | None = None) -> date:
    """Return the start date for a full backfill for ``interval``.

    Parameters
    ----------
    interval:
        Interval key such as ``"1d"`` or ``"1m"``.
    today:
        Override the reference ``today`` (defaults to :func:`date.today`).
    """

    if interval not in INTERVAL_WINDOWS:
        raise KeyError(f"Unsupported interval: {interval}")
    today = today or date.today()
    return today - INTERVAL_WINDOWS[interval]
