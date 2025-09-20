"""Tests for :func:`highest_volatility.compute.metrics.periods_per_year`."""

from __future__ import annotations

import pytest

from highest_volatility.compute.metrics import (
    TRADING_DAYS_PER_YEAR,
    TRADING_MINUTES_PER_DAY,
    periods_per_year,
)


def test_periods_per_year_supports_intraday_and_hourly_intervals() -> None:
    """Minute and hour based intervals should scale from trading minutes."""

    expected_minutes = (
        TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY / 15.0
    )
    assert periods_per_year("15m") == pytest.approx(expected_minutes)
    assert periods_per_year("15M") == pytest.approx(expected_minutes)

    expected_hours = TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY / 60.0
    assert periods_per_year("1h") == pytest.approx(expected_hours)
    assert periods_per_year("1H") == pytest.approx(expected_hours)


def test_periods_per_year_supports_daily_weekly_and_monthly_intervals() -> None:
    """Daily and coarser intervals must map to sensible annual counts."""

    assert periods_per_year("1d") == pytest.approx(TRADING_DAYS_PER_YEAR)
    assert periods_per_year("5d") == pytest.approx(TRADING_DAYS_PER_YEAR / 5.0)
    assert periods_per_year("1wk") == pytest.approx(52.0)
    assert periods_per_year("3mo") == pytest.approx(4.0)


@pytest.mark.parametrize("interval", ["bad", "0m", "-5d", "10"], ids=str)
def test_periods_per_year_rejects_invalid_intervals(interval: str) -> None:
    """Invalid strings should raise ``ValueError`` to signal misuse."""

    with pytest.raises(ValueError):
        periods_per_year(interval)

