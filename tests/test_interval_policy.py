"""Tests for interval backfill policy aliases."""

from datetime import date

import pytest

from src.config.interval_policy import INTERVAL_WINDOWS, full_backfill_start


def test_60m_interval_alias_matches_1h() -> None:
    today = date(2024, 1, 1)
    assert INTERVAL_WINDOWS["60m"] == INTERVAL_WINDOWS["1h"]
    assert full_backfill_start("60m", today=today) == full_backfill_start("1h", today=today)


def test_full_backfill_start_rejects_unknown_interval() -> None:
    with pytest.raises(KeyError):
        full_backfill_start("unknown")
