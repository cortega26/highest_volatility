"""Regression tests for the Streamlit analysis session state helper."""

from __future__ import annotations

from typing import Any

from highest_volatility.app.analysis_state import (
    ANALYSIS_READY_KEY,
    CONFIG_SIGNATURE_KEY,
    update_analysis_ready_flag,
)


def _make_config(
    *,
    top_n: int = 10,
    lookback: int = 60,
    interval: str = "1d",
    metric: str = "volatility",
    min_days: int = 30,
    prepost: bool = False,
    validate: bool = False,
    async_fetch: bool = False,
) -> tuple[Any, ...]:
    """Return a tuple matching the AnalysisConfig.signature ordering."""

    return (top_n, lookback, interval, metric, min_days, prepost, validate, async_fetch)


def test_update_analysis_ready_flag_tracks_configuration_transitions() -> None:
    """The session flag should persist for display-only interactions."""

    session: dict[str, Any] = {}

    first_config = _make_config()
    second_config = _make_config(top_n=20)

    ready = update_analysis_ready_flag(session, first_config, run_requested=False)
    assert not ready
    assert session[ANALYSIS_READY_KEY] is False
    assert session[CONFIG_SIGNATURE_KEY] == first_config

    ready = update_analysis_ready_flag(session, first_config, run_requested=True)
    assert ready
    assert session[ANALYSIS_READY_KEY] is True

    ready = update_analysis_ready_flag(session, first_config, run_requested=False)
    assert ready
    assert session[ANALYSIS_READY_KEY] is True

    ready = update_analysis_ready_flag(session, second_config, run_requested=False)
    assert not ready
    assert session[ANALYSIS_READY_KEY] is False
    assert session[CONFIG_SIGNATURE_KEY] == second_config

    ready = update_analysis_ready_flag(session, second_config, run_requested=True)
    assert ready
    assert session[ANALYSIS_READY_KEY] is True
