"""Helpers for managing persisted Streamlit analysis state."""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence

ANALYSIS_READY_KEY = "analysis_ready"
CONFIG_SIGNATURE_KEY = "analysis_config_signature"


def update_analysis_ready_flag(
    session_state: MutableMapping[str, Any],
    config_signature: Sequence[Any],
    run_requested: bool,
) -> bool:
    """Update the analysis-ready flag and return the resulting status.

    Parameters
    ----------
    session_state:
        A mutable mapping compatible with ``st.session_state`` where the flag
        and configuration snapshot are stored.
    config_signature:
        A hashable snapshot of the sidebar configuration (typically obtained
        from :meth:`highest_volatility.app.streamlit_app.AnalysisConfig.signature`).
    run_requested:
        Indicates whether the **Run analysis** button was pressed in the
        current Streamlit rerun.

    Returns
    -------
    bool
        ``True`` when previously fetched analysis results should be
        considered valid for display, ``False`` otherwise.
    """

    previous_signature = session_state.get(CONFIG_SIGNATURE_KEY)
    analysis_ready = bool(session_state.get(ANALYSIS_READY_KEY))

    if previous_signature != tuple(config_signature):
        analysis_ready = False

    if run_requested:
        analysis_ready = True

    session_state[ANALYSIS_READY_KEY] = analysis_ready
    session_state[CONFIG_SIGNATURE_KEY] = tuple(config_signature)

    return analysis_ready
