"""Tests for Streamlit input helpers.

Run with:
    pytest tests/test_streamlit_inputs.py
"""

from __future__ import annotations

from highest_volatility.app import streamlit_app


def test_clamp_int_within_bounds() -> None:
    assert streamlit_app._clamp_int(50, minimum=10, maximum=100) == 50


def test_clamp_int_above_maximum() -> None:
    assert streamlit_app._clamp_int(200, minimum=10, maximum=100) == 100
