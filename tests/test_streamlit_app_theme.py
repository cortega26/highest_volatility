"""Tests for theme-aware styling helpers in the Streamlit app."""

from __future__ import annotations

from typing import Any

import pytest

from highest_volatility.app import streamlit_app


class StubTheme:
    """Lightweight helper to emulate ``st.get_option`` behaviour."""

    def __init__(self, mapping: dict[str, Any]):
        self._mapping = mapping

    def get_option(self, key: str) -> Any:
        return self._mapping.get(key)


@pytest.mark.parametrize(
    ("theme_base", "primary", "text", "expected"),
    (
        (
            "light",
            "#3366ff",
            "#000000",
            "background-color: rgba(51, 102, 255, 0.18); color: #000000;",
        ),
        (
            "dark",
            "#ff9900",
            None,
            "background-color: rgba(255, 153, 0, 0.45); color: #FAFAFA;",
        ),
    ),
)
def test_resolve_highlight_css_uses_theme_palette(
    theme_base: str, primary: str | None, text: str | None, expected: str
) -> None:
    stub = StubTheme(
        {
            "theme.base": theme_base,
            "theme.primaryColor": primary,
            "theme.textColor": text,
        }
    )

    def fake_get_option(key: str) -> Any:  # pragma: no cover - simple delegator
        return stub.get_option(key)

    result = streamlit_app._resolve_highlight_css(fake_get_option)
    assert result == expected


def test_resolve_highlight_css_handles_invalid_primary() -> None:
    stub = StubTheme(
        {
            "theme.base": "light",
            "theme.primaryColor": "invalid",
            "theme.textColor": None,
        }
    )

    def fake_get_option(key: str) -> Any:  # pragma: no cover - simple delegator
        return stub.get_option(key)

    result = streamlit_app._resolve_highlight_css(fake_get_option)
    assert result == "background-color: #f0f4c3; color: #262730;"
