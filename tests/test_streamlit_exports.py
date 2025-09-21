"""Regression tests for Streamlit CSV export helpers."""

from __future__ import annotations

import pandas as pd

from highest_volatility.app.streamlit_app import _escape_spreadsheet_formula


def test_escape_spreadsheet_formula_neutralises_dangerous_prefixes() -> None:
    """Values beginning with control characters should be prefixed."""

    df = pd.DataFrame(
        {
            "a": [
                "=SUM(A1:A2)",
                "+1",
                "-2",
                "@eval('1')",
                "\tTabbed",
                "\nNewline",
                "Safe",
            ],
            "b": [
                "Normal",
                123,
                0.0,
                None,
                "=HYPERLINK('http://example.com')",
                "-Leading",
                "Ready",
            ],
        }
    )

    escaped = df.map(_escape_spreadsheet_formula)
    dangerous_prefixes = ("=", "+", "-", "@", "\t", "\n")

    for value in escaped.stack():
        if isinstance(value, str) and value:
            assert not value.startswith(dangerous_prefixes)


def test_escape_spreadsheet_formula_preserves_non_strings() -> None:
    """Numeric and null values should remain untouched."""

    df = pd.DataFrame({"a": [1, None, 3.5]})

    escaped = df.map(_escape_spreadsheet_formula)

    pd.testing.assert_frame_equal(df, escaped)
