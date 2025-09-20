"""Tests for security validation helpers."""

import pytest

from src.security.validation import SanitizationError, sanitize_interval


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("5wk", "5wk"),
        ("3WK", "3wk"),
        ("2mo", "2mo"),
    ],
)
def test_sanitize_interval_accepts_extended_suffixes(raw: str, expected: str) -> None:
    """Intervals with supported suffixes should be normalized."""

    assert sanitize_interval(raw) == expected


@pytest.mark.parametrize("raw", ["10yr", "15", "7 weeks"])
def test_sanitize_interval_rejects_invalid_suffixes(raw: str) -> None:
    """Unsupported interval suffixes must raise a sanitization error."""

    with pytest.raises(SanitizationError):
        sanitize_interval(raw)
