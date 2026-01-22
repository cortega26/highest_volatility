"""Regression test to enforce pinned dependencies.

Run with:
    python -m pytest tests/test_requirements_pins.py
"""

from pathlib import Path


def test_requirements_are_pinned() -> None:
    requirements = Path("requirements.txt").read_text().splitlines()
    for line in requirements:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "==" in stripped, f"Unpinned requirement: {stripped}"
