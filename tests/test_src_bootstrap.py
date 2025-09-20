"""Tests for bootstrapping the legacy ``src`` namespace."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_highest_volatility_bootstraps_src(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    """Importing :mod:`highest_volatility` makes ``src`` available."""

    cleaned = [
        entry for entry in sys.path if Path(entry).resolve() != repo_root
    ]
    monkeypatch.setattr(sys, "path", cleaned, raising=False)
    monkeypatch.delitem(sys.modules, "highest_volatility", raising=False)
    monkeypatch.delitem(sys.modules, "src", raising=False)

    module = importlib.import_module("highest_volatility")

    assert "src" in sys.modules
    assert any(Path(entry).resolve() == repo_root for entry in sys.path)
    assert module is sys.modules["highest_volatility"]
