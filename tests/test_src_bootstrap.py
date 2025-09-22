"""Tests for bootstrapping the legacy ``src`` namespace."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

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


def test_highest_volatility_bootstraps_without_repo(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    """Import succeeds when the repository ``src`` tree is unavailable."""

    cleaned = [
        entry for entry in sys.path if Path(entry).resolve() != repo_root
    ]
    monkeypatch.setattr(sys, "path", cleaned, raising=False)
    monkeypatch.delitem(sys.modules, "highest_volatility", raising=False)
    monkeypatch.delitem(sys.modules, "src", raising=False)

    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:  # type: ignore[override]
        if self == repo_root / "src":
            return False
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists, raising=False)

    module = importlib.import_module("highest_volatility")

    src_module = sys.modules.get("src")
    assert isinstance(src_module, ModuleType)
    assert getattr(src_module, "__path__", []) == []
    assert "src.cache" in sys.modules
    assert module is sys.modules["highest_volatility"]
