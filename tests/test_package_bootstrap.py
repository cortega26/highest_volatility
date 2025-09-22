"""Tests covering the canonical :mod:`highest_volatility` package bootstrap."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture()
def cleanup_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "highest_volatility",
        "highest_volatility.cache",
        "highest_volatility.cli",
        "src",
        "src.cache",
        "src.cli",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_import_exposes_version(cleanup_modules: None) -> None:
    """Importing :mod:`highest_volatility` provides package metadata."""

    module = importlib.import_module("highest_volatility")

    assert module is sys.modules["highest_volatility"]
    assert isinstance(module.__version__, str)
    assert module.__version__


def test_legacy_src_namespace_removed(cleanup_modules: None) -> None:
    """The retired ``src`` namespace is no longer importable."""

    importlib.import_module("highest_volatility")

    assert "src" not in sys.modules
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.cache")
