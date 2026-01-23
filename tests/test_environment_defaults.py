"""Tests for Windows environment defaults."""

from __future__ import annotations

import os

import pytest

from highest_volatility.config.environment import ensure_windows_environment


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific environment defaults")
def test_ensure_windows_environment_sets_defaults(monkeypatch):
    monkeypatch.delenv("SystemDrive", raising=False)
    monkeypatch.delenv("ProgramData", raising=False)
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)

    ensure_windows_environment()

    assert os.getenv("SystemDrive")
    assert os.getenv("ProgramData")
    assert os.getenv("ProgramFiles")
    assert os.getenv("ProgramFiles(x86)")
