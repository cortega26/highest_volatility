"""Tools for finding the most volatile stocks among the Fortune 100."""

from __future__ import annotations

from importlib import metadata as _metadata

__all__ = ["__version__"]

try:
    __version__ = _metadata.version("highest-volatility")
except _metadata.PackageNotFoundError:  # pragma: no cover - during local dev
    __version__ = "0.0.0"
