"""Compatibility shim for the legacy :mod:`src.cli` module."""

from __future__ import annotations

from highest_volatility.app.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
