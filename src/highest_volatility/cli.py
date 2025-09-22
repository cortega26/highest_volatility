"""Command-line entrypoints exposed via :mod:`highest_volatility.cli`."""

from __future__ import annotations

from importlib import import_module

_APP_CLI = import_module("highest_volatility.app.cli")

__all__ = getattr(_APP_CLI, "__all__", None)
if __all__ is None:
    __all__ = [name for name in dir(_APP_CLI) if not name.startswith("__")]

for _attr in __all__:
    globals()[_attr] = getattr(_APP_CLI, _attr)

del _attr, _APP_CLI, import_module


if __name__ == "__main__":
    raise SystemExit(globals()["main"]())
