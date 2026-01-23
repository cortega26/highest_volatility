"""Helpers for expanding environment-driven paths."""

from __future__ import annotations

import os
import re
from pathlib import Path

_PERCENT_VAR_RE = re.compile(r"%([^%]+)%")
_DOLLAR_VAR_RE = re.compile(r"\$(\w+)|\$\{([^}]+)\}")


def _unresolved_env_vars(value: str) -> set[str]:
    if os.name == "nt":
        return {match.group(1) for match in _PERCENT_VAR_RE.finditer(value)}
    unresolved: set[str] = set()
    for match in _DOLLAR_VAR_RE.finditer(value):
        name = match.group(1) or match.group(2)
        if name:
            unresolved.add(name)
    return unresolved


def expand_env_path(raw: str, *, field: str | None = None) -> Path:
    """Expand environment variables in ``raw`` and validate placeholders."""

    if not isinstance(raw, str):
        raise TypeError("Path must be a string.")
    expanded = os.path.expandvars(raw)
    unresolved = _unresolved_env_vars(expanded)
    if unresolved:
        label = f" in {field}" if field else ""
        names = ", ".join(sorted(unresolved))
        plural = "s" if len(unresolved) > 1 else ""
        raise ValueError(f"Unresolved environment variable{plural}{label}: {names}")
    return Path(expanded).expanduser()
