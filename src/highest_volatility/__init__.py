"""Tools for finding the most volatile stocks among the Fortune 100."""

from __future__ import annotations

from pathlib import Path
import importlib
import sys
from typing import Iterable

__all__: list[str] = []


def _ensure_src_namespace() -> None:
    """Ensure the legacy ``src`` namespace is importable.

    Historically this project relied on ``import src.*`` even when running
    modules that live inside the :mod:`highest_volatility` package. When these
    modules are executed as scripts (e.g. ``streamlit run`` on a file inside the
    package) Python's module search path might only include the package
    directory itself, so ``import src`` fails with ``ModuleNotFoundError``.

    We attempt to import ``src`` first. If that fails we prepend the repository
    root (two levels up from this file) to ``sys.path`` and import the package
    from there. The helper is idempotent and safe to call multiple times.
    """

    if "src" in sys.modules:
        return

    try:
        importlib.import_module("src")
        return
    except ModuleNotFoundError:
        pass

    package_root = Path(__file__).resolve().parents[2]
    candidate = package_root / "src"
    if not candidate.exists():
        # Nothing we can do—bubble up the original error for visibility.
        raise ModuleNotFoundError(
            "The 'src' package is not available and could not be bootstrapped."
        ) from None

    root_str = str(package_root)
    if root_str not in _sys_path_iter():
        sys.path.insert(0, root_str)

    importlib.import_module("src")


def _sys_path_iter() -> Iterable[str]:
    """Yield the current entries in ``sys.path``.

    A tiny wrapper is provided to keep the bootstrap helper testable without
    mutating ``sys.path`` during import-time evaluation.
    """

    return tuple(sys.path)


_ensure_src_namespace()
