"""Tools for finding the most volatile stocks among the Fortune 100."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Mapping

LEGACY_NAMESPACE_ALIASES: Mapping[str, str] = {
    "src.cache": "highest_volatility.cache",
    "src.config": "highest_volatility.config",
    "src.datasource": "highest_volatility.datasource",
    "src.ingest": "highest_volatility.ingest",
    "src.pipeline": "highest_volatility.pipeline",
    "src.security": "highest_volatility.security",
    "src.api": "highest_volatility.api",
    "src.cli": "highest_volatility.cli",
}

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
        src_module = importlib.import_module("src")
    except ModuleNotFoundError:
        src_module = _bootstrap_src_namespace()
    else:
        _register_src_aliases(src_module)
        return

    if src_module is not None:
        _register_src_aliases(src_module)


def _bootstrap_src_namespace() -> ModuleType:
    """Create or load the legacy ``src`` namespace."""

    package_root = Path(__file__).resolve().parents[2]
    candidate = package_root / "src"
    if candidate.exists():
        root_str = str(package_root)
        if root_str not in _sys_path_iter():
            sys.path.insert(0, root_str)
        return importlib.import_module("src")

    module = ModuleType("src")
    module.__path__ = []  # type: ignore[attr-defined]
    module.__package__ = "src"
    existing = sys.modules.setdefault("src", module)
    if existing is module:
        _register_cache_namespace_aliases()
    return existing


def _register_src_aliases(module: ModuleType) -> None:
    """Populate ``sys.modules`` with legacy aliases for :mod:`highest_volatility`."""

    if getattr(module, "_highest_volatility_aliases", False):
        return

    alias_helper = getattr(module, "_alias_namespace", None)
    if alias_helper is None:
        alias_helper = _alias_namespace
        setattr(module, "_alias_namespace", alias_helper)

    for legacy, target in LEGACY_NAMESPACE_ALIASES.items():
        alias_helper(legacy, target)

    setattr(module, "_highest_volatility_aliases", True)


def _alias_namespace(old: str, new: str) -> None:
    """Register ``old`` as an alias of ``new`` within :mod:`sys.modules`."""

    try:
        new_module = importlib.import_module(new)
    except ModuleNotFoundError:
        return

    sys.modules.setdefault(old, new_module)

    new_path = getattr(new_module, "__path__", None)
    if not new_path:
        return

    prefix = f"{new}."
    prefix_len = len(new)
    for module_info in pkgutil.walk_packages(new_path, prefix):
        try:
            module = importlib.import_module(module_info.name)
        except ModuleNotFoundError:
            continue
        alias = f"{old}{module_info.name[prefix_len:]}"
        sys.modules.setdefault(alias, module)


def _register_cache_namespace_aliases() -> None:
    """Expose historic ``cache`` modules when running from a wheel install."""

    try:
        cache_module = importlib.import_module("highest_volatility.cache")
    except ModuleNotFoundError:
        return

    sys.modules.setdefault("cache", cache_module)

    for suffix in ("merge", "store"):
        try:
            module = importlib.import_module(f"highest_volatility.cache.{suffix}")
        except ModuleNotFoundError:
            continue
        sys.modules.setdefault(f"cache.{suffix}", module)


def _sys_path_iter() -> Iterable[str]:
    """Yield the current entries in ``sys.path``.

    A tiny wrapper is provided to keep the bootstrap helper testable without
    mutating ``sys.path`` during import-time evaluation.
    """

    return tuple(sys.path)


_ensure_src_namespace()
