"""Runtime helpers for the ``src`` namespace package.

The project uses a ``src``-layout repository while the CI pipeline executes
modules directly via ``python -m src.<module>``.  When invoked this way Python
does not automatically add the ``src`` directory itself to ``sys.path``, which
means absolute imports such as ``cache.store`` or ``highest_volatility.errors``
are not resolvable even though they live under ``src/``.  Previously this led
to ``ModuleNotFoundError`` exceptions in the price ingestion workflow on CI.

To make the entrypoints robust in both development (``python -m src.cli``) and
packaged environments we eagerly insert the ``src`` directory into ``sys.path``
when this namespace package is initialised.  The check ensures we avoid
duplicate entries and keeps the change side-effect free for installed wheels
where the path is already present.
"""

from __future__ import annotations

from importlib import import_module
import pkgutil
from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parent
_src_root_str = str(_SRC_ROOT)
if _src_root_str not in sys.path:
    sys.path.insert(0, _src_root_str)


def _alias_namespace(old: str, new: str) -> None:
    """Register ``old`` as an alias of ``new`` within :mod:`sys.modules`."""

    try:
        new_module = import_module(new)
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
            module = import_module(module_info.name)
        except ModuleNotFoundError:
            continue
        alias = f"{old}{module_info.name[prefix_len:]}"
        sys.modules.setdefault(alias, module)


for _old, _new in {
    "src.api": "highest_volatility.app.api",
    "src.cache": "highest_volatility.cache",
    "src.config": "highest_volatility.config",
    "src.datasource": "highest_volatility.datasource",
    "src.ingest": "highest_volatility.ingest",
    "src.pipeline": "highest_volatility.pipeline",
    "src.security": "highest_volatility.security",
}.items():
    _alias_namespace(_old, _new)


# Maintain historical ``cache`` namespace compatibility.
if "highest_volatility.cache" in sys.modules:
    sys.modules.setdefault("cache", sys.modules["highest_volatility.cache"])
    for _suffix in ("merge", "store"):
        try:
            module = import_module(f"highest_volatility.cache.{_suffix}")
        except ModuleNotFoundError:
            continue
        sys.modules.setdefault(f"cache.{_suffix}", module)
