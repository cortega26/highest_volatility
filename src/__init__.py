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

from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parent
_src_root_str = str(_SRC_ROOT)
if _src_root_str not in sys.path:
    sys.path.insert(0, _src_root_str)
