"""Cache utilities bridging the ``src.cache`` and legacy ``cache`` namespaces."""

from __future__ import annotations

import sys

_module = sys.modules[__name__]
sys.modules.setdefault("cache", _module)

__all__: list[str] = []
