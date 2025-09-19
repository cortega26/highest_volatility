"""Utilities for merging cached and new price data."""

from __future__ import annotations

import pandas as pd


def merge_incremental(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge ``new`` prices into ``existing`` incrementally.

    Rows are concatenated, sorted by index and duplicate indices removed, keeping
    the *last* occurrence (i.e. favouring ``new`` data).
    """

    if existing is None or existing.empty:
        return new.sort_index()
    if new is None or new.empty:
        return existing.sort_index()

    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


import sys as _sys

_sys.modules.setdefault("cache.merge", _sys.modules[__name__])
