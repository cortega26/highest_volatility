"""CSV storage helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save ``df`` to ``path`` without the index."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
