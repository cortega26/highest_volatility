"""SQLite storage helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def save_sqlite(df: pd.DataFrame, path: Path, *, table: str = "volatility") -> None:
    """Persist ``df`` to a SQLite database at ``path``.

    Parameters
    ----------
    df:
        Data to store.
    path:
        Path to the SQLite database file. Will be created if missing.
    table:
        Table name within the database.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)
