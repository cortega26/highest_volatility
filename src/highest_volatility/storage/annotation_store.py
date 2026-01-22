"""SQLite-backed storage for ticker annotations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AnnotationRecord:
    ticker: str
    note: str
    updated_at: datetime
    client_timestamp: datetime


def _ensure_tz(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _serialize_ts(value: datetime) -> str:
    return _ensure_tz(value).isoformat()


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


class AnnotationStore:
    """Persist annotations and their history in SQLite."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if self._path.exists() and self._path.is_dir():
            raise ValueError("Annotation store path must be a file.")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def list_annotations(self) -> list[AnnotationRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ticker, note, updated_at, client_timestamp
                FROM annotations
                ORDER BY ticker
                """
            ).fetchall()
        return [
            AnnotationRecord(
                ticker=row["ticker"],
                note=row["note"],
                updated_at=_parse_ts(row["updated_at"]),
                client_timestamp=_parse_ts(row["client_timestamp"]),
            )
            for row in rows
        ]

    def load_history(self, ticker: str) -> list[AnnotationRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ticker, note, updated_at, client_timestamp
                FROM annotation_history
                WHERE ticker = ?
                ORDER BY id
                """,
                (ticker,),
            ).fetchall()
        return [
            AnnotationRecord(
                ticker=row["ticker"],
                note=row["note"],
                updated_at=_parse_ts(row["updated_at"]),
                client_timestamp=_parse_ts(row["client_timestamp"]),
            )
            for row in rows
        ]

    def upsert(
        self,
        *,
        ticker: str,
        note: str,
        updated_at: datetime,
        client_timestamp: datetime,
    ) -> AnnotationRecord:
        record = AnnotationRecord(
            ticker=ticker,
            note=note,
            updated_at=_ensure_tz(updated_at),
            client_timestamp=_ensure_tz(client_timestamp),
        )
        updated_at_value = _serialize_ts(record.updated_at)
        client_timestamp_value = _serialize_ts(record.client_timestamp)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO annotations (ticker, note, updated_at, client_timestamp)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    note = excluded.note,
                    updated_at = excluded.updated_at,
                    client_timestamp = excluded.client_timestamp
                """,
                (ticker, note, updated_at_value, client_timestamp_value),
            )
            conn.execute(
                """
                INSERT INTO annotation_history (ticker, note, updated_at, client_timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (ticker, note, updated_at_value, client_timestamp_value),
            )
        return record

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        statements: Iterable[str] = (
            """
            CREATE TABLE IF NOT EXISTS annotations (
                ticker TEXT PRIMARY KEY,
                note TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                client_timestamp TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS annotation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                note TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                client_timestamp TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_annotation_history_ticker
            ON annotation_history (ticker)
            """,
        )
        with self._connect() as conn:
            for statement in statements:
                conn.execute(statement)
