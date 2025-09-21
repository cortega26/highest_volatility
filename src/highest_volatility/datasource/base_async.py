"""Asynchronous DataSource protocol for price fetching."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class AsyncDataSource(Protocol):
    """Abstract async data source for price history."""

    async def get_prices(self, ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
        """Return price history for ``ticker`` asynchronously."""

    async def validate_ticker(self, ticker: str) -> bool:
        """Return ``True`` if ``ticker`` is recognised by the source."""
