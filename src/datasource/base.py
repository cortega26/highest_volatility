"""DataSource protocol for price fetching."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class DataSource(Protocol):
    """Abstract data source for price history."""

    def get_prices(self, ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
        """Return price history for ``ticker``.

        Parameters
        ----------
        ticker:
            Symbol to query.
        start, end:
            Date range to retrieve.
        interval:
            Bar size supported by the underlying source (e.g. ``"1d"``).
        """

    def validate_ticker(self, ticker: str) -> bool:
        """Return ``True`` if ``ticker`` is recognised by the source."""
