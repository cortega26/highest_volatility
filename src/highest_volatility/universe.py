from __future__ import annotations

"""Utilities for building the ticker universe.

This module fetches the Fortune list, normalises tickers and ensures the
resulting universe is deduplicated.  It logs the final size so callers can
verify that enough data was retrieved before continuing with expensive
processing.
"""

import logging
from typing import Tuple

import pandas as pd

from .ingest.tickers import fetch_fortune_tickers

logger = logging.getLogger(__name__)


def build_universe(first_n_fortune: int) -> Tuple[list[str], pd.DataFrame]:
    """Return a deduplicated list of tickers and the Fortune table.

    Parameters
    ----------
    first_n_fortune:
        Number of Fortune 500 entries to fetch.

    Returns
    -------
    list[str]
        Ticker symbols in the order they appeared.
    DataFrame
        Original Fortune table for the requested slice.
    """

    table = fetch_fortune_tickers(top_n=first_n_fortune)
    tickers = [t for t in table["ticker"].tolist() if t]
    seen = set()
    deduped: list[str] = []
    for t in tickers:
        if t in seen:
            continue
        deduped.append(t)
        seen.add(t)

    logger.info("Universe size after fetch/dedupe: %d", len(deduped))
    return deduped, table
