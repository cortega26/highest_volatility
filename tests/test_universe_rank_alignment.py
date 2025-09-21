
from typing import Iterable

import pandas as pd
import pytest

from highest_volatility.universe import (
    FortuneData,
    _align_ranks,
    _deduplicate_symbols,
    _normalize_fortune,
    build_universe,
)


def test_cached_rank_alignment_uses_normalized_index(monkeypatch):
    cached = pd.DataFrame(
        [
            {"rank": 7, "company": "Berkshire Hathaway", "ticker": "BRK.B"},
            {"rank": 11, "company": "Apple", "ticker": "AAPL"},
        ]
    )

    def fake_loader(*_: Iterable, **__: dict) -> pd.DataFrame:
        return cached

    monkeypatch.setattr(
        "highest_volatility.universe.load_cached_fortune",
        fake_loader,
    )

    tickers, fortune = build_universe(
        1, validate=False, use_ticker_cache=True, ticker_cache_days=365
    )

    assert tickers == ["BRK-B"]
    assert list(fortune["ticker"]) == ["BRK-B"]
    assert list(fortune["rank"]) == [7]


def test_normalize_fortune_sorts_and_normalizes():
    data = FortuneData(
        pd.DataFrame(
            [
                {"rank": 5, "company": "Beta", "ticker": " beta "},
                {"rank": 1, "company": "Alpha", "ticker": "A"},
            ]
        )
    )

    normalized = _normalize_fortune(data)

    assert list(normalized.frame["rank"]) == [1, 5]
    assert list(normalized.frame["ticker"]) == ["A", "beta"]
    assert list(normalized.frame["normalized_ticker"]) == ["A", "BETA"]


@pytest.mark.parametrize(
    "rows, expected_companies, expected_tickers",
    [
        (
            [
                {"company": "Valid", "ticker": "GOOD", "normalized_ticker": "GOOD"},
                {"company": "Valid", "ticker": "GOOD", "normalized_ticker": "GOOD"},
                {"company": "Bad", "ticker": "bad", "normalized_ticker": "bad"},
                {"company": "Other", "ticker": "OT-H1", "normalized_ticker": "OT-H1"},
            ],
            ["Valid", "Other"],
            ["GOOD", "OT-H1"],
        ),
        (
            [
                {"company": "Also", "ticker": "ALSO", "normalized_ticker": "ALSO"},
                {"company": "Extra", "ticker": "EXTRA/", "normalized_ticker": "EXTRA/"},
            ],
            ["Also"],
            ["ALSO"],
        ),
    ],
)
def test_deduplicate_symbols_filters_invalid(rows, expected_companies, expected_tickers):
    fortune = FortuneData(pd.DataFrame(rows))

    companies, tickers = _deduplicate_symbols(fortune)

    assert companies == expected_companies
    assert tickers == expected_tickers


def test_align_ranks_falls_back_to_enumeration_when_lookup_fails():
    data = FortuneData(
        pd.DataFrame(
            [
                {"rank": 3, "company": "Gamma", "ticker": "G", "normalized_ticker": "G"},
            ]
        )
    )

    fortune = _align_ranks(
        data,
        companies=["Missing"],
        tickers=["MISSING"],
    )

    assert list(fortune["rank"]) == [1]
    assert list(fortune["company"]) == ["Missing"]
    assert list(fortune["ticker"]) == ["MISSING"]


def test_align_ranks_uses_existing_ranks_when_available():
    frame = pd.DataFrame(
        [
            {
                "rank": 2,
                "company": "Alpha",
                "ticker": "A",
                "normalized_ticker": "A",
            },
            {
                "rank": 4,
                "company": "Beta",
                "ticker": "B",
                "normalized_ticker": "B",
            },
        ]
    )
    data = FortuneData(frame)

    fortune = _align_ranks(data, companies=["Beta"], tickers=["B"])

    assert list(fortune["rank"]) == [4]
    assert list(fortune["company"]) == ["Beta"]
    assert list(fortune["ticker"]) == ["B"]
