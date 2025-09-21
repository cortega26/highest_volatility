from __future__ import annotations

from typing import Iterable

import pandas as pd

from highest_volatility.universe import build_universe


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
