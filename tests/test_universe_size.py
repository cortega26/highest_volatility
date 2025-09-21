import builtins
from typing import Iterable

import pandas as pd
import pytest

from highest_volatility.universe import (
    FortuneData,
    _load_cached_fortune,
    _validate_history,
    build_universe,
)


def test_load_cached_fortune_returns_none_when_missing(monkeypatch):
    monkeypatch.setattr(
        "highest_volatility.universe.load_cached_fortune",
        lambda *args, **kwargs: None,
    )

    result = _load_cached_fortune(first_n_fortune=10, ticker_cache_days=1)
    assert result is None


def test_load_cached_fortune_wraps_dataframe(monkeypatch):
    def fake_loader(*_: Iterable, **__: dict) -> pd.DataFrame:
        return pd.DataFrame([{"rank": 1, "company": "A", "ticker": "A"}])

    monkeypatch.setattr(
        "highest_volatility.universe.load_cached_fortune",
        fake_loader,
    )
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    result = _load_cached_fortune(first_n_fortune=5, ticker_cache_days=30)
    assert isinstance(result, FortuneData)
    assert list(result.frame["ticker"]) == ["A"]


@pytest.mark.parametrize(
    "tickers, validate, expected",
    [
        ([], True, set()),
        (["ONE"], False, {"ONE"}),
    ],
)
def test_validate_history_respects_flag(monkeypatch, tickers, validate, expected):
    monkeypatch.setattr(
        "highest_volatility.universe._validate_tickers_have_history",
        lambda tkrs: [t.lower() for t in tkrs],
    )

    result = _validate_history(tickers, validate=validate)
    if validate:
        assert result == {t.lower() for t in tickers}
    else:
        assert result == expected


@pytest.mark.skip("requires selenium and internet access")
def test_universe_has_at_least_300_public_tickers():
    # Live test: requires Chrome, internet access
    tickers, _ = build_universe(first_n_fortune=300)
    assert len(tickers) >= 300, f"Only got {len(tickers)} tickers"
