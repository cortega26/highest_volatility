"""Tests for the Streamlit-facing helper utilities."""

from __future__ import annotations

import importlib
import sys
from typing import Iterable

import pandas as pd
import pytest

MODULE_NAME = "highest_volatility.app.ui_helpers"


def _reload_helpers() -> object:
    """Import the helpers module, ensuring a fresh module load."""

    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


@pytest.fixture
def helpers_module() -> object:
    """Provide a freshly imported helpers module for each test."""

    return _reload_helpers()


@pytest.fixture
def metric_registry(helpers_module: object) -> dict[str, object]:
    """Return a temporary metric registry that is restored after the test."""

    from highest_volatility.compute import metrics as metrics_module

    original = metrics_module.METRIC_REGISTRY.copy()
    metrics_module.METRIC_REGISTRY.clear()
    try:
        yield metrics_module.METRIC_REGISTRY
    finally:
        metrics_module.METRIC_REGISTRY.clear()
        metrics_module.METRIC_REGISTRY.update(original)


def test_helpers_import_does_not_require_streamlit() -> None:
    """Importing the helpers should not implicitly import Streamlit."""

    was_loaded = "streamlit" in sys.modules
    module = _reload_helpers()
    assert module is not None
    if not was_loaded:
        assert "streamlit" not in sys.modules


@pytest.mark.parametrize(
    ("metric_key", "values"),
    [
        ("alpha_score", [0.3, 1.2, 0.8]),
        ("beta_ratio", [2.5, 0.4, 1.5]),
    ],
)
def test_prepare_metric_table_sorts_by_metric(
    metric_key: str,
    values: Iterable[float],
    helpers_module: object,
    metric_registry: dict[str, object],
) -> None:
    """The helper should sort rows and label metric columns per key."""

    tickers = ["AAA", "BBB", "CCC"]
    value_list = list(values)

    def _fake_metric(
        prices: pd.DataFrame,
        *,
        tickers: Iterable[str],
        min_periods: int,
        interval: str,
        **_: object,
    ) -> pd.DataFrame:
        del prices, min_periods, interval
        return pd.DataFrame({"ticker": list(tickers), metric_key: value_list})

    metric_registry[metric_key] = _fake_metric

    prices = pd.DataFrame(
        {"Adj Close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    fortune = pd.DataFrame(
        {
            "ticker": tickers,
            "rank": [1, 2, 3],
            "company": ["Alpha", "Beta", "Gamma"],
        }
    )

    table = helpers_module.prepare_metric_table(
        prices,
        metric_key=metric_key,
        min_days=2,
        interval="1d",
        tickers=tickers,
        fortune=fortune,
    )

    assert list(table.columns) == ["ticker", "rank", "company", metric_key]

    expected_pairs = sorted(zip(value_list, tickers), reverse=True)
    assert table[metric_key].tolist() == [value for value, _ in expected_pairs]
    assert table["ticker"].tolist() == [ticker for _, ticker in expected_pairs]
