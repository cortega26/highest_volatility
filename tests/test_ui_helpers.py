import pandas as pd
import pytest

from highest_volatility.app.ui_helpers import (
    prepare_metric_table,
    sanitize_price_matrix,
)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=6, freq="D")
    data = {
        ("Adj Close", "AAA"): [100, 101, 102, 104, 103, 105],
        ("Adj Close", "BBB"): [50, 49, 48, 47, 46, 45],
    }
    return pd.DataFrame(data, index=idx)


def test_sanitize_price_matrix_filters_short_series(sample_prices: pd.DataFrame) -> None:
    prices = sample_prices.copy()
    prices[("Adj Close", "BBB")] = [50, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]

    filtered, close_only, dropped_short, dropped_duplicate, dropped_empty = (
        sanitize_price_matrix(prices, min_days=3, tickers=["AAA", "BBB"])
    )

    assert list(close_only.columns) == ["AAA"]
    assert dropped_short == ["BBB"]
    assert dropped_duplicate == []
    assert dropped_empty == []
    assert (
        filtered.columns.get_level_values(1).tolist() == ["AAA"]
    ), "Only the surviving ticker should remain"


def test_prepare_metric_table_sorts_and_joins(sample_prices: pd.DataFrame) -> None:
    fortune = pd.DataFrame(
        {
            "rank": [1, 2],
            "company": ["Alpha", "Bravo"],
            "ticker": ["AAA", "BBB"],
        }
    )
    filtered, close_only, *_ = sanitize_price_matrix(
        sample_prices, min_days=3, tickers=["AAA", "BBB"]
    )

    table = prepare_metric_table(
        filtered,
        metric_key="cc_vol",
        min_days=3,
        interval="1d",
        tickers=close_only.columns,
        fortune=fortune,
        close_prices=close_only,
    )

    assert list(table.columns) == ["ticker", "rank", "company", "cc_vol"]
    assert table.shape[0] == 2
    assert table.loc[0, "cc_vol"] >= table.loc[1, "cc_vol"]
    assert table.loc[table["ticker"] == "AAA", "company"].iloc[0] == "Alpha"


def test_prepare_metric_table_handles_empty_prices() -> None:
    table = prepare_metric_table(
        pd.DataFrame(),
        metric_key="cc_vol",
        min_days=5,
        interval="1d",
        tickers=[],
        fortune=None,
    )
    assert table.empty


def test_prepare_metric_table_unknown_metric(sample_prices: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        prepare_metric_table(
            sample_prices,
            metric_key="unknown",
            min_days=3,
            interval="1d",
            tickers=["AAA"],
            fortune=None,
        )
