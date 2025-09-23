import pandas as pd
import pytest

from highest_volatility.app.ui_helpers import (
    SanitizedPrices,
    prepare_metric_table,
    sanitize_price_matrix,
)
from highest_volatility.compute.metrics import metric_display_name


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

    result = sanitize_price_matrix(prices, min_days=3, tickers=["AAA", "BBB"])

    assert isinstance(result, SanitizedPrices)
    assert result.available_tickers == ["AAA"]
    assert result.dropped_short == ["BBB"]
    assert result.dropped_duplicate == []
    assert result.dropped_empty == []
    assert (
        result.filtered.columns.get_level_values(1).tolist() == ["AAA"]
    ), "Only the surviving ticker should remain"


def test_prepare_metric_table_sorts_and_joins(sample_prices: pd.DataFrame) -> None:
    fortune = pd.DataFrame(
        {
            "rank": [1, 2],
            "company": ["Alpha", "Bravo"],
            "ticker": ["AAA", "BBB"],
        }
    )
    result = sanitize_price_matrix(sample_prices, min_days=3, tickers=["AAA", "BBB"])

    table = prepare_metric_table(
        result.filtered,
        metric_key="cc_vol",
        min_days=3,
        interval="1d",
        tickers=result.available_tickers,
        fortune=fortune,
        close_prices=result.close_only,
    )

    assert list(table.columns) == [
        "ticker",
        "rank",
        "company",
        metric_display_name("cc_vol"),
        metric_display_name("ewma_vol"),
        metric_display_name("mad_vol"),
        metric_display_name("sharpe_ratio"),
        metric_display_name("max_drawdown"),
        metric_display_name("var"),
        metric_display_name("sortino"),
    ]
    assert table.shape[0] == 2
    metric_label = metric_display_name("cc_vol")
    assert table.loc[0, metric_label] >= table.loc[1, metric_label]
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


def test_prepare_metric_table_additional_columns_with_ohlc() -> None:
    idx = pd.date_range("2022-01-01", periods=6, freq="D")
    data = {
        ("Open", "AAA"): [100, 101, 102, 103, 104, 105],
        ("High", "AAA"): [101, 102, 103, 104, 105, 106],
        ("Low", "AAA"): [99, 100, 101, 102, 103, 104],
        ("Close", "AAA"): [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
        ("Adj Close", "AAA"): [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
        ("Open", "BBB"): [50, 51, 52, 53, 54, 55],
        ("High", "BBB"): [51, 52, 53, 54, 55, 56],
        ("Low", "BBB"): [49, 50, 51, 52, 53, 54],
        ("Close", "BBB"): [50.5, 51.5, 52.5, 53.5, 54.5, 55.5],
        ("Adj Close", "BBB"): [50.5, 51.5, 52.5, 53.5, 54.5, 55.5],
    }
    prices = pd.DataFrame(data, index=idx)

    sanitized = sanitize_price_matrix(prices, min_days=4, tickers=["AAA", "BBB"])

    table = prepare_metric_table(
        sanitized.filtered,
        metric_key="cc_vol",
        min_days=4,
        interval="1d",
        tickers=sanitized.available_tickers,
        fortune=None,
        close_prices=sanitized.close_only,
    )

    expected_columns = {
        "ticker",
        metric_display_name("cc_vol"),
        metric_display_name("parkinson_vol"),
        metric_display_name("gk_vol"),
        metric_display_name("rs_vol"),
        metric_display_name("yz_vol"),
        metric_display_name("ewma_vol"),
        metric_display_name("mad_vol"),
        metric_display_name("sharpe_ratio"),
        metric_display_name("max_drawdown"),
        metric_display_name("var"),
        metric_display_name("sortino"),
    }
    assert expected_columns.issubset(set(table.columns))
    assert table.shape[0] == 2


def test_sanitized_prices_summarize_drops() -> None:
    frame = pd.DataFrame({"Adj Close": [1, 2, 3]})
    result = SanitizedPrices(
        filtered=frame,
        close_only=frame,
        dropped_short=["AAA", "BBB"],
        dropped_duplicate=["CCC"],
        dropped_empty=["DDD", "EEE"],
    )

    messages = result.summarize_drops(min_days=15)

    assert messages[0] == "2 tickers lacked the required 15 observations."
    assert "duplicate price series" in messages[1]
    assert messages[-1].startswith("2 tickers contained")
