import importlib

import pandas as pd
import pytest

from highest_volatility.compute import metrics


def _sample_prices() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {"A": [100, 110, 105, 120], "B": [50, 55, 60, 58]}, index=idx
    )


def test_metrics_expected_values():
    prices = _sample_prices()

    # daily returns
    expected_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03", "2020-01-04", "2020-01-04"]
            ),
            "ticker": ["A", "B", "A", "B", "A", "B"],
            "daily_return": [
                0.1,
                0.1,
                -0.045455,
                0.090909,
                0.142857,
                -0.033333,
            ],
        }
    )
    result = metrics.daily_returns(prices)
    pd.testing.assert_frame_equal(result, expected_returns)

    # annualized volatility
    expected_vol = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "annualized_volatility": [1.505921, 1.148106],
        }
    )
    result_vol = metrics.annualized_volatility(prices).round(6)
    pd.testing.assert_frame_equal(result_vol, expected_vol)

    # maximum drawdown
    expected_mdd = pd.DataFrame(
        {"ticker": ["A", "B"], "max_drawdown": [-0.045455, -0.033333]}
    )
    result_mdd = metrics.max_drawdown(prices).round(6)
    pd.testing.assert_frame_equal(result_mdd, expected_mdd)

    # rolling volatility
    expected_roll = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-03", "2020-01-03", "2020-01-04", "2020-01-04"]
            ),
            "ticker": ["A", "B", "A", "B"],
            "window": [2, 2, 2, 2],
            "rolling_volatility": [1.59204, 0.093154, 2.021072, 1.357244],
        }
    )
    result_roll = metrics.rolling_volatility(prices, windows=(2,)).round(6)
    pd.testing.assert_frame_equal(result_roll, expected_roll)

    # Sharpe ratio
    expected_sharpe = pd.DataFrame(
        {"ticker": ["A", "B"], "sharpe_ratio": [10.5827, 11.192938]}
    )
    result_sharpe = metrics.sharpe_ratio(prices).round(6)
    pd.testing.assert_frame_equal(result_sharpe, expected_sharpe)


def test_compute_volatility_guardrail():
    module = importlib.import_module("highest_volatility.compute")
    with pytest.raises(AttributeError) as excinfo:
        getattr(module, "volatility")
    assert "highest_volatility.compute.metrics" in str(excinfo.value)


try:  # pragma: no cover - optional dependency
    from hypothesis import given, strategies as st
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    st = None
    def given(*_args, **_kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator


if st is not None:  # pragma: no branch - optional dependency

    @given(
        st.lists(
            st.floats(
                min_value=1, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=10,
        )
    )
    def test_daily_returns_scale_invariance(values):
        idx = pd.date_range("2021-01-01", periods=len(values), freq="D")
        df = pd.DataFrame({"A": values}, index=idx)
        scaled = df * 7.5

        res1 = metrics.daily_returns(df)
        res2 = metrics.daily_returns(scaled)

        pd.testing.assert_series_equal(
            res1["daily_return"], res2["daily_return"], check_names=False
        )

else:  # pragma: no cover - optional dependency

    @pytest.mark.skip("hypothesis not installed")
    def test_daily_returns_scale_invariance():
        pass

