import importlib
from pathlib import Path

import pandas as pd

from highest_volatility.compute import metrics


def test_metrics_plugin_directory(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "metrics"
    plugin_dir.mkdir()
    plugin_file = plugin_dir / "my_metric.py"
    plugin_file.write_text(
        """
import pandas as pd

def my_metric(prices, **kwargs):
    return pd.DataFrame({'ticker': ['A'], 'my_metric': [1.23]})

METRICS = {'my_metric': my_metric}
"""
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'src'))
    importlib.reload(metrics)
    metrics.load_plugins()
    assert 'my_metric' in metrics.METRIC_REGISTRY


def _sample_price_matrix() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=5)
    close = pd.DataFrame(
        {
            "A": [100, 110, 105, 102, 100],
            "B": [100, 90, 95, 94, 96],
        },
        index=idx,
    )
    return pd.concat({"Adj Close": close}, axis=1)


def test_value_at_risk_metric():
    prices = _sample_price_matrix()
    result = metrics.METRIC_REGISTRY["var"](prices)
    expected = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "var": [-0.042922, -0.086579],
        }
    )
    pd.testing.assert_frame_equal(
        result.sort_values("ticker").reset_index(drop=True).round(6),
        expected,
    )


def test_sortino_metric():
    prices = _sample_price_matrix()
    result = metrics.METRIC_REGISTRY["sortino"](prices)
    expected = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "sortino": [1.925098, -2.113560],
        }
    )
    pd.testing.assert_frame_equal(
        result.sort_values("ticker").reset_index(drop=True).round(6),
        expected,
    )
