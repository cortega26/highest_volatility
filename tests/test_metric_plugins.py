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
