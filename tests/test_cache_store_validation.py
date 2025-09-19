from __future__ import annotations

import pandas as pd
import pytest

from src.cache.store import save_cache


def test_save_cache_rejects_invalid_ticker(tmp_path, monkeypatch):
    monkeypatch.setattr("cache.store.CACHE_ROOT", tmp_path)
    df = pd.DataFrame({"Close": [1.0]}, index=pd.date_range("2024-01-01", periods=1))
    with pytest.raises(ValueError):
        save_cache("BAD/TICKER", "1d", df, "test", validate=False)
