import pytest

pytest.skip("requires selenium and internet access", allow_module_level=True)

import os

from highest_volatility.universe import build_universe


def test_universe_has_at_least_300_public_tickers():
    # Live test: requires Chrome, internet access
    tickers, fortune = build_universe(first_n_fortune=300)
    assert len(tickers) >= 300, f"Only got {len(tickers)} tickers"

