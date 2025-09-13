import pandas as pd
import pytest

from highest_volatility.app import cli


def test_cli_fails_if_universe_too_small(monkeypatch):
    fortune = pd.DataFrame(
        {"rank": [1, 2], "company": ["A", "B"], "ticker": ["A", "B"]}
    )

    def fake_build(_top_n, **__):
        return fortune["ticker"].tolist(), fortune

    monkeypatch.setattr(cli, "build_universe", fake_build)
    monkeypatch.setattr(
        cli,
        "download_price_history",
        lambda tickers, lookback_days, interval="1d", prepost=False, **_: pd.DataFrame(),
    )

    with pytest.raises(SystemExit, match="Universe too small"):
        cli.main(["--top-n", "2", "--print-top", "3"])
