import pandas as pd
import re

from highest_volatility.app import cli


def _mock_data():
    idx = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame(
        {("Adj Close", "A"): [100, 110, 105, 120], ("Adj Close", "B"): [100, 90, 80, 70]},
        index=idx,
    )
    fortune = pd.DataFrame(
        {"rank": [1, 2], "company": ["A Co", "B Co"], "ticker": ["A", "B"]}
    )
    return prices, fortune


def test_cli_rank_by_sharpe_ratio(monkeypatch, capsys):
    prices, fortune = _mock_data()

    monkeypatch.setattr(
        cli,
        "build_universe",
        lambda top_n, **__: (fortune["ticker"].head(top_n).tolist(), fortune.head(top_n)),
    )
    monkeypatch.setattr(
        cli,
        "download_price_history",
        lambda tickers, lookback_days, interval="1d", prepost=False, **_: prices,
    )

    cli.main([
        "--metric",
        "sharpe_ratio",
        "--top-n",
        "2",
        "--print-top",
        "2",
        "--min-days",
        "2",
    ])
    out = capsys.readouterr().out.strip().splitlines()
    data_lines = [ln for ln in out if re.match(r"^[A-Z]", ln)]
    assert data_lines[0].startswith("A")


def test_cli_rank_by_max_drawdown(monkeypatch, capsys):
    prices, fortune = _mock_data()

    monkeypatch.setattr(
        cli,
        "build_universe",
        lambda top_n, **__: (fortune["ticker"].head(top_n).tolist(), fortune.head(top_n)),
    )
    monkeypatch.setattr(
        cli,
        "download_price_history",
        lambda tickers, lookback_days, interval="1d", prepost=False, **_: prices,
    )

    cli.main([
        "--metric",
        "max_drawdown",
        "--top-n",
        "2",
        "--print-top",
        "2",
        "--min-days",
        "2",
    ])
    out = capsys.readouterr().out.strip().splitlines()
    data_lines = [ln for ln in out if re.match(r"^[A-Z]", ln)]
    assert data_lines[0].startswith("A")

