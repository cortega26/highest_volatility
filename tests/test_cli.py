import pandas as pd

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
        cli, "fetch_fortune_tickers", lambda top_n: fortune.head(top_n)
    )
    monkeypatch.setattr(
        cli,
        "download_price_history",
        lambda tickers, lookback_days, interval="1d", prepost=False: prices,
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
    assert out[2].startswith("A")


def test_cli_rank_by_max_drawdown(monkeypatch, capsys):
    prices, fortune = _mock_data()

    monkeypatch.setattr(
        cli, "fetch_fortune_tickers", lambda top_n: fortune.head(top_n)
    )
    monkeypatch.setattr(
        cli,
        "download_price_history",
        lambda tickers, lookback_days, interval="1d", prepost=False: prices,
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
    assert out[2].startswith("A")

