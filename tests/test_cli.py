import argparse
import re
from argparse import Namespace

import pandas as pd
import pytest

from highest_volatility import cli as public_cli
from highest_volatility.app import cli


def _mock_data():
    idx = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame(
        {
            ("Adj Close", "A"): [100, 110, 105, 120],
            ("Adj Close", "B"): [100, 90, 80, 70],
        },
        index=idx,
    )
    fortune = pd.DataFrame(
        {"rank": [1, 2], "company": ["A Co", "B Co"], "ticker": ["A", "B"]}
    )
    return prices, fortune


def _make_args(**overrides) -> Namespace:
    base = vars(cli.parse_args([]))
    base.update(overrides)
    return Namespace(**base)


def test_public_cli_reexports_parser():
    parser = public_cli.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.format_usage() == cli.build_parser().format_usage()
    assert public_cli.build_parser is cli.build_parser


def test_public_cli_main_forwards(monkeypatch):
    assert public_cli.main is cli.main


def test_build_universe_step_returns_dataclass(monkeypatch):
    fortune = pd.DataFrame({"ticker": ["A"], "company": ["A Co"], "rank": [1]})
    monkeypatch.setattr(cli, "build_universe", lambda *_, **__: (["A"], fortune))
    args = _make_args(top_n=1)

    result = cli._build_universe_step(args)

    assert result.tickers == ["A"]
    assert result.fortune.equals(fortune)
    assert result.duration >= 0


def test_download_prices_step_sanitizes(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=5)
    raw_prices = pd.DataFrame(
        {
            ("Adj Close", "A"): [1, 2, 3, 4, 5],
            ("Adj Close", "B"): [1, 2, 3, 4, 5],
            ("Adj Close", "C"): [1, None, None, None, None],
        },
        index=idx,
    )

    monkeypatch.setattr(
        cli,
        "download_price_history",
        lambda *_, **__: raw_prices,
    )

    args = _make_args(min_days=3)
    result = cli._download_prices_step(args, ["A", "B", "C"])

    assert result.tickers == ["A"]
    assert result.close.columns.tolist() == ["A"]
    assert "C" in result.dropped_short
    assert result.duration >= 0


def test_main_handles_empty_price_download(monkeypatch, capsys):
    fortune = pd.DataFrame({"rank": [1], "company": ["A Co"], "ticker": ["A"]})

    monkeypatch.setattr(
        cli,
        "build_universe",
        lambda top_n, **__: (
            fortune["ticker"].head(top_n).tolist(),
            fortune.head(top_n),
        ),
    )
    monkeypatch.setattr(cli, "download_price_history", lambda *_, **__: pd.DataFrame())

    cli.main(["--top-n", "1", "--print-top", "1", "--min-days", "2"])

    out = capsys.readouterr().out
    assert "Empty DataFrame" in out


def test_compute_metrics_step_uses_registry(monkeypatch):
    args = _make_args(metric="cc_vol", min_days=2)
    prices = pd.DataFrame(
        {("Adj Close", "A"): [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3)
    )
    close = prices["Adj Close"]
    download_result = cli.DownloadPricesResult(
        prices=prices,
        close=close,
        tickers=["A"],
        dropped_short=[],
        dropped_duplicate=[],
        duration=0.0,
    )
    fortune = pd.DataFrame({"ticker": ["A"], "company": ["A Co"], "rank": [1]})

    def fake_metric(*_, **__):
        return pd.DataFrame({"ticker": ["A"], "cc_vol": [0.5]})

    monkeypatch.setitem(cli.METRIC_REGISTRY, "cc_vol", fake_metric)

    result = cli._compute_metrics_step(args, download_result, fortune)

    assert pytest.approx(result.result.loc["A", "cc_vol"]) == 0.5
    assert result.duration >= 0


def test_render_output_step_exports(monkeypatch, tmp_path, capsys):
    args = _make_args(print_top=1, output_csv=tmp_path / "out.csv")
    df = pd.DataFrame({"rank": [1], "company": ["A"], "cc_vol": [0.1]}, index=["A"])
    compute_result = cli.ComputeMetricsResult(result=df, duration=0.0)

    monkeypatch.setattr(
        cli, "save_sqlite", lambda *_, **__: pytest.fail("should not be called")
    )

    result = cli._render_output_step(args, compute_result)

    captured = capsys.readouterr().out
    assert "cc_vol" in captured
    assert (tmp_path / "out.csv").exists()
    assert result.duration >= 0


def test_cli_rank_by_sharpe_ratio(monkeypatch, capsys):
    prices, fortune = _mock_data()

    monkeypatch.setattr(
        cli,
        "build_universe",
        lambda top_n, **__: (
            fortune["ticker"].head(top_n).tolist(),
            fortune.head(top_n),
        ),
    )
    monkeypatch.setattr(
        cli,
        "download_price_history",
        lambda tickers, lookback_days, interval="1d", prepost=False, **_: prices,
    )

    cli.main(
        [
            "--metric",
            "sharpe_ratio",
            "--top-n",
            "2",
            "--print-top",
            "2",
            "--min-days",
            "2",
        ]
    )
    out = capsys.readouterr().out.strip().splitlines()
    data_lines = [ln for ln in out if re.match(r"^[A-Z]", ln)]
    assert data_lines[0].startswith("A")


def test_cli_rank_by_max_drawdown(monkeypatch, capsys):
    prices, fortune = _mock_data()

    monkeypatch.setattr(
        cli,
        "build_universe",
        lambda top_n, **__: (
            fortune["ticker"].head(top_n).tolist(),
            fortune.head(top_n),
        ),
    )
    monkeypatch.setattr(
        cli,
        "download_price_history",
        lambda tickers, lookback_days, interval="1d", prepost=False, **_: prices,
    )

    cli.main(
        [
            "--metric",
            "max_drawdown",
            "--top-n",
            "2",
            "--print-top",
            "2",
            "--min-days",
            "2",
        ]
    )
    out = capsys.readouterr().out.strip().splitlines()
    data_lines = [ln for ln in out if re.match(r"^[A-Z]", ln)]
    assert data_lines[0].startswith("A")


def test_cli_async_flag(monkeypatch):
    prices, fortune = _mock_data()
    called = {}

    def _mock_download(*args, **kwargs):
        called["matrix_mode"] = kwargs.get("matrix_mode")
        return prices

    monkeypatch.setattr(
        cli,
        "build_universe",
        lambda top_n, **__: (
            fortune["ticker"].head(top_n).tolist(),
            fortune.head(top_n),
        ),
    )
    monkeypatch.setattr(cli, "download_price_history", _mock_download)

    cli.main(
        [
            "--metric",
            "sharpe_ratio",
            "--top-n",
            "2",
            "--print-top",
            "2",
            "--min-days",
            "2",
            "--async-fetch",
        ]
    )
    assert called.get("matrix_mode") == "async"


def test_cli_passes_max_retries(monkeypatch):
    prices, fortune = _mock_data()
    called = {}

    def _mock_download(*args, **kwargs):
        called["max_retries"] = kwargs.get("max_retries")
        return prices

    monkeypatch.setattr(
        cli,
        "build_universe",
        lambda top_n, **__: (
            fortune["ticker"].head(top_n).tolist(),
            fortune.head(top_n),
        ),
    )
    monkeypatch.setattr(cli, "download_price_history", _mock_download)

    cli.main(
        [
            "--metric",
            "sharpe_ratio",
            "--top-n",
            "2",
            "--print-top",
            "2",
            "--min-days",
            "2",
            "--max-retries",
            "7",
        ]
    )
    assert called.get("max_retries") == 7


def test_cli_manual_tickers_normalize(monkeypatch, capsys):
    idx = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({("Adj Close", "BRK-A"): [100, 101, 102]}, index=idx)
    captured: dict[str, list[str]] = {}

    def _mock_download(tickers, *args, **kwargs):
        captured["tickers"] = list(tickers)
        return prices

    def _identity_sanitize(close, min_days):
        return close, [], []

    def _fake_metric(*_, tickers, **__):
        return pd.DataFrame({"ticker": tickers, "cc_vol": [0.0 for _ in tickers]})

    monkeypatch.setattr(cli, "download_price_history", _mock_download)
    monkeypatch.setattr(cli, "sanitize_close", _identity_sanitize)
    monkeypatch.setitem(cli.METRIC_REGISTRY, "cc_vol", _fake_metric)

    cli.main(["--tickers", "BRK.A", "--print-top", "1", "--min-days", "1"])
    capsys.readouterr()

    assert captured.get("tickers") == ["BRK-A"]
