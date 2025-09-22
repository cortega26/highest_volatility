import re
import subprocess
import sys

import pandas as pd
import pytest

from highest_volatility import cli as public_cli
from highest_volatility.app import cli


@pytest.mark.skip(reason="integration test requires selenium and network")
def test_cli_prints_200_rows():
    cmd = [
        sys.executable,
        "-m",
        "highest_volatility",
        "--top-n",
        "300",
        "--print-top",
        "200",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    # Count data lines that start with an uppercase ticker-like token
    lines = [ln for ln in r.stdout.splitlines() if re.match(r"^[A-Z0-9\.-]+\s+", ln)]
    assert len(lines) == 200, f"Expected 200 rows, got {len(lines)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


def test_main_orchestrates_steps(monkeypatch, capsys):
    fortune = pd.DataFrame({"ticker": ["A"], "company": ["A Co"], "rank": [1]})

    monkeypatch.setattr(
        cli,
        "_build_universe_step",
        lambda args: cli.BuildUniverseResult(["A"], fortune, duration=0.1),
    )

    monkeypatch.setattr(
        cli,
        "_download_prices_step",
        lambda args, tickers: cli.DownloadPricesResult(
            prices=pd.DataFrame(),
            close=pd.DataFrame(),
            tickers=["A"],
            dropped_short=[],
            dropped_duplicate=[],
            duration=0.2,
        ),
    )

    df = pd.DataFrame({"rank": [1], "company": ["A Co"], "cc_vol": [0.3]}, index=["A"])
    monkeypatch.setattr(
        cli,
        "_compute_metrics_step",
        lambda args, prices_result, fortune_df: cli.ComputeMetricsResult(
            result=df,
            duration=0.3,
        ),
    )

    monkeypatch.setattr(
        cli,
        "_render_output_step",
        lambda args, compute_result: (print("Rendered"), cli.RenderOutputResult(duration=0.4))[1],
    )

    cli.main(["--print-top", "1", "--timings"])
    output = capsys.readouterr().out
    assert "Rendered" in output
    assert "Timings:" in output


def test_main_skips_fortune_when_tickers_provided(monkeypatch, capsys):
    expected = ["AAPL", "MSFT"]

    monkeypatch.setattr(
        cli,
        "_build_universe_step",
        lambda args: (_ for _ in ()).throw(AssertionError("unexpected Fortune scrape")),
    )

    def fake_download(args, tickers):
        assert tickers == expected
        prices = pd.DataFrame(columns=tickers)
        close = pd.DataFrame(columns=tickers)
        return cli.DownloadPricesResult(
            prices=prices,
            close=close,
            tickers=tickers,
            dropped_short=[],
            dropped_duplicate=[],
            duration=0.0,
        )

    monkeypatch.setattr(cli, "_download_prices_step", fake_download)

    def fake_compute(args, prices_result, fortune_df):
        assert list(fortune_df["ticker"]) == expected
        assert "company" in fortune_df.columns
        assert "rank" in fortune_df.columns
        df = pd.DataFrame({"company": ["Demo"], "rank": [None], "cc_vol": [0.0]}, index=[expected[0]])
        return cli.ComputeMetricsResult(result=df, duration=0.0)

    monkeypatch.setattr(cli, "_compute_metrics_step", fake_compute)

    rendered: dict[str, pd.DataFrame] = {}

    def fake_render(args, compute_result):
        rendered["result"] = compute_result.result
        return cli.RenderOutputResult(duration=0.0)

    monkeypatch.setattr(cli, "_render_output_step", fake_render)

    cli.main(["--tickers", "aapl", "MSFT", "AAPL", "--print-top", "1"])

    output = capsys.readouterr().out
    assert "Using provided tickers" in output
    assert rendered["result"].index.tolist() == [expected[0]]


def test_public_cli_main_orchestrates(monkeypatch, capsys):
    fortune = pd.DataFrame({"ticker": ["A"], "company": ["A Co"], "rank": [1]})

    monkeypatch.setattr(
        cli,
        "_build_universe_step",
        lambda args: cli.BuildUniverseResult(["A"], fortune, duration=0.1),
    )

    monkeypatch.setattr(
        cli,
        "_download_prices_step",
        lambda args, tickers: cli.DownloadPricesResult(
            prices=pd.DataFrame(),
            close=pd.DataFrame(),
            tickers=["A"],
            dropped_short=[],
            dropped_duplicate=[],
            duration=0.2,
        ),
    )

    df = pd.DataFrame({"rank": [1], "company": ["A Co"], "cc_vol": [0.3]}, index=["A"])
    monkeypatch.setattr(
        cli,
        "_compute_metrics_step",
        lambda args, prices_result, fortune_df: cli.ComputeMetricsResult(
            result=df,
            duration=0.3,
        ),
    )

    monkeypatch.setattr(
        cli,
        "_render_output_step",
        lambda args, compute_result: (print("Rendered"), cli.RenderOutputResult(duration=0.4))[1],
    )

    public_cli.main(["--print-top", "1", "--timings"])
    output = capsys.readouterr().out
    assert "Rendered" in output
    assert "Timings:" in output

