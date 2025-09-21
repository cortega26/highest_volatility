import re
import subprocess
import sys

import pandas as pd
import pytest

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

