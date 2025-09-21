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


def test_cli_reports_sanitization(monkeypatch, capsys):
    fortune = pd.DataFrame({"rank": [1, 2], "company": ["A", "B"], "ticker": ["A", "B"]})

    monkeypatch.setattr(
        cli,
        "_build_universe_step",
        lambda args: cli.BuildUniverseResult(["A", "B"], fortune, duration=0.0),
    )

    def fake_download(args, tickers):
        return cli.DownloadPricesResult(
            prices=pd.DataFrame(),
            close=pd.DataFrame(),
            tickers=["A"],
            dropped_short=["B"],
            dropped_duplicate=[],
            duration=0.0,
        )

    monkeypatch.setattr(cli, "_download_prices_step", fake_download)
    monkeypatch.setattr(
        cli,
        "_compute_metrics_step",
        lambda args, prices_result, fortune_df: cli.ComputeMetricsResult(
            result=pd.DataFrame(
                {"rank": [1], "company": ["A"], args.metric: [0.1]}, index=["A"]
            ),
            duration=0.0,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_render_output_step",
        lambda args, compute_result: cli.RenderOutputResult(duration=0.0),
    )

    cli.main(["--print-top", "1"])
    output = capsys.readouterr().out
    assert "Sanitized price matrix" in output
