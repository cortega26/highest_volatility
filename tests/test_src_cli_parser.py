import pytest

from highest_volatility.cli import build_parser


@pytest.mark.parametrize("interval", ["1h"])
def test_cli_parser_accepts_new_interval(interval):
    parser = build_parser()
    args = parser.parse_args(["--interval", interval, "--tickers", "AAPL"])
    assert args.interval == interval


def test_cli_parser_rejects_invalid_interval():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--interval", "2m", "--tickers", "AAPL"])


@pytest.mark.parametrize("interval", ["10m", "60m"])
def test_cli_parser_rejects_removed_intervals(interval):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--interval", interval, "--tickers", "AAPL"])


def test_cli_parser_returns_explicit_tickers():
    parser = build_parser()
    args = parser.parse_args(["--tickers", "AAPL", "MSFT"])
    assert args.tickers == ["AAPL", "MSFT"]


def test_cli_parser_rejects_abbreviated_flags():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--tick", "5", "--tickers", "AAPL"])
