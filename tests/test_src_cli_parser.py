import pytest

from src.cli import build_parser


@pytest.mark.parametrize("interval", ["10m", "60m"])
def test_cli_parser_accepts_new_intervals(interval):
    parser = build_parser()
    args = parser.parse_args(["--interval", interval, "--tickers", "AAPL"])
    assert args.interval == interval


def test_cli_parser_rejects_invalid_interval():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--interval", "2m", "--tickers", "AAPL"])
