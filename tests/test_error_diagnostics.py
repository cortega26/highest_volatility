"""Tests for enhanced error diagnostic output."""

from __future__ import annotations

from highest_volatility.errors import DataSourceError, describe_exception


class DummyConnectorError(Exception):
    """Simulate aiohttp connector errors with OS error details."""

    def __init__(self, message: str, errno: int) -> None:
        super().__init__(message)
        self.os_error = OSError(errno, message)


def test_describe_exception_includes_os_error_details() -> None:
    error = DummyConnectorError("Cannot reach host", errno=101)
    payload = describe_exception(error)

    assert payload["type"] == "DummyConnectorError"
    assert "Cannot reach host" in payload["message"]
    assert payload["os_error"]["errno"] == 101
    assert "Cannot reach host" in payload["os_error"]["message"]


def test_hverror_to_dict_embeds_full_cause_chain() -> None:
    try:
        raise DummyConnectorError("Cannot reach host", errno=101)
    except DummyConnectorError as exc:
        wrapped = DataSourceError("Datasource request failed", cause=exc)

    payload = wrapped.to_dict()
    cause = payload["cause"]

    assert cause["type"] == "DummyConnectorError"
    assert cause["os_error"]["errno"] == 101
    # Ensure we do not just emit the class name.
    assert "Cannot reach host" in cause["message"]
