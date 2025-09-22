from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from highest_volatility.app import api


@pytest.fixture(name="prices_client")
def fixture_prices_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient with deterministic price responses."""

    def fake_download_price_history(
        tickers: list[str],
        lookback_days: int,
        *,
        interval: str,
        prepost: bool,
    ) -> pd.DataFrame:
        _ = (lookback_days, interval, prepost)
        index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
        fields = ["Open", "High", "Low", "Close", "Adj Close"]
        columns = pd.MultiIndex.from_product([fields, tickers])
        data: list[list[float]] = []
        for offset, _ in enumerate(index):
            row: list[float] = []
            base = float(offset + 1)
            for field in fields:
                for _ticker in tickers:
                    multiplier = 1.0 if field == "Close" else 10.0
                    row.append(base * multiplier)
            data.append(row)
        return pd.DataFrame(data, index=index, columns=columns)

    async def _noop_refresh(**_: Any) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(api, "download_price_history", fake_download_price_history)
    monkeypatch.setattr(api, "schedule_cache_refresh", _noop_refresh)

    with TestClient(api.app) as client:
        yield client


def test_prices_column_filtering_returns_only_requested_fields(prices_client: TestClient) -> None:
    response = prices_client.get(
        "/prices",
        params={"tickers": "AAA,BBB", "columns": "Close"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["columns"] == [["Close", "AAA"], ["Close", "BBB"]]
    assert all(len(row) == 2 for row in payload["data"])


def test_prices_column_filtering_rejects_unknown_fields(prices_client: TestClient) -> None:
    response = prices_client.get(
        "/prices",
        params={"tickers": "AAA", "columns": "Close,BadField"},
    )
    assert response.status_code == 400
    assert response.json()["detail"].startswith("Unknown columns requested")


def test_prices_response_includes_gzip_encoding(prices_client: TestClient) -> None:
    response = prices_client.get(
        "/prices",
        params={"tickers": "AAA"},
        headers={"Accept-Encoding": "gzip"},
    )
    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"
    assert "gzip" in response.request.headers["accept-encoding"].lower()
