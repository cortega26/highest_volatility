from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from highest_volatility.app import api


class DummyFortune:
    def to_dict(self, orient: str) -> list[dict[str, Any]]:
        assert orient == "records"
        return [
            {
                "rank": 1,
                "company": "Acme Corp",
                "ticker": "ACME",
                "normalized_ticker": "ACME",
            }
        ]


class DummyMetricResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def to_dict(self, orient: str) -> list[dict[str, Any]]:
        assert orient == "records"
        return self._rows


class StubFrame:
    empty = False

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def to_json(self, orient: str, date_format: str) -> str:  # noqa: D401 - simple proxy
        assert orient == "split"
        assert date_format == "iso"
        return json.dumps(self._payload)


@pytest.fixture(name="api_client")
def fixture_api_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient with deterministic cache dependencies."""

    monkeypatch.setattr(api, "build_universe", lambda limit, validate=True: (["ACME"], DummyFortune()))

    def fake_download_price_history(*args: Any, **kwargs: Any) -> StubFrame:
        payload = {
            "columns": ["Close"],
            "index": ["2024-01-01T00:00:00+00:00"],
            "data": [[123.45]],
        }
        frame = StubFrame(payload)
        return frame

    monkeypatch.setattr(api, "download_price_history", fake_download_price_history)

    def fake_metric(prices: Any, *, tickers: list[str], min_periods: int, interval: str) -> DummyMetricResult:
        rows = [{"ticker": ticker, "value": idx + 0.1} for idx, ticker in enumerate(tickers)]
        return DummyMetricResult(rows)

    monkeypatch.setitem(api.METRIC_REGISTRY, "cc_vol", fake_metric)
    monkeypatch.setattr(api.settings, "cache_ttl_universe", 90, raising=False)
    monkeypatch.setattr(api.settings, "cache_ttl_prices", 120, raising=False)
    monkeypatch.setattr(api.settings, "cache_ttl_metrics", 150, raising=False)

    async def _noop_refresh(**kwargs: Any) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(api, "schedule_cache_refresh", _noop_refresh)

    with TestClient(api.app) as client:
        yield client


def _assert_cache_headers(response, ttl: int) -> None:
    assert response.headers["cache-control"] == f"public, max-age={ttl}"
    assert response.headers["surrogate-control"] == f"max-age={ttl}"
    expires = parsedate_to_datetime(response.headers["expires"])
    now = datetime.now(timezone.utc)
    delta = expires - now
    lower = timedelta(seconds=max(ttl - 2, 0))
    upper = timedelta(seconds=ttl + 2)
    assert lower <= delta <= upper


@pytest.mark.parametrize(
    ("path", "params", "ttl_attr"),
    (
        ("/universe", {}, "cache_ttl_universe"),
        ("/prices", {"tickers": "ACME"}, "cache_ttl_prices"),
        (
            "/metrics",
            {"tickers": "ACME", "metric": "cc_vol"},
            "cache_ttl_metrics",
        ),
    ),
)
def test_cache_headers_and_etag(api_client: TestClient, path: str, params: dict[str, str], ttl_attr: str) -> None:
    response = api_client.get(path, params=params)
    assert response.status_code == 200

    ttl = getattr(api.settings, ttl_attr)
    _assert_cache_headers(response, ttl)

    etag = response.headers["etag"]
    assert etag.startswith('"') and etag.endswith('"')
    digest = hashlib.sha256(response.content).hexdigest()
    assert etag == f'"{digest}"'

    revalidated = api_client.get(path, params=params, headers={"If-None-Match": etag})
    assert revalidated.status_code == 304
    assert revalidated.content == b""
    _assert_cache_headers(revalidated, ttl)
    assert revalidated.headers["etag"] == etag
