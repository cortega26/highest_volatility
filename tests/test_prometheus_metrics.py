from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from highest_volatility.app import api


async def _noop_refresh(**_: object) -> None:
    await asyncio.sleep(0)


def test_prometheus_metrics_endpoint_exposes_fastapi_metrics(monkeypatch) -> None:
    monkeypatch.setattr(api, "schedule_cache_refresh", _noop_refresh, raising=False)

    with TestClient(api.app) as client:
        client.get("/healthz")
        response = client.get("/metrics/prometheus")

    assert response.status_code == 200
    assert "hv_fastapi_requests_total" in response.text
    assert "hv_fastapi_request_latency_ms" in response.text
    assert "hv_ingestor_job_duration_seconds" in response.text
    assert "hv_ingestor_job_results_total" in response.text


def test_prometheus_metrics_records_validation_errors(monkeypatch) -> None:
    monkeypatch.setattr(api, "schedule_cache_refresh", _noop_refresh, raising=False)

    with TestClient(api.app) as client:
        client.get("/metrics")
        response = client.get("/metrics/prometheus")

    assert response.status_code == 200
    assert 'path="/metrics"' in response.text
    assert 'status="422"' in response.text
