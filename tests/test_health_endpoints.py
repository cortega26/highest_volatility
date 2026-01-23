from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from highest_volatility.app import api


class _StubRedisClient:
    """Lightweight Redis stub supporting ``ping`` and ``close``."""

    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    async def ping(self) -> None:
        if self.should_fail:
            raise RedisConnectionError("redis unavailable")

    async def close(self) -> None:  # pragma: no cover - behaviourless stub
        return None

    async def eval(self, *args: Any, **kwargs: Any) -> int:  # pragma: no cover - stubbed API
        return 0


@contextmanager
def _instrumented_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    redis_client: _StubRedisClient,
    refresh_factory: Callable[..., Any],
) -> Iterator[TestClient]:
    """Create a :class:`TestClient` with injected Redis and refresh behaviour."""

    monkeypatch.setattr(
        "highest_volatility.app.api.redis.from_url", lambda *args, **kwargs: redis_client
    )
    monkeypatch.setattr(
        "highest_volatility.app.api.schedule_cache_refresh", refresh_factory
    )
    with TestClient(api.app) as client:
        yield client


async def _noop_refresh(*args: Any, **kwargs: Any) -> None:
    await asyncio.sleep(0)


async def _failing_refresh(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("refresh failed")


def test_health_and_readyz_report_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Endpoints should report success when Redis and tasks are healthy."""

    redis_client = _StubRedisClient()
    with _instrumented_client(monkeypatch, redis_client=redis_client, refresh_factory=_noop_refresh) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        payload = health.json()
        assert payload["status"] == "ok"
        assert payload["redis"]["status"] == "up"

        ready = client.get("/readyz")
        assert ready.status_code == 200
        ready_payload = ready.json()
        assert ready_payload["status"] == "ok"


def test_readyz_fails_when_redis_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Readiness must fail fast if Redis cannot be contacted."""

    redis_client = _StubRedisClient(should_fail=True)
    with _instrumented_client(monkeypatch, redis_client=redis_client, refresh_factory=_noop_refresh) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        health_payload = health.json()
        assert health_payload["redis"]["status"] == "down"

        ready = client.get("/readyz")
        assert ready.status_code == 503
        ready_payload = ready.json()
        assert ready_payload["redis"]["status"] == "down"


def test_readyz_allows_in_memory_cache_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Readiness can succeed without Redis when explicitly permitted."""

    redis_client = _StubRedisClient(should_fail=True)
    monkeypatch.setattr(api.settings, "require_redis_for_readyz", False, raising=False)
    with _instrumented_client(monkeypatch, redis_client=redis_client, refresh_factory=_noop_refresh) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 200
        ready_payload = ready.json()
        assert ready_payload["status"] == "ok"


def test_healthz_detects_background_task_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed cache refresh task should trip the liveness probe."""

    redis_client = _StubRedisClient()
    with _instrumented_client(monkeypatch, redis_client=redis_client, refresh_factory=_failing_refresh) as client:
        time.sleep(0.05)
        health = client.get("/healthz")
        assert health.status_code == 503
        payload = health.json()
        assert payload["cache_refresh_task"]["status"] == "error"

        ready = client.get("/readyz")
        assert ready.status_code == 503
