"""Locust scenarios for load and soak testing the public API endpoints."""

from __future__ import annotations

from locust import HttpUser, LoadTestShape, between, events, task

from .config import PerfConfig


CONFIG = PerfConfig.load()


@events.request.add_listener
def enforce_slo(
    request_type,
    name,
    response_time,
    response_length,
    response,
    context,
    exception,
    **kwargs,
):
    """Mark requests exceeding the SLO as failures without aborting the run."""

    if exception or response_time <= CONFIG.target_slo_ms:
        return
    events.request_failure.fire(
        request_type=request_type,
        name=f"{name} (SLO {CONFIG.target_slo_ms}ms)",
        response_time=response_time,
        response_length=response_length,
        exception=AssertionError(
            f"SLO breach: {name} took {response_time:.0f}ms "
            f"(limit {CONFIG.target_slo_ms}ms)"
        ),
    )


class VolatilityUser(HttpUser):
    """User journey exercising `/prices` and `/metrics` endpoints."""

    host = CONFIG.base_url
    wait_time = between(1.0, 3.0)

    @task(3)
    def prices(self) -> None:
        self.client.get(
            "/prices",
            name="/prices",
            params={
                "tickers": CONFIG.tickers,
                "lookback_days": CONFIG.lookback_days,
            },
        )

    @task(1)
    def metrics(self) -> None:
        self.client.get(
            "/metrics",
            name="/metrics",
            params={
                "tickers": CONFIG.tickers,
                "metric": CONFIG.metric,
                "lookback_days": CONFIG.lookback_days,
                "min_days": CONFIG.min_days,
            },
        )


class RampAndSoakShape(LoadTestShape):
    """Ramp up to target throughput, soak, then gracefully ramp down."""

    def __init__(self):
        super().__init__()
        spawn_ramp = max(1, CONFIG.ramp_users // max(1, CONFIG.stage_duration // 10))
        spawn_steady = max(1, CONFIG.steady_users // max(1, CONFIG.stage_duration // 10))
        self._stages = [
            {"length": CONFIG.stage_duration, "users": CONFIG.ramp_users, "spawn": spawn_ramp},
            {"length": CONFIG.stage_duration * 2, "users": CONFIG.steady_users, "spawn": spawn_steady},
            {"length": CONFIG.stage_duration, "users": 0, "spawn": spawn_ramp},
        ]

    def tick(self):
        run_time = self.get_run_time()
        elapsed = 0
        for stage in self._stages:
            elapsed += stage["length"]
            if run_time < elapsed:
                return stage["users"], stage["spawn"]
        return None
