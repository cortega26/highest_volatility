"""Configuration helpers for performance testing."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PerfConfig:
    """Locust configuration derived from environment variables."""

    base_url: str
    tickers: str
    metric: str
    lookback_days: int
    min_days: int
    stage_duration: int
    ramp_users: int
    steady_users: int
    target_slo_ms: int

    @classmethod
    def load(cls) -> "PerfConfig":
        base_url = os.getenv("HV_PERF_BASE_URL", "http://localhost:8000")
        tickers = os.getenv("HV_PERF_TICKERS", "AAPL,MSFT,GOOGL,AMZN,TSLA")
        metric = os.getenv("HV_PERF_METRIC", "cc_vol")
        lookback_days = int(os.getenv("HV_PERF_LOOKBACK_DAYS", "90"))
        min_days = int(os.getenv("HV_PERF_MIN_DAYS", "60"))
        stage_duration = int(os.getenv("HV_PERF_STAGE_DURATION", "300"))
        ramp_users = int(os.getenv("HV_PERF_RAMP_USERS", "25"))
        steady_users = int(os.getenv("HV_PERF_STEADY_USERS", "75"))
        target_slo_ms = int(os.getenv("HV_PERF_SLO_MS", "500"))
        return cls(
            base_url=base_url,
            tickers=tickers,
            metric=metric,
            lookback_days=lookback_days,
            min_days=min_days,
            stage_duration=stage_duration,
            ramp_users=ramp_users,
            steady_users=steady_users,
            target_slo_ms=target_slo_ms,
        )
