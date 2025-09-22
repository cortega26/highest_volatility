# Reliability Test Playbook

This guide explains how to exercise the new resilience, chaos, and performance
tooling introduced for the Highest Volatility platform.

## Prerequisites

- Python 3.10+ with the project dependencies installed (`pip install -r
  requirements.txt`).
- Optional: Locust for load tests (`pip install locust==2.24.1`) or use the
  `perf` extras group (`pip install .[perf]`).
- A running instance of the FastAPI service. The examples assume the default
  development server bound to `http://localhost:8000`.

## Chaos Experiments (Pytest)

Targeted experiments live in `tests/test_chaos_experiments.py` and are marked
with the `chaos` marker.

```bash
pytest -m chaos tests/test_chaos_experiments.py
```

What the suite covers:

1. **Redis outage** – forces a cache backend failure during startup and asserts
   the service falls back to the in-memory cache while staying within the 500 ms
   API latency SLO.
2. **Data-source outage** – simulates upstream price feed failures and verifies
   502 responses are returned quickly with metrics recorded against the
   `DATA_SOURCE` error code.
3. **Cache-refresh cancellation** – injects cancellation into the periodic cache
   refresh loop and checks that it exits without logging failure events.

These tests execute quickly (<1 second each) and can run inside the existing CI
pipeline alongside unit tests. Add the following step to GitHub Actions, Azure
Pipelines, or similar systems after the unit test job:

```yaml
- name: Chaos experiments
  run: pytest -m chaos tests/test_chaos_experiments.py
```

## Performance and Soak Testing (Locust)

Locust scenarios are defined under `tests/performance/`.

1. **Configuration** – environment variables override defaults:
   - `HV_PERF_BASE_URL` (default `http://localhost:8000`)
   - `HV_PERF_TICKERS` (comma-separated, default `AAPL,MSFT,GOOGL,AMZN,TSLA`)
   - `HV_PERF_METRIC` (default `cc_vol`)
   - `HV_PERF_LOOKBACK_DAYS` (default `90`)
   - `HV_PERF_MIN_DAYS` (default `60`)
   - `HV_PERF_STAGE_DURATION` (seconds, default `300`)
   - `HV_PERF_RAMP_USERS` (default `25`)
   - `HV_PERF_STEADY_USERS` (default `75`)
   - `HV_PERF_SLO_MS` (per-request latency SLO, default `500`)

2. **Run a headless soak**:

   ```bash
   locust -f tests/performance/locustfile.py \
     --headless \
     -u "$HV_PERF_STEADY_USERS" \
     -r 5 \
     -t 30m
   ```

   The `RampAndSoakShape` automatically handles warm-up, soak, and ramp-down
   phases while failing any request that breaches the latency SLO.

3. **Scheduled execution** – integrate into CI/CD by running Locust as a gated
   job or a nightly soak:

   ```yaml
   - name: Nightly soak test
     if: github.event_name == 'schedule'
     run: |
       pip install .[perf]
       locust -f tests/performance/locustfile.py --headless -t 45m
   ```

Locust exit codes reflect success/failure, allowing pipelines to fail if the SLO
is breached or if requests error.

## Observability Hooks

- Performance runs emit Locust request failures tagged with `SLO` when latency
  exceeds the configured threshold, providing immediate visibility in test
  reports.
- Chaos experiments rely on the existing `ErrorCode` metrics to confirm outages
  are measured and budgets tracked against the SLO catalogue documented in
  `docs/reliability/slo.md`.
