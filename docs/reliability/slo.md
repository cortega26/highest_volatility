# Service Level Objectives

This guide documents the core service level objectives (SLOs) and error budgets
for the Highest Volatility platform. Two services carry explicit reliability
commitments:

1. **FastAPI data API** – interactive pricing and ranking queries.
2. **Data ingestion pipeline** – scheduled Fortune universe refresh and price
   backfills.

Prometheus scrapes the FastAPI service at `/metrics/prometheus`, which exposes
the `hv_fastapi_*` metrics referenced below.

All SLOs adopt a 28-day rolling window unless otherwise specified to balance
rapid incident detection with long-term trend analysis.

## Latency SLOs

### FastAPI data API

- **Objective**: 95% of HTTP responses complete within 300 ms.
- **Formula**:
  \[
  \text{Latency Compliance} = \frac{\text{count}_\text{fast}(t_{0..28d})}{\text{count}_\text{total}(t_{0..28d})}
  \]
  Where `count_fast` counts requests with `latency_ms <= 300`.
- **Measurement source**: Prometheus histogram `hv_fastapi_request_latency_ms`
  scraped via the query:
  ```promql
  sum(increase(hv_fastapi_request_latency_ms_bucket{le="300"}[28d]))
  /
  sum(increase(hv_fastapi_request_latency_ms_count[28d]))
  ```
- **Target**: ≥ 0.95.
- **Dashboard panel**: Grafana row "FastAPI → Latency".

### Data ingestion pipeline

- **Objective**: 99% of batch refresh tasks finish within 10 minutes of their
  scheduled trigger.
- **Formula**:
  \[
  \text{Latency Compliance} = \frac{\text{jobs}_\text{on\_time}(t_{0..28d})}{\text{jobs}_\text{triggered}(t_{0..28d})}
  \]
- **Measurement source**: Prometheus histogram `hv_ingestor_job_duration_seconds`
  with query:
  ```promql
  sum(increase(hv_ingestor_job_duration_seconds_bucket{le="600"}[28d]))
  /
  sum(increase(hv_ingestor_job_duration_seconds_count[28d]))
  ```
- **Target**: ≥ 0.99.

## Availability SLOs

### FastAPI data API

- **Objective**: 99.5% availability based on successful responses.
- **Formula**:
  \[
  \text{Availability} = 1 - \frac{\text{5xx responses}}{\text{total responses}}
  \]
- **Measurement source**: Prometheus counter `hv_fastapi_requests_total` with
  status labels. Query:
  ```promql
  1 - (
    sum(increase(hv_fastapi_requests_total{status=~"5.."}[28d]))
    /
    sum(increase(hv_fastapi_requests_total[28d]))
  )
  ```
- **Target**: ≥ 0.995.

### Data ingestion pipeline

- **Objective**: 99% of scheduled jobs complete successfully without retries.
- **Formula**:
  \[
  \text{Availability} = \frac{\text{jobs}_\text{success}}{\text{jobs}_\text{scheduled}}
  \]
- **Measurement source**: Prometheus counter
  `hv_ingestor_job_results_total{result="success"}`. Query:
  ```promql
  sum(increase(hv_ingestor_job_results_total{result="success"}[28d]))
  /
  sum(increase(hv_ingestor_job_results_total[28d]))
  ```
- **Target**: ≥ 0.99.

## Error-rate SLOs

### FastAPI data API

- **Objective**: Error rate under 1% across rolling 10-minute windows.
- **Formula**:
  \[
  \text{Error Rate} = \frac{\text{4xx} + \text{5xx responses}}{\text{total responses}}
  \]
- **Measurement source**: Same `hv_fastapi_requests_total` counter aggregated
  per 10-minute interval. PromQL alert expression:
  ```promql
  (
    sum(rate(hv_fastapi_requests_total{status=~"4..|5.."}[10m]))
    /
    sum(rate(hv_fastapi_requests_total[10m]))
  )
  > 0.01
  ```
- **Target**: ≤ 0.01 (1%).

### Data ingestion pipeline

- **Objective**: Task failure ratio below 0.5% per day.
- **Formula**:
  \[
  \text{Failure Ratio} = \frac{\text{jobs}_\text{failed}}{\text{jobs}_\text{scheduled}}
  \]
- **Measurement source**: Prometheus counter `hv_ingestor_job_results_total`
  with `result="failed"`. Query:
  ```promql
  (
    sum(increase(hv_ingestor_job_results_total{result="failed"}[1d]))
    /
    sum(increase(hv_ingestor_job_results_total[1d]))
  )
  ```
- **Target**: ≤ 0.005 (0.5%).

## Error Budgets

Error budgets define the allowable downtime or errors before action is required.
Budgets are calculated as `1 - SLO target`. For example, the FastAPI availability
SLO (99.5%) yields a 0.5% budget over 28 days (~3 hours 22 minutes of downtime).

| Service | SLO | Target | Error Budget (28d) |
| --- | --- | --- | --- |
| FastAPI | Availability | 99.5% | 0.5% (~3h22m) |
| FastAPI | Latency | 95% | 5% of requests |
| FastAPI | Error rate | ≤1% | 1% of requests |
| Ingestor | Availability | 99% | 1% (~6h44m) |
| Ingestor | Latency | 99% | 1% of jobs |
| Ingestor | Error rate | ≤0.5% | 0.5% of jobs |

### Budget Burn Policies

| Burn Rate | Condition | Action |
| --- | --- | --- |
| 1x | Consumption equals budget over 28 days | Track in weekly review, no immediate action. |
| 2x | Budget exhausted twice as fast (14-day projection) | Trigger PagerDuty low-urgency incident; assign on-call to investigate root cause within 24h. |
| 4x | Budget exhausted four times as fast (7-day projection) | Trigger PagerDuty high-urgency incident; initiate incident bridge and engage engineering manager. |
| ≥6x | Budget exhausted in <5 days | Escalate to director, freeze feature releases until mitigation plan approved. |

Prometheus burn rate query template:
```promql
(
  sum(rate(<relevant_error_metric>[5m]))
  /
  sum(rate(<relevant_total_metric>[5m]))
)
/
(1 - <slo_target>)
```

Evaluate both 5-minute and 1-hour windows for fast and slow burn alerts
respectively.

## Escalation & Alerting Matrix

| Severity | Trigger | Alert Channels | Responders |
| --- | --- | --- | --- |
| SEV-1 | Burn rate ≥6x or total outage >30 minutes | PagerDuty (high), Slack #hv-incident, phone bridge | Primary on-call, backup on-call, engineering manager, SRE lead |
| SEV-2 | Burn rate ≥4x for 30 minutes, or consecutive batch failures >3 | PagerDuty (medium), Slack #hv-incident | Primary on-call, service owner |
| SEV-3 | Burn rate ≥2x sustained for 2 hours | Slack #hv-reliability, Opsgenie FYI | Primary on-call |
| SEV-4 | Single job failure outside maintenance window | Slack #hv-reliability | Triage rotation |

Alert routing is configured in PagerDuty with FastAPI and ingestor service tags.
Slack alerts originate from Alertmanager webhooks using the `hv-alerts`
integration.

## Keeping Dashboards & Alerts in Sync

- Version control all Prometheus rules and Grafana JSON dashboards alongside the
  application manifests under `deploy/`.
- During each deployment, run the `scripts/deploy_dashboards.py` helper to apply
  dashboard updates and sync alert rules.
- Require changes to SLO targets to include matching updates to the dashboards
  and alert rule files within the same pull request.
- The release checklist includes verifying Grafana panels render the latest
  queries and that PagerDuty services reference the current Alertmanager routes.

