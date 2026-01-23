# Highest Volatility Audit Report

Date: 2026-01-23
Repository: c:\Users\corte\VS Code Projects\highest_volatility

## Audit metadata
- Reviewer: Codex CLI
- Model: gpt-5 (exact model id not exposed by runtime); fallbacks: none.
- Determinism: requested temperature 0.0-0.2; seed not exposed by runtime.
- Scope: static review of local repository files listed below; no external services invoked.

## Scope and method
- Goals: verify data ingestion and cache correctness, API reliability, security boundaries, PWA offline workflow, Streamlit UX stability, observability, and build reproducibility.
- Method: file-by-file static inspection, cross-check docs against implementation, and review unit tests where present.
- Out of scope: production infrastructure configs, live Yahoo Finance or Redis calls, and runtime performance profiling.

## Summary
- Findings tracked: 5 total; 0 open, 5 closed. See `docs/audit/findings_ledger.md`.
- Highest risk: none identified in this audit cycle; continue monitoring auth and key rotation practices.

## Detailed review by audit area

### A1 Data ingestion and cache hydration
Objective: ensure cache data can be hydrated and refreshed safely across intervals.

Files reviewed and steps:
- `src/highest_volatility/cache/store.py`
  - Verified `HV_CACHE_ROOT` expansion via `expand_env_path` to avoid unresolved placeholders.
  - Confirmed cache hydration uses `/prices` with `tickers` and `interval` params.
  - Checked `orient="split"` parsing and single-ticker extraction logic.
  - Verified manifest versioning and corrupt-manifest cleanup.
- `src/highest_volatility/ingest/prices.py`
  - Reviewed cache fetch plan, incremental merge flow, and fingerprint refresh logic.
  - Checked retry behavior and batch download concurrency using thread pools.
- `src/highest_volatility/pipeline/cache_refresh.py`
  - Verified background refresh iterates cached tickers via `asyncio.to_thread`.
  - Confirmed Prometheus metrics for job duration and result counts.
- `scripts/refresh_cache_all_intervals.py`
  - Reviewed interval handling, per-interval lookback policy, failure logging, and CSV output.
- `tests/test_refresh_cache_all_intervals.py`
  - Confirmed happy-path and failure-path coverage for refresh reporting.

Findings:
- F-0002 is resolved: cache hydration aligns with the `/prices` endpoint and split payloads.

Performance notes:
- Cache refresh complexity is O(T) per interval for tickers T, dominated by network I/O.
- Micro-benchmark suggestion: `time python scripts/refresh_cache_all_intervals.py --intervals 1d --tickers AAPL MSFT --max-workers 1 --chunk-sleep 0`.

### A2 Security and API boundaries
Objective: validate input handling and protect write endpoints.

Files reviewed and steps:
- `src/highest_volatility/app/api.py`
  - Verified SlowAPI rate limiting and security headers middleware.
  - Checked input sanitization for tickers, intervals, and numeric bounds.
  - Confirmed annotations persist via SQLite rather than in-memory storage.
  - Reviewed readiness and health logic for Redis availability.
- `src/highest_volatility/security/validation.py`
  - Verified ticker, interval, and metric validation logic.
- `docs/api.md`
  - Cross-checked route contracts, parameters, and error handling with implementation.

Findings:
- F-0001 is resolved: API key authentication enforced on data and annotation endpoints.

### A3 Streamlit UI reliability
Objective: ensure the UI renders without Streamlit runtime errors.

Files reviewed and steps:
- `src/highest_volatility/app/streamlit_app.py`
  - Confirmed `st.altair_chart` uses `use_container_width=True`.
  - Verified `_clamp_int` prevents `min_days` exceeding `lookback_days`.
  - Reviewed data table rendering and chart sizing.
- `tests/test_streamlit_inputs.py`
  - Validated clamp logic for in-range and above-max values.

Findings:
- No open findings in this area.

### A4 PWA offline annotations
Objective: ensure offline queueing, sync, and audit trails operate consistently.

Files reviewed and steps:
- `src/highest_volatility/web/main.js`
  - Verified offline/online state handling and sync UX feedback.
- `src/highest_volatility/web/data-layer.js`
  - Reviewed queueing, last-write-wins conflict policy, and audit log entries.
  - Confirmed annotation PUT payload includes `client_timestamp`.
- `src/highest_volatility/web/db.js`
  - Verified IndexedDB schema and mutation/audit storage.
- `src/highest_volatility/web/service-worker.js`
  - Checked static precache list, API cache fallback, and offline mutation storage.

Findings:
- No open findings; server-side persistence is handled via SQLite.

### A5 Observability and SLO alignment
Objective: confirm metrics exist for documented SLOs and CI validates changes.

Files reviewed and steps:
- `src/highest_volatility/app/api.py`
  - Verified `hv_fastapi_requests_total` and `hv_fastapi_request_latency_ms`.
- `src/highest_volatility/pipeline/cache_refresh.py`
  - Verified `hv_ingestor_job_duration_seconds` and `hv_ingestor_job_results_total`.
- `docs/reliability/slo.md`
  - Confirmed PromQL queries match the implemented metrics.
- `.github/workflows/ci.yml`
  - Confirmed PR CI executes pytest for non-e2e/chaos tests.

Findings:
- F-0005 is closed; metrics and CI align with SLO documentation.

### A6 Build and configuration reproducibility
Objective: ensure deterministic builds and safe environment configuration.

Files reviewed and steps:
- `requirements.txt`
  - Confirmed runtime dependencies are pinned.
- `Dockerfile`
  - Verified build uses pinned requirements and includes a healthcheck.
- `src/highest_volatility/config/paths.py`
  - Verified env var expansion and unresolved placeholder checks.
- `src/highest_volatility/config/environment.py`
  - Confirmed Windows defaults prevent `%SystemDrive%` folder creation.
- `src/highest_volatility/__init__.py`
  - Verified Windows environment defaults are applied on import.

Findings:
- F-0004 is closed: runtime dependencies are pinned and Docker builds are deterministic.

Residual note:
- `pyproject.toml` uses unpinned dependency ranges for editable installs; use `requirements.txt` for production builds.

## Test and validation commands
- Unit tests: `pytest -m "not e2e and not chaos"` (expected pass).
- Streamlit input tests: `pytest tests/test_streamlit_inputs.py` (expected pass).
- Cache refresh script tests: `pytest tests/test_refresh_cache_all_intervals.py` (expected pass).
- Lint (if enabled): `ruff check src tests scripts` (expected pass if ruff is installed).

## Open items
- None.
