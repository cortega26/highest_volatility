# Highest Volatility Audit Report
Date: 2026-01-22

This report follows the A00 audit program (A-1 through A8, then A4b).
Evidence references point to files in the repository.

## A-1 Architecture and System Context Pack

### System context (text diagram)
- Users and automation
  - CLI users run `python -m highest_volatility` or `highest_volatility.cli`.
  - Streamlit users run `streamlit run src/highest_volatility/app/streamlit_app.py`.
  - API clients call FastAPI endpoints `/universe`, `/prices`, `/metrics`.
  - PWA users load the web app served by FastAPI at `/` and use offline annotations.
- Core system
  - `highest_volatility` Python package provides universe building, price download, metrics.
- External systems
  - Yahoo Finance via `yfinance` and async HTTP datasource.
  - Fortune list via Selenium scraping of `us500.com`.
- Data stores and caches
  - On-disk price cache in `cache/prices/<interval>`.
  - Redis cache for API response caching (FastAPICache).
  - In-memory annotation store in the FastAPI process.
  - PWA IndexedDB for offline mutation queue and audit trail.

Evidence: `src/highest_volatility/app/cli.py`, `src/highest_volatility/app/streamlit_app.py`, `src/highest_volatility/app/api.py`, `src/highest_volatility/ingest/prices.py`, `src/highest_volatility/universe.py`, `src/highest_volatility/web/main.js`.

### Container diagram (text)
- CLI container
  - Entrypoint: `src/highest_volatility/app/cli.py`.
  - Depends on: universe builder, price downloader, metric registry, local cache.
- Streamlit UI container
  - Entrypoint: `src/highest_volatility/app/streamlit_app.py`.
  - Depends on: universe builder, price downloader, metric table helpers.
- FastAPI service container
  - Entrypoint: `highest_volatility.app.api:app` (Docker/uvicorn).
  - Exposes: `/universe`, `/prices`, `/metrics`, `/healthz`, `/readyz`, `/annotations`.
  - Background task: cache refresh loop.
  - Serves: PWA static assets under `/web` and root `/`.
- Data plane
  - Local cache in `cache/prices` (Parquet + JSON manifest).
  - Redis (optional) for API response caching.

Evidence: `README.md`, `Dockerfile`, `src/highest_volatility/app/api.py`, `src/highest_volatility/cache/store.py`, `src/highest_volatility/pipeline/cache_refresh.py`.

### Runtime topology table
| Component | Runtime | Entrypoint | Env vars (examples) | Dependencies | Ports | Persistence |
| --- | --- | --- | --- | --- | --- | --- |
| CLI | Python 3.10+ | `highest_volatility.app.cli:main` | none required | Yahoo Finance, Selenium | n/a | local cache, optional CSV/SQLite outputs |
| Streamlit UI | Python 3.10+ | `streamlit run src/highest_volatility/app/streamlit_app.py` | none required | Yahoo Finance, Selenium | 8501 (default) | local cache |
| FastAPI API | Python 3.10+ | `uvicorn highest_volatility.app.api:app` | `HV_*`, `HV_REDIS_URL` | Redis, Yahoo Finance | 8000 | local cache, Redis |
| Cache refresh worker | Async task | `schedule_cache_refresh` | `HV_CACHE_REFRESH_INTERVAL` | Yahoo Finance, local cache | n/a | local cache |
| PWA frontend | Browser | `/` and `/web/*` | n/a | IndexedDB, service worker | n/a | IndexedDB |

Evidence: `src/highest_volatility/app/api.py`, `src/highest_volatility/pipeline/cache_refresh.py`, `README.md`.

### Trust boundaries and data classification
- Boundary: browser or client -> FastAPI. All API inputs are untrusted.
- Boundary: FastAPI/CLI -> Yahoo Finance and `us500.com` (external systems).
- Boundary: FastAPI -> Redis (external service).
- Boundary: local filesystem for `cache/prices`.

Data classification:
- Public data: ticker symbols, price history, derived metrics.
- User-provided data: annotation notes (treat as potentially sensitive).
- Operational data: logs and error context (should be redacted).

Evidence: `src/highest_volatility/security/validation.py`, `src/highest_volatility/logging.py`, `src/highest_volatility/app/api.py`.

### Critical invariants (must not break)
- Ticker normalization and deduplication preserve Fortune rankings.
- `min_days` filtering prevents metrics on insufficient history.
- Cache manifests match data (version, row count, date range).
- Price frames sorted and without duplicate indices.
- API inputs validated for range and format.

Evidence: `src/highest_volatility/universe.py`, `src/highest_volatility/app/sanitization.py`, `src/highest_volatility/pipeline/validation.py`, `src/highest_volatility/security/validation.py`.

### Critical journeys (sampling anchors)
1. CLI analysis: build universe -> download prices -> sanitize -> compute metrics -> output.
2. API metrics: `/metrics` -> sanitize -> download prices -> compute metric -> JSON response.
3. Streamlit UI: config -> build universe -> download prices -> sanitize -> compute -> charts.
4. PWA offline annotations: service worker queue -> `/annotations` write -> merge -> audit trail.

Evidence: `src/highest_volatility/app/cli.py`, `src/highest_volatility/app/api.py`, `src/highest_volatility/app/streamlit_app.py`, `src/highest_volatility/web/*`.

### Glossary (domain language)
- Universe: set of tickers from the Fortune list.
- Fortune list: ranked Fortune 500 entries scraped via Selenium.
- Lookback days: rolling window size for price history.
- Interval: Yahoo Finance interval string (1d, 1h, 15m, etc.).
- Metric: computed volatility or risk measure (cc_vol, gk_vol, etc.).
- Cache manifest: JSON metadata describing cached price data.
- Cache refresh: background re-download of cached tickers.
- Annotation: user note attached to a ticker in the PWA.
- PWA: progressive web app served by FastAPI root.

## A2 Security, AppSec, Threat Model

Threat model (lightweight):
- Assets: cached price data, annotation notes, API availability, Redis cache.
- Actors: public users, internal analysts, automated clients, unauthenticated internet.
- Entrypoints: `/universe`, `/prices`, `/metrics`, `/annotations`, `/healthz`, `/readyz`.
- Trust boundaries: client to API, API to external data sources, API to Redis.

Top abuse cases (non-exhaustive):
- Unauthorized write to `/annotations` to tamper with audit trail.
- Excessive requests to `/prices` or `/metrics` to exhaust rate limit.
- Injection via ticker parameters (mitigated by validation).
- Redis outage causes degraded caching (handled with fallback).

Findings:
- F-0001 (S1) Lack of authentication/authorization on API endpoints.

Notes:
- Input validation exists for ticker, interval, and metric parameters.
- Security headers are set in the API middleware.

Evidence: `src/highest_volatility/app/api.py`, `src/highest_volatility/security/validation.py`.

## A6 Release, Environment, Change Safety

Environment map:
- Local dev: direct Python execution, `pip install -r requirements.txt`.
- Docker: `uvicorn` in container, healthcheck on `/healthz`.
- GitHub Actions: scheduled workflows for price fetch and Fortune list updates.

Configuration inventory (non-secret):
- `HV_LOOKBACK_DAYS`, `HV_INTERVAL`, `HV_PREPOST`, `HV_TOP_N`, `HV_METRIC`, `HV_MIN_DAYS`.
- `HV_REDIS_URL`, `HV_CACHE_TTL_*`, `HV_RATE_LIMIT`, `HV_CACHE_REFRESH_INTERVAL`.
- `HV_CACHE_ROOT`, `HV_API_BASE_URL` (cache hydration).

Release safety observations:
- Health and readiness endpoints exist and are used by Docker healthcheck.
- No pinning of Python package versions in `requirements.txt` or Docker builds.

Findings:
- F-0004 (S2) Non-deterministic builds due to unpinned dependencies.

Evidence: `Dockerfile`, `requirements.txt`, `README.md`, `.github/workflows/*.yml`.

## A1 Business Logic and Behavioral Integrity

Invariants review:
- Universe builder preserves rank alignment and normalization.
- Sanitization removes short or duplicate time series before metrics.
- Cache validation enforces index continuity and manifest alignment.

Findings:
- F-0002 (S2) Cache hydration points to a non-existent API endpoint.
- F-0003 (S2) Annotation audit trail stored in memory only.

Evidence: `src/highest_volatility/cache/store.py`, `src/highest_volatility/app/api.py`, `docs/api.md`.

## A0 Project Structure and Modularity (Lite)

Observations:
- Core code is organized under `src/highest_volatility` with clear subpackages.
- Root directory contains debug artifacts and CSVs that are not part of the runtime.

Risks:
- Root clutter may reduce discoverability and packaging hygiene.

No blocking structural issues observed for ship safety.

Evidence: repository root, `src/highest_volatility/*`.

## A4a Engineering Quality, Tests, Maintainability, Performance

Test posture:
- Unit and integration tests cover CLI, API, cache, sanitization, and metrics.
- E2E PWA offline sync test exists with Playwright.
- Chaos experiments and performance benchmarks are present.

Performance notes:
- `additional_volatility_measures` is optimized to reduce repeated coercions.
- Complexity (hot path):
  - `additional_volatility_measures`: O(F * N + T * N) where F = fields, N = rows, T = tickers.
  - `download_batch`: O(T) downloads with chunking and thread pool.

Micro-benchmark:
- `python scripts/benchmarks/ohlc_volatility_perf.py --rows 100000 --tickers 500 --repeats 3 --interval 1m`

Evidence: `tests/*`, `scripts/benchmarks/ohlc_volatility_perf.py`, `docs/reliability/testing.md`.

## A5 Process, Operations, DevEx

Observations:
- Scheduled workflows exist for price fetch and Fortune list updates.
- SLO and reliability docs exist; health endpoints implemented.

Gaps:
- No CI workflow running unit tests on push or pull request.
- Prometheus metrics referenced in docs are not defined in code or deployment configs.

Findings:
- F-0005 (S2) CI and observability gaps relative to documented SLOs.

Evidence: `.github/workflows/*.yml`, `docs/reliability/slo.md`, `src/highest_volatility/logging.py`.

## A3 Data and AI Lineage

Lineage:
- Fortune list (Selenium) -> universe cache -> CLI/Streamlit/API.
- Yahoo Finance data -> on-disk cache -> metrics -> API/UI outputs.
- PWA annotations -> FastAPI in-memory store -> PWA audit view.

Data quality controls:
- Cache validation checks for gaps and manifest alignment.
- Sanitization drops short or duplicate series prior to metrics.

AI/ML:
- No ML training or inference paths detected.

Evidence: `src/highest_volatility/universe.py`, `src/highest_volatility/ingest/prices.py`, `src/highest_volatility/pipeline/validation.py`.

## A7 Compliance and Governance

Applicability:
- Primarily public market data with optional user-generated notes.

Gaps to address if serving external users:
- Define retention and deletion policy for annotation notes.
- Clarify audit trail durability expectations.

Evidence: `README.md`, `src/highest_volatility/app/api.py`.

## A8 FinOps and Efficiency

Cost drivers:
- Selenium scraping and Yahoo Finance downloads.
- Redis cache and storage if deployed.

Efficiency notes:
- On-disk caching and batch downloads reduce repeated network calls.
- Background refresh can be tuned via `HV_CACHE_REFRESH_INTERVAL`.

Evidence: `src/highest_volatility/ingest/prices.py`, `src/highest_volatility/cache/store.py`.

## A4b UX, Accessibility, Design System

Gate status:
- A2 and A1 have open findings (see ledger). Defer major UX refactors until ship-safety issues are addressed.

Notes:
- Streamlit UI includes download buttons and warning banners for empty data.
- PWA includes offline indicators and sync feedback.

Evidence: `src/highest_volatility/app/streamlit_app.py`, `src/highest_volatility/web/main.js`.
