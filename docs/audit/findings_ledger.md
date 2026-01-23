# Findings Ledger

Severity: S0 Stop-ship | S1 High | S2 Medium | S3 Low

| ID | Severity | Home Audit | Area | Title | Evidence | Impact | Recommendation | Owner | Status | Target Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F-0001 | S1 | A2 | AuthN/AuthZ | API endpoints lack authentication and authorization | `src/highest_volatility/app/api.py` enforces API keys via `_require_api_key`; `src/highest_volatility/web/data-layer.js` sends bearer tokens from local storage. | Unauthorized users can read/write annotations or scrape data; audit trail can be tampered with if exposed publicly. | Implemented: API key authentication enforced for data and annotation endpoints; tests cover authorized and unauthorized access. | TBD | Closed | 2026-01-23 |
| F-0002 | S2 | A1 | Cache Hydration | Cache hydration calls a non-existent API endpoint | `src/highest_volatility/cache/store.py` `_hydrate_from_api` now calls `/prices` with `tickers`/`interval` params; `src/highest_volatility/app/api.py` returns `orient="split"` JSON. | Cache hydration from API silently fails; offline cache misses remain unresolved. | Implemented: cache hydration now aligns with `/prices` contract and parses split payloads before writing cache. | TBD | Closed | 2026-01-23 |
| F-0003 | S2 | A1 | Data Durability | Annotation audit trail stored only in memory | `src/highest_volatility/storage/annotation_store.py` persists annotations in SQLite; `src/highest_volatility/app/api.py` uses `AnnotationStore` with `HV_ANNOTATIONS_DB` configuration. | Notes and audit history are lost on restart; conflicts may be mis-resolved after restarts. | Implemented: SQLite-backed persistence for annotations and audit history. | TBD | Closed | 2026-01-23 |
| F-0004 | S2 | A6 | Build Reproducibility | Dependencies are unpinned in runtime builds | `requirements.txt` now pins runtime dependencies; `Dockerfile` installs from it. | Builds can drift and break or introduce regressions across releases. | Implemented: pinned dependency versions and Docker build alignment. | TBD | Closed | 2026-01-23 |
| F-0005 | S2 | A5 | CI/Observability | Missing PR CI tests and metrics instrumentation for documented SLOs | `.github/workflows/ci.yml` runs pytest; `src/highest_volatility/app/api.py` exposes `hv_fastapi_*` metrics; `src/highest_volatility/pipeline/cache_refresh.py` exposes `hv_ingestor_*`; `docs/reliability/slo.md` references these metrics. | Reduced confidence in changes and unclear SLO enforcement. | Implemented: CI test workflow and Prometheus metrics matching SLO docs. | TBD | Closed | 2026-01-23 |

Notes:
- Evidence should reference concrete files and functions.
- Each finding has one home audit; other audits may reference it but should not duplicate it.
