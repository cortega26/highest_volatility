# Findings Ledger

Severity: S0 Stop-ship | S1 High | S2 Medium | S3 Low

| ID | Severity | Home Audit | Area | Title | Evidence | Impact | Recommendation | Owner | Status | Target Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F-0001 | S1 | A2 | AuthN/AuthZ | API endpoints lack authentication and authorization | `src/highest_volatility/app/api.py` defines all routes without auth dependencies. | Unauthorized users can read/write annotations or scrape data; audit trail can be tampered with if exposed publicly. | Add auth (API key, OAuth, or internal auth middleware) and restrict write endpoints; add tests for protected routes. | TBD | Open | TBD |
| F-0002 | S2 | A1 | Cache Hydration | Cache hydration calls a non-existent API endpoint | `src/highest_volatility/cache/store.py` uses `/prices/{ticker}?interval={interval}&fmt=parquet`; `src/highest_volatility/app/api.py` exposes only `/prices` and no `fmt` param. | Cache hydration from API silently fails; offline cache misses remain unresolved. | Align API and cache: add a single-ticker parquet endpoint or update hydration to match `/prices` contract; update docs. | TBD | Open | TBD |
| F-0003 | S2 | A1 | Data Durability | Annotation audit trail stored only in memory | `src/highest_volatility/app/api.py` uses `app.state` for annotations and history with no persistence. | Notes and audit history are lost on restart; conflicts may be mis-resolved after restarts. | Persist annotations (SQLite/Redis) or document ephemeral behavior and scope. | TBD | Open | TBD |
| F-0004 | S2 | A6 | Build Reproducibility | Dependencies are unpinned in runtime builds | `requirements.txt` uses floating versions; `Dockerfile` installs from it without pinning. | Builds can drift and break or introduce regressions across releases. | Add pinned versions or a lockfile/constraints file; update Docker builds to use it. | TBD | Open | TBD |
| F-0005 | S2 | A5 | CI/Observability | Missing PR CI tests and metrics instrumentation for documented SLOs | `.github/workflows` only schedule jobs; `docs/reliability/slo.md` references Prometheus metrics not present in `src`. | Reduced confidence in changes and unclear SLO enforcement. | Add CI workflow for tests and implement or revise metrics instrumentation to match SLO docs. | TBD | Open | TBD |

Notes:
- Evidence should reference concrete files and functions.
- Each finding has one home audit; other audits may reference it but should not duplicate it.
