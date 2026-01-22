# Highest Volatility API Reference

The FastAPI application in `highest_volatility.app.api` exposes operational
endpoints for working with cached equity data and derived metrics. This
maintained module replaces the deprecated `src/api/__init__.py` entry point, so
deployments must target `highest_volatility.app.api:app` directly. Unless
otherwise noted, all endpoints respond with JSON and share a common error
contract described in [Error handling](#error-handling). Run `uvicorn
highest_volatility.app.api:app` (or an equivalent ASGI server command) to start
the service. The table below summarises the exposed routes.

| Method | Path              | Description                             |
| ------ | ----------------- | --------------------------------------- |
| GET    | `/universe`       | Return a validated Fortune ticker list. |
| GET    | `/prices`         | Fetch cached Yahoo Finance price data.  |
| GET    | `/metrics`        | Compute metrics for requested tickers.  |
| GET    | `/healthz`        | Report process and background task state. |
| GET    | `/readyz`         | Report readiness of external services.  |

## HTTP caching

The `/universe`, `/prices`, and `/metrics` endpoints emit cache-friendly
headers derived from the configured TTLs (`Settings.cache_ttl_*`). Responses are
tagged as public and include:

* `Cache-Control: public, max-age=<TTL>`
* `Surrogate-Control: max-age=<TTL>` for CDN-aware caching layers
* `Expires` calculated from the request time plus the TTL
* A strong `ETag` generated from the canonical JSON payload

Clients may revalidate cached payloads by sending the `If-None-Match` header.
When the entity tag matches the current payload the API returns `304 Not
Modified` with the caching headers above so intermediaries can retain their
objects. Updating the TTL environment variables takes effect on the next
process restart.

All JSON payloads now use FastAPI's `ORJSONResponse` for faster serialization,
and the ASGI stack enables `GZipMiddleware`. Clients that send
`Accept-Encoding: gzip` receive compressed responses annotated with
`Content-Encoding: gzip`.

## `/universe`

Return a curated Fortune universe backed by Selenium scraping and cached
normalisation logic.

### Query parameters

| Name     | Type    | Default (`HV_` override) | Description |
| -------- | ------- | ------------------------ | ----------- |
| `top_n`  | int     | `Settings.top_n` (`HV_TOP_N`, default `100`) | Upper bound on ranked Fortune rows to include. Must satisfy `10 ≤ top_n ≤ 500`.

If omitted, `top_n` falls back to the configured default.

### Success response

Status `200 OK` with payload:

```json
{
  "tickers": ["AAPL", "MSFT", "BRK-B", "NVDA", ...],
  "fortune": [
    {"rank": 1, "company": "Walmart", "ticker": "WMT", "normalized_ticker": "WMT"},
    {"rank": 2, "company": "Amazon", "ticker": "AMZN", "normalized_ticker": "AMZN"},
    ...
  ]
}
```

The `fortune` array mirrors the cached DataFrame records and can include
additional diagnostic columns such as `normalized_ticker`.

### Error responses

Validation failures raise `400 Bad Request` with a body shaped as
`{"detail": "Value must be at least 10."}` when input bounds are violated.
Backend failures while scraping or loading caches surface `502 Bad Gateway`
or `503 Service Unavailable` depending on the underlying error code.

## `/prices`

Return Yahoo Finance price history for one or more symbols.  Prices are read
from disk caches; cache misses trigger background refresh tasks.

### Query parameters

| Name           | Type    | Default (`HV_` override) | Description |
| -------------- | ------- | ------------------------ | ----------- |
| `tickers`      | string  | _required_               | Comma-separated list of tickers. Limited to 100 items and validated against `^[A-Z0-9.\-]{1,10}$`.
| `lookback_days`| int     | `Settings.lookback_days` (`HV_LOOKBACK_DAYS`, default `252`) | Rolling window size. Bounds: `30 ≤ lookback_days ≤ 2000`.
| `interval`     | string  | `Settings.interval` (`HV_INTERVAL`, default `"1d"`) | Yahoo Finance interval string (`1d`, `1h`, `1wk`, `30m`, ...).
| `prepost`      | bool    | `Settings.prepost` (`HV_PREPOST`, default `false`) | Include pre/post-market prices when supported.
| `columns`      | string  | _optional_               | Comma-separated, case-insensitive OHLCV column groups to retain (for example, `columns=close` limits the payload to Close slices for every ticker).

### Success response

Status `200 OK`. When data exists the response matches Pandas' `orient="split"`
layout:

```json
{
  "columns": ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
  "index": ["2024-01-02T00:00:00+00:00", "2024-01-03T00:00:00+00:00"],
  "data": [[185.35, 187.33, 183.77, 184.25, 184.25, 48216800], ...]
}
```

If no rows are returned the endpoint responds with `{"data": []}` to denote an
empty frame.

When a `columns` parameter is supplied the service filters the DataFrame prior
to serialization. For example, requesting
`/prices?tickers=AAPL,MSFT&columns=close,adj close` returns only the `Close` and
`Adj Close` entries for each ticker, dramatically reducing payload size for
clients that do not need full OHLCV data.

### Error responses

* `400 Bad Request` for sanitisation errors such as unknown intervals or too
  many tickers (`{"detail": "Too many tickers supplied."}`).
* `502 Bad Gateway` for upstream download problems surfaced as `DataSourceError`s.
* `502 Bad Gateway` for integration failures when wrapping unexpected exceptions.

## `/metrics`

Compute ranking metrics for the requested tickers.

### Query parameters

| Name           | Type    | Default (`HV_` override) | Description |
| -------------- | ------- | ------------------------ | ----------- |
| `tickers`      | string  | _required_               | Comma-separated tickers; same validation as `/prices`.
| `metric`       | string  | `Settings.metric` (`HV_METRIC`, default `"cc_vol"` – Close-to-Close Volatility) | Metric key registered in `METRIC_REGISTRY`.
| `lookback_days`| int     | `Settings.lookback_days` (`HV_LOOKBACK_DAYS`) | Rolling window used for data downloads.
| `interval`     | string  | `Settings.interval` (`HV_INTERVAL`) | Yahoo Finance interval.
| `min_days`     | int     | `Settings.min_days` (`HV_MIN_DAYS`, default `126`) | Minimum observations needed per ticker. Must satisfy `10 ≤ min_days ≤ lookback_days`.

### Success response

Status `200 OK` with an array of records.  Each record reflects the DataFrame
returned by the selected metric.  A common schema is:

```json
[
  {"ticker": "AAPL", "value": 0.42},
  {"ticker": "MSFT", "value": 0.37}
]
```

Metrics that emit richer data (for example, Sharpe ratios) include additional
fields per row.

### Error responses

* `400 Bad Request` when a metric key is unknown or query parameters fail
  sanitisation.
* `502 Bad Gateway` when the price download stage raises a cache or data source
  error.
* `500 Internal Server Error` when the metric computation itself raises an
  unexpected exception.  The response still follows the `{"detail": "..."}`
  contract described below.

## `/healthz`

Return process-level diagnostics and background task state. The payload mirrors
the structure returned by `highest_volatility.app.api.healthz`. A healthy
response includes `{"status": "ok"}` while degraded Redis/cache refresh tasks
surface `status="error"` alongside contextual details.

## `/readyz`

Signal that dependencies such as Redis are reachable and caches are primed. The
endpoint returns `200 OK` when FastAPI Cache uses Redis and the background task
is running. Otherwise it returns `503 Service Unavailable` with diagnostic
fields describing the failing component.

## Error handling

All endpoints delegate to `_handle_error`, which maps internal `HVError`
subclasses to HTTP status codes:

| Error code           | HTTP status |
| -------------------- | ----------- |
| `validation`         | `400 Bad Request` |
| `data_source`        | `502 Bad Gateway` |
| `integration`        | `502 Bad Gateway` |
| `cache`              | `503 Service Unavailable` |
| `compute` / `config` | `500 Internal Server Error` |

The FastAPI exception handler serialises messages as `{"detail": "<user_message>"}`.
A global rate limiter applies the configured policy and returns `429 Too Many
Requests` with the default SlowAPI payload `{"detail": "Rate limit exceeded"}`.

## Configuration reference

Runtime behaviour is controlled through environment variables exposed via
`highest_volatility.app.api.Settings`.  Set variables with the `HV_` prefix to
override defaults:

| Variable                     | Default value        | Description |
| ---------------------------- | -------------------- | ----------- |
| `HV_LOOKBACK_DAYS`           | `252`                | Default `lookback_days` for `/prices` and `/metrics` (bounds: 30–2000).
| `HV_INTERVAL`                | `"1d"`               | Default Yahoo Finance interval used during downloads.
| `HV_PREPOST`                 | `false`              | Include pre/post-market candles in price fetches.
| `HV_TOP_N`                   | `100`                | Default Fortune rank limit for `/universe`.
| `HV_METRIC`                  | `"cc_vol"`          | Default metric key (Close-to-Close Volatility) when `/metrics` omits `metric`.
| `HV_MIN_DAYS`                | `126`                | Minimum observations for metric calculations (bounds: 10–lookback).
| `HV_REDIS_URL`               | `redis://localhost:6379/0` | Connection string for the Redis cache backend used by FastAPI Cache.
| `HV_CACHE_TTL_UNIVERSE`      | `60`                 | Cache TTL (seconds) for `/universe` responses.
| `HV_CACHE_TTL_PRICES`        | `60`                 | Cache TTL (seconds) for `/prices` responses.
| `HV_CACHE_TTL_METRICS`       | `60`                 | Cache TTL (seconds) for `/metrics` responses.
| `HV_RATE_LIMIT`              | `"60/minute"`       | Default SlowAPI rate-limit applied per client.
| `HV_CACHE_REFRESH_INTERVAL`  | `86400`              | Delay (seconds) between background cache refresh runs.
| `HV_ANNOTATIONS_DB`          | `cache/annotations.db` | SQLite path for persisted annotation notes.

Redis must be reachable at `HV_REDIS_URL` during startup so that
`FastAPICache.init` succeeds.  When deploying in containerised environments
configure the URL to point at the managed Redis instance (for example,
`redis://:password@redis.internal:6379/0`).

The FastAPI process inherits additional CLI defaults from `highest_volatility.app.cli`.
Any change to these environment variables requires an application restart to
ensure settings are reloaded.
