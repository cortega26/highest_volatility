# Highest Volatility

Tools for exploring equity price volatility.  The project includes utilities for
loading and caching price history.

## Data API

A lightweight FastAPI service exposes cached price data and the Fortune ticker
universe. Build and run it with Docker:

```bash
docker build -t hv-api .
docker run --rm -p 8000:8000 hv-api
```

Runtime notes:

- The container publishes the API on port ``8000``; map it to a host port as
  needed (for example ``-p 8000:8000``).
- ``PYTHONPATH`` is set to ``/app/src`` so ``highest_volatility`` can be
  imported without installing the package separately.
- Override any FastAPI settings with ``HV_``-prefixed environment variables.
  Common examples include ``HV_REDIS_URL`` (default ``redis://localhost:6379/0``)
  and ``HV_CACHE_REFRESH_INTERVAL``.

The service provides two endpoints:

- ``/prices/{ticker}`` – return cached prices for ``ticker``. Use the ``fmt``
  query parameter to request ``json`` (default) or ``parquet`` bytes.
- ``/fortune-tickers`` – return the cached Fortune 500 ticker list.

### Validation Notes

Cached Fortune tables may contain raw tickers that use punctuation such as dots
(``BRK.B``). The universe builder now preserves those raw strings while carrying
their normalized Yahoo Finance representation (``BRK-B``). Downstream alignment
and ranking operate on the normalized variant so that cached ``rank`` and
``company`` metadata are retained without re-enumerating positions. The
``tests/test_universe_rank_alignment.py`` coverage exercises this code path by
loading cached data with dotted tickers and ensuring the original ranks remain
intact.

Incremental price refreshes now re-request the final cached session for
intraday intervals (``1m``/``1h`` variants) so late-published bars within the
same trading day are merged correctly. The synchronous and asynchronous
fetchers share this behavior, with ``tests/test_price_fetcher.py`` verifying the
intraday cache replay alongside the existing daily interval regression.

### Batch Price Downloads

The synchronous ``download_price_history`` helper downloads tickers in chunks of
40 symbols. Each chunk is submitted to a thread pool so multiple batches can be
in-flight when ``max_workers`` is greater than one. A failed chunk will fall
back to sequential single-ticker retries inside its worker without blocking
other threads. Tune ``max_workers`` based on available CPU capacity and the
latency of your network connection; values between 4 and 8 work well for most
desktop environments. Introduce a short ``chunk_sleep`` (for example, ``0.5``
seconds) after each batch if the upstream API begins throttling requests.

Client utilities such as ``cache.store`` will hydrate missing local cache files
from this API when the ``HV_API_BASE_URL`` environment variable is set.

## Metrics

The command line interface exposes a number of built-in metrics that can be
selected with the ``--metric`` option:

- ``cc_vol`` – close-to-close annualised volatility
- ``parkinson_vol`` – Parkinson's high-low estimator
- ``gk_vol`` – Garman–Klass volatility
- ``rs_vol`` – Rogers–Satchell volatility
- ``yz_vol`` – Yang–Zhang volatility
- ``ewma_vol`` – exponentially weighted moving average volatility
- ``mad_vol`` – median absolute deviation volatility
- ``sharpe_ratio`` – annualised Sharpe ratio
- ``max_drawdown`` – maximum drawdown
- ``var`` – value at risk (VaR)
- ``sortino`` – annualised Sortino ratio

## CLI Internals & Rollback Plan

The CLI now orchestrates four focused steps – universe construction, price
downloads (with sanitisation via ``highest_volatility.app.sanitization``),
metric computation, and result rendering. Each step returns a small data class
so the workflow can be unit tested in isolation and downstream tooling can
reuse the intermediate structures.

To roll back to the previous monolithic ``main`` implementation, restore
``src/highest_volatility/app/cli.py`` from the last release tag or a specific
commit and remove ``src/highest_volatility/app/sanitization.py``. For example,
``git checkout <prior-tag> -- src/highest_volatility/app/cli.py`` followed by
``git rm src/highest_volatility/app/sanitization.py`` reverts the refactor
while keeping the rest of the repository unchanged.

## Streamlit App

An interactive dashboard is available via Streamlit. It mirrors the CLI
defaults, letting you pick the lookback window, interval, metric, and minimum
observation count directly from the sidebar. Additional toggles allow skipping
Selenium validation of the Fortune universe and opting into asynchronous price
fetching.

Set up and launch the UI with:

```bash
pip install -r requirements.txt
streamlit run src/highest_volatility/app/streamlit_app.py
```

The Streamlit entry point injects the repository's ``src`` directory onto
``PYTHONPATH`` when launched directly, so you can run it without an editable
install. Installing with ``pip install -e .`` remains a convenient alternative
if you prefer relying on standard packaging workflows.

The Streamlit app reads from the on-disk cache created by the CLI and API
utilities. Populate it ahead of time by running your preferred cache refresh
flow (for example, ``python scripts/refresh_cache.py``) or by hitting the FastAPI
service.

When the ``HV_API_BASE_URL`` environment variable is defined, the UI will source
missing price history directly from the FastAPI backend instead of the local
cache. Unset the variable to operate entirely offline.

Results are displayed as a sortable table with warning banners when price data
is unavailable for the selected configuration.

## Cache Refresh

A background task runs when the API starts, periodically refreshing cached
price data for any locally stored tickers. The interval between refreshes is
configured with the ``HV_CACHE_REFRESH_INTERVAL`` environment variable and
defaults to once every 24 hours.

A helper script is provided to refresh cached price data for all locally stored
tickers. It iterates over the tickers under `cache/prices/<interval>` and
updates each one sequentially. Each ticker's Parquet file and JSON manifest
live side by side in that directory (for example, `AAPL.parquet` and
`AAPL.json`), matching the `_paths` helper in `src/cache/store.py`.

Run the scheduler from the repository root:

```bash
python scripts/refresh_cache.py
```

By default the script refreshes once per day.  To run it on a regular schedule
using cron, add an entry similar to:

```
0 0 * * * cd /path/to/repo && /usr/bin/python scripts/refresh_cache.py >> refresh.log 2>&1
```

Adjust the interval and paths as needed.

## Data Sources

A monthly GitHub Action refreshes the Fortune 500 universe. On the first day of
each month the workflow runs ``scripts/scrape_fortune_500_tickers.py`` and
commits the resulting ``fortune500_tickers.csv`` to the ``data`` branch. These
tickers serve as the basis for downstream analyses and API responses.
