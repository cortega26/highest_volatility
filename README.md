# Highest Volatility

Tools for exploring equity price volatility.  The project includes utilities for
loading and caching price history.

## Data API

A lightweight FastAPI service exposes cached price data and the Fortune ticker
universe. Build and run it with Docker:

```bash
docker build -t hv-api .
docker run -p 8000:8000 hv-api
```

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
tickers.  It iterates over the tickers under `.cache/prices/<interval>` and
updates each one sequentially.

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
