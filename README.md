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
