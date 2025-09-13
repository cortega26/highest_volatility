# Highest Volatility

Tools for exploring equity price volatility.  The project includes utilities for
loading and caching price history.

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
