# Highest Volatility

Tools for exploring equity price volatility.  The project includes utilities for
loading and caching price history.

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
