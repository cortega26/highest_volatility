#!/usr/bin/env python3
"""CLI to launch periodic cache refresh."""

from __future__ import annotations

import argparse
import asyncio

from highest_volatility.pipeline.cache_refresh import schedule_cache_refresh


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh cached price data periodically")
    parser.add_argument("--interval", default="1d", help="Price interval to refresh (default: 1d)")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Number of lookback days for each ticker (default: 365)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=60 * 60 * 24,
        help="Seconds between refresh runs (default: one day)",
    )
    args = parser.parse_args()

    asyncio.run(
        schedule_cache_refresh(
            interval=args.interval, lookback_days=args.lookback_days, delay=args.delay
        )
    )


if __name__ == "__main__":
    main()
