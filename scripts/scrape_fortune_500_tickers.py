from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highest_volatility.ingest.tickers import fetch_fortune_tickers  # noqa: E402
from highest_volatility.storage.ticker_cache import (  # noqa: E402
    CACHE_PATH,
    save_cached_fortune,
)
from highest_volatility.storage.csv_store import save_csv  # noqa: E402

OUTPUT_FILE = ROOT / "fortune500_tickers.csv"


def main() -> None:
    previous: set[str] = set()
    if OUTPUT_FILE.exists():
        try:
            previous = set(pd.read_csv(OUTPUT_FILE)["ticker"])
        except Exception as e:
            print(f"Warning: failed to read existing ticker list: {e}")

    table = fetch_fortune_tickers(top_n=500)
    current = set(table["ticker"])

    added = sorted(current - previous)
    removed = sorted(previous - current)
    if added or removed:
        print(f"Ticker changes detected. Added: {added or 'none'}, Removed: {removed or 'none'}")
    else:
        print("No changes detected in Fortune 500 tickers")

    save_csv(table, OUTPUT_FILE)
    # Also save to local cache path used by the CLI
    try:
        save_cached_fortune(table)
        print(f"Saved ticker cache to {CACHE_PATH}")
    except Exception as e:
        print(f"Warning: failed to save local ticker cache: {e}")


if __name__ == "__main__":
    main()
