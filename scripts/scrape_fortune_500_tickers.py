from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highest_volatility.ingest.tickers import fetch_fortune_tickers
from highest_volatility.storage.csv_store import save_csv

OUTPUT_FILE = ROOT / "fortune500_tickers.csv"


def main() -> None:
    table = fetch_fortune_tickers(top_n=500)
    save_csv(table, OUTPUT_FILE)


if __name__ == "__main__":
    main()
