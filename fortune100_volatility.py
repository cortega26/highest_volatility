#!/usr/bin/env python3
"""
Find the 5 most volatile stocks among the current Fortune 100 (by revenue).

What this script does
---------------------
1) Scrapes a live Fortune 500 table (with tickers) and extracts the top 100 entries (Fortune 100).
   - Source used: https://us500.com/fortune-500-companies
   - The page includes "Rank, Company, Ticker, ..." in a single HTML table.
2) Cleans the tickers (e.g., converts class-share dots like 'BRK.A' to Yahoo Finance's 'BRK-A').
3) Filters to publicly traded tickers (rows with missing tickers are discarded).
4) Downloads daily price history for the Fortune 100 tickers using yfinance.
5) Computes annualized historical volatility (standard deviation of daily log returns * sqrt(252))
   over a configurable lookback window (default: last 252 trading days, ~1 year).
6) Ranks by volatility, prints the top 5, and saves the full table to CSV.

Usage
-----
$ python fortune100_volatility.py
$ python fortune100_volatility.py --lookback-days 252 --top-n 100 --print-top 5 \
    --output-csv fortune100_vols.csv --min-days 126

Dependencies
------------
- pandas
- numpy
- yfinance
- requests
- lxml or html5lib (for pandas.read_html), and beautifulsoup4 (optional but recommended)

Install:
  pip install pandas numpy yfinance requests lxml beautifulsoup4

Notes
-----
- Some Fortune 100 companies are private or otherwise lack tradable tickers; they are skipped automatically.
- Fannie Mae / Freddie Mac trade OTC (FNMA / FMCC); yfinance can fetch them, but data quality can vary.
- If the source HTML structure changes, the scraper includes reasonable fallbacks and helpful error messages.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError as e:
    print("yfinance is required. Install with: pip install yfinance", file=sys.stderr)
    raise

# -------- Configuration defaults --------
DEFAULT_SOURCE_URL = "https://us500.com/fortune-500-companies"
DEFAULT_LOOKBACK_DAYS = 252            # ~1Y of trading days
DEFAULT_TOP_N = 100                    # Fortune 100
DEFAULT_PRINT_TOP = 5                  # show top 5 by volatility
DEFAULT_MIN_DAYS = 126                 # require at least ~6 months of data
DEFAULT_OUTPUT_CSV = "fortune100_volatility.csv"
REQUESTS_TIMEOUT = 30                  # seconds


@dataclass
class FortuneEntry:
    rank: int
    company: str
    raw_ticker: str
    cleaned_ticker: str


def _normalize_ticker_for_yahoo(ticker: str) -> str:
    """
    Normalize a US ticker to Yahoo Finance format:
    - Replace class share dots with dashes (e.g., BRK.A -> BRK-A ; BRK.B -> BRK-B ; BF.B -> BF-B).
    - Trim spaces and uppercase.
    """
    if not isinstance(ticker, str):
        return ""
    t = ticker.strip().upper()
    if not t:
        return t
    # Common transformation: class shares use '-' on Yahoo
    t = t.replace(".", "-")
    return t


def fetch_fortune_table(source_url: str) -> pd.DataFrame:
    """
    Fetch the HTML from the source URL and return the first table
    that contains the expected columns for Rank/Company/Ticker.

    We use pandas.read_html because the source is a straightforward HTML table.
    """
    # Use requests to retrieve the content explicitly (so we can set headers and check status)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FortuneVolatilityBot/1.0; +https://example.com)"
    }
    resp = requests.get(source_url, headers=headers, timeout=REQUESTS_TIMEOUT)
    resp.raise_for_status()

    # pandas.read_html can directly parse raw HTML
    tables = pd.read_html(resp.text, flavor="lxml")
    if not tables:
        raise RuntimeError("No HTML tables found on the source page.")

    # Identify the relevant table by presence of required columns (case-insensitive)
    wanted = {"rank", "company", "ticker"}
    for df in tables:
        cols_norm = {str(c).strip().lower() for c in df.columns}
        if wanted.issubset(cols_norm):
            return df

    # If not found directly, try looser matching by renaming columns
    # and picking the widest table
    best = max(tables, key=lambda d: d.shape[1])
    return best


def extract_fortune100_with_tickers(table: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> List[FortuneEntry]:
    """
    From the parsed table, extract (rank, company, ticker) for the first top_n rows.

    The function is resilient to different column label variants by normalizing names.
    Rows with missing or blank tickers are dropped (private / non-traded).
    """
    # Normalize column names
    df = table.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Try to find likely columns
    # We support variants such as 'Rank', 'Company', 'Ticker', case-insensitive and with spaces.
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if "rank" == cl:
            colmap["rank"] = c
        elif cl in ("company", "company name", "name"):
            colmap["company"] = c
        elif cl == "ticker":
            colmap["ticker"] = c

    # If the exact columns aren't available, try to infer by heuristics
    if "rank" not in colmap:
        # find a numeric-like column named 'Rank' in content
        candidates = [c for c in df.columns if "rank" in c.lower()]
        if candidates:
            colmap["rank"] = candidates[0]
    if "company" not in colmap:
        candidates = [c for c in df.columns if "company" in c.lower()
                      or "name" in c.lower()]
        if candidates:
            colmap["company"] = candidates[0]
    if "ticker" not in colmap:
        candidates = [c for c in df.columns if "ticker" in c.lower()]
        if candidates:
            colmap["ticker"] = candidates[0]

    missing = [k for k in ("rank", "company", "ticker") if k not in colmap]
    if missing:
        raise RuntimeError(
            f"Could not find required columns in source table. Missing: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only needed columns and drop rows with NaNs in rank/company
    slim = df[[colmap["rank"], colmap["company"], colmap["ticker"]]].copy()
    slim.columns = ["Rank", "Company", "Ticker"]

    # Coerce rank numeric and sort
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slim["Rank"] = pd.to_numeric(slim["Rank"], errors="coerce")

    slim = slim.dropna(subset=["Rank", "Company"]).sort_values("Rank")
    slim = slim.head(top_n)

    # Drop entries without a usable ticker
    slim["Ticker"] = slim["Ticker"].astype(str).str.strip()
    slim = slim[slim["Ticker"].str.len() > 0]
    slim = slim[~slim["Ticker"].str.contains(
        "^-$|^N/A$", case=False, regex=True)]

    # Build entries with cleaned tickers
    entries: List[FortuneEntry] = []
    for _, row in slim.iterrows():
        rank = int(row["Rank"])
        company = str(row["Company"]).strip()
        raw_ticker = str(row["Ticker"]).strip()
        cleaned = _normalize_ticker_for_yahoo(raw_ticker)
        if cleaned:
            entries.append(FortuneEntry(rank=rank, company=company,
                           raw_ticker=raw_ticker, cleaned_ticker=cleaned))

    # Deduplicate by cleaned ticker, keeping the best (lowest) rank if duplicate appears
    seen = {}
    for e in sorted(entries, key=lambda x: x.rank):
        if e.cleaned_ticker not in seen:
            seen[e.cleaned_ticker] = e
    return list(seen.values())


def download_prices(tickers: List[str], lookback_days: int) -> pd.DataFrame:
    """
    Download daily OHLCV for given tickers and return a MultiIndex DataFrame as returned by yfinance.
    Uses start date set to lookback_days*1.5 to increase chance of getting enough trading days, then trims to last N days.
    """
    if not tickers:
        raise ValueError("No tickers supplied to download_prices().")

    # yfinance period supports strings like '1y', but to be precise with N trading days,
    # we request ~1.5x days in calendar terms and then trim later.
    # Approximate conversion: assume ~252 trading days/year => ~365 calendar days. Scale accordingly.
    # We'll add a safety factor of 2.0 to be generous, then trim.
    approx_calendar_days = int(math.ceil(lookback_days * 2.0 * 365.0 / 252.0))
    period_str = f"{approx_calendar_days}d"

    # Download all tickers at once for speed
    data = yf.download(
        tickers=tickers,
        period=period_str,
        interval="1d",
        auto_adjust=False,      # keep Adj Close column available
        group_by="column",      # columns: top level is OHLCV field
        threads=True,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(
            "yfinance returned an empty dataset for the requested tickers/period.")

    # Keep only the last 'lookback_days' rows
    data = data.tail(lookback_days + 2)  # +2 to provide shift/return buffer
    return data


def build_price_matrix(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Given the yfinance multiindex DataFrame and the desired tickers (Yahoo-normalized),
    return a price matrix (rows = dates, columns = tickers) using 'Adj Close' when available
    else falling back to 'Close' per ticker. Columns missing data are dropped here and
    will be checked against min days later.
    """
    # Ensure MultiIndex columns
    if not isinstance(data.columns, pd.MultiIndex):
        # Single ticker download returns columns as simple Index.
        # In that case, build a 1-column matrix.
        if "Adj Close" in data.columns:
            prices = data[["Adj Close"]].rename(
                columns={"Adj Close": tickers[0]})
        elif "Close" in data.columns:
            prices = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            raise RuntimeError(
                "Downloaded data missing 'Adj Close' and 'Close' columns.")
        return prices

    fields = set(data.columns.get_level_values(0))
    use_field = "Adj Close" if "Adj Close" in fields else (
        "Close" if "Close" in fields else None)
    if not use_field:
        raise RuntimeError(
            f"Downloaded data missing 'Adj Close'/'Close'. Available: {sorted(fields)}")

    # Slice to the chosen field -> a wide (dates x tickers) frame
    wide = data[use_field].copy()

    # Keep only requested tickers (some may not be present due to failed fetches)
    cols_present = [c for c in tickers if c in wide.columns]
    if not cols_present:
        raise RuntimeError(
            "None of the requested tickers are present in the downloaded dataset.")
    wide = wide[cols_present]

    # Drop columns with all-NaN
    wide = wide.dropna(axis=1, how="all")

    # Sort columns for reproducibility
    wide = wide.sort_index(axis=1)
    return wide


def compute_annualized_volatility(price_wide: pd.DataFrame, min_days: int = DEFAULT_MIN_DAYS) -> pd.DataFrame:
    """
    Compute annualized volatility for each column (ticker) in the price_wide matrix.
    Volatility metric: std dev of daily log returns * sqrt(252)
    - Requires at least min_days of non-NaN returns for a ticker to be included.
    Returns a DataFrame with columns: ['ticker', 'n_obs', 'daily_std', 'ann_vol']
    """
    # Compute daily log returns
    returns = np.log(price_wide / price_wide.shift(1))
    returns = returns.dropna(how="all")

    results = []
    for t in returns.columns:
        s = returns[t].dropna()
        n = s.shape[0]
        if n < min_days:
            continue
        daily_std = float(s.std(ddof=1))
        ann_vol = daily_std * math.sqrt(252.0)
        results.append((t, n, daily_std, ann_vol))

    out = pd.DataFrame(results, columns=[
                       "ticker", "n_obs", "daily_std", "ann_vol"])
    out = out.sort_values("ann_vol", ascending=False).reset_index(drop=True)
    return out


def attach_company_names(vol_df: pd.DataFrame, entries: List[FortuneEntry]) -> pd.DataFrame:
    """
    Add 'company' and 'rank' columns to the volatility table using the FortuneEntry mapping.
    """
    mapping_name: Dict[str, str] = {
        e.cleaned_ticker: e.company for e in entries}
    mapping_rank: Dict[str, int] = {e.cleaned_ticker: e.rank for e in entries}
    vol_df["company"] = vol_df["ticker"].map(mapping_name)
    vol_df["fortune_rank"] = vol_df["ticker"].map(mapping_rank)
    # Reorder columns for readability
    cols = ["fortune_rank", "company", "ticker",
            "n_obs", "daily_std", "ann_vol"]
    return vol_df[cols].sort_values(["ann_vol", "fortune_rank"], ascending=[False, True]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute the 5 most volatile stocks among the Fortune 100.")
    parser.add_argument("--source-url", type=str, default=DEFAULT_SOURCE_URL,
                        help=f"URL to scrape the Fortune 500 table (default: {DEFAULT_SOURCE_URL})")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help=f"Trading-day lookback window for volatility computation (default: {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"How many top-ranked companies to consider (default: {DEFAULT_TOP_N})")
    parser.add_argument("--print-top", type=int, default=DEFAULT_PRINT_TOP,
                        help=f"How many of the highest-volatility tickers to print (default: {DEFAULT_PRINT_TOP})")
    parser.add_argument("--min-days", type=int, default=DEFAULT_MIN_DAYS,
                        help=f"Minimum number of return observations required for a ticker (default: {DEFAULT_MIN_DAYS})")
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV,
                        help=f"Where to save the full volatility table (default: {DEFAULT_OUTPUT_CSV})")
    args = parser.parse_args()

    # --- Step 1: Scrape Fortune 500 page and extract Fortune 100 with tickers
    print(f"[1/5] Fetching Fortune 500 table from: {args.source_url}")
    try:
        table = fetch_fortune_table(args.source_url)
    except Exception as e:
        print(
            f"ERROR: Failed to fetch/parse source table: {e}", file=sys.stderr)
        return 2

    print(f"[2/5] Extracting top {args.top_n} entries and cleaning tickers...")
    try:
        entries = extract_fortune100_with_tickers(table, top_n=args.top_n)
    except Exception as e:
        print(
            f"ERROR: Failed to extract Fortune {args.top_n} with tickers: {e}", file=sys.stderr)
        return 2

    if not entries:
        print("ERROR: No tradable tickers found among the requested Fortune entries.", file=sys.stderr)
        return 2

    # Build lists for downstream processing
    tickers = [e.cleaned_ticker for e in entries]
    rank_by_ticker = {e.cleaned_ticker: e.rank for e in entries}
    company_by_ticker = {e.cleaned_ticker: e.company for e in entries}

    print(
        f"    Found {len(tickers)} tickers with symbols. Examples: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")

    # --- Step 2: Download prices
    print(
        f"[3/5] Downloading ~{args.lookback_days} trading days of daily prices for {len(tickers)} tickers via yfinance...")
    start_t = time.time()
    try:
        raw = download_prices(tickers, lookback_days=args.lookback_days)
    except Exception as e:
        print(f"ERROR: Price download failed: {e}", file=sys.stderr)
        return 2
    elapsed = time.time() - start_t
    print(f"    Download completed in {elapsed:.1f}s. Data shape: {raw.shape}")

    # --- Step 3: Build adjusted close matrix
    print("[4/5] Building price matrix (Adj Close if available, else Close), trimming and cleaning...")
    try:
        price_wide = build_price_matrix(raw, tickers)
    except Exception as e:
        print(f"ERROR: Failed to build price matrix: {e}", file=sys.stderr)
        return 2

    # --- Step 4: Compute volatility
    print(
        f"[5/5] Computing annualized volatility (std(log-returns) * sqrt(252)); requiring at least {args.min_days} observations...")
    vol = compute_annualized_volatility(price_wide, min_days=args.min_days)
    if vol.empty:
        print(
            "ERROR: No tickers had sufficient data to compute volatility.", file=sys.stderr)
        return 2

    vol = attach_company_names(vol, entries)

    # Save results
    out_csv = args.output_csv
    vol.to_csv(out_csv, index=False)
    print(f"\nSaved full results to: {out_csv}")

    # Display top N by volatility
    top_k = min(args.print_top, vol.shape[0])
    print(f"\nTop {top_k} highest-volatility Fortune {args.top_n} *stocks* (lookback={args.lookback_days} trading days):\n")
    to_show = vol.head(top_k).copy()
    # Pretty print
    with pd.option_context("display.max_colwidth", 80, "display.width", 120):
        print(to_show.to_string(index=False,
                                formatters={
                                    "fortune_rank": "{:>3d}".format,
                                    "n_obs": "{:>4d}".format,
                                    "daily_std": "{:.4f}".format,
                                    "ann_vol": "{:.4f}".format,
                                }))

    # Helpful note about any skipped companies (no ticker)
    total_requested = args.top_n
    total_used = len(set(price_wide.columns))
    total_found = len(tickers)
    if total_used < total_found:
        skipped = set(tickers) - set(price_wide.columns)
        if skipped:
            print(
                f"\nNote: {len(skipped)} tickers were found but lacked usable price data in the window and were skipped.")
            # Uncomment if you want to list them:
            # print("Skipped due to missing data:", ", ".join(sorted(skipped)))

    missing_tickers = total_requested - len(entries)
    if missing_tickers > 0:
        print(f"Note: {missing_tickers} of the top {total_requested} Fortune entries appear to be private / have no ticker and were excluded.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
