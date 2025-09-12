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
import io
import os
from urllib.parse import urlparse

# Optional Selenium imports are performed lazily in code paths that need them.
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import json

try:
    import yfinance as yf
except ImportError as e:
    print("yfinance is required. Install with: pip install yfinance", file=sys.stderr)
    raise

# -------- Configuration defaults --------
DEFAULT_SOURCE_URL = "https://us500.com/fortune-500-companies"
DEFAULT_LOOKBACK_DAYS = 252            # ~1Y of trading days
DEFAULT_TOP_N = 200                    # Fortune 200
DEFAULT_PRINT_TOP = 10                  # show top 5 by volatility
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


def _debug_print(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[DEBUG] {msg}")


def _selenium_available() -> bool:
    try:
        import selenium  # type: ignore
        import webdriver_manager  # type: ignore
        return True
    except Exception:
        return False

def fetch_fortune_table(
    source_url: str,
    *,
    debug: bool = False,
    use_selenium: bool = False,
    selenium_timeout: int = 20,
    source_file: Optional[str] = None,
    desired_count: int = DEFAULT_TOP_N,
    selenium_headless: bool = True,
) -> pd.DataFrame:
    """
    Fetch the HTML from the source URL and return the first table
    that contains the expected columns for Rank/Company/Ticker.

    We use pandas.read_html because the source is a straightforward HTML table.
    """
    # Use requests to retrieve the content explicitly (so we can set headers and check status)
    # If a local source file is provided, parse it directly (no network)
    if source_file:
        _debug_print(debug, f"Parsing local source file: {source_file}")
        if not os.path.exists(source_file):
            raise RuntimeError(f"--source-file not found: {source_file}")
        ext = os.path.splitext(source_file)[1].lower()
        if ext in (".csv", ".tsv"):
            sep = "," if ext == ".csv" else "\t"
            df = pd.read_csv(source_file, sep=sep)
            _debug_print(
                debug, f"Loaded CSV/TSV with shape={df.shape} and columns={df.columns.tolist()}")
            return df
        elif ext in (".html", ".htm"):
            with open(source_file, "r", encoding="utf-8") as fh:
                html_text = fh.read()
            html_io = io.StringIO(html_text)
            tables: List[pd.DataFrame] = []
            for flavor in ("lxml", "bs4"):
                try:
                    _debug_print(
                        debug, f"Local HTML: attempting pandas.read_html with flavor='{flavor}'")
                    tables = pd.read_html(html_io, flavor=flavor)
                    _debug_print(
                        debug, f"Local HTML: flavor='{flavor}' yielded {len(tables)} tables")
                    if tables:
                        break
                except Exception as e:
                    _debug_print(
                        debug, f"Local HTML parse error with flavor='{flavor}': {e}")
                finally:
                    html_io.seek(0)
            if not tables:
                raise RuntimeError(
                    "No HTML tables found in --source-file HTML")
            # Select best table below
        else:
            raise RuntimeError(
                f"--source-file must be .csv, .tsv, .html or .htm (got {ext})")
    else:
        tables = None

    # Try only the requested URL. Cut failing fallback sources to narrow surface area.
    candidate_urls = [source_url]

    headers = {
        # A realistic desktop UA reduces chance of basic bot-blocks
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    last_err: Optional[Exception] = None
    for idx, url in enumerate(candidate_urls, start=1):
        _debug_print(
            debug, f"Fetching URL [{idx}/{len(candidate_urls)}]: {url}")
        try:
            resp = requests.get(url, headers=headers,
                                timeout=REQUESTS_TIMEOUT, allow_redirects=True)
        except Exception as e:
            last_err = e
            _debug_print(debug, f"Request error for {url}: {e}")
            continue

        # Log redirect chain and final URL
        if resp.history:
            chain = " -> ".join(
                f"{r.status_code}:{r.headers.get('Location', '')}" for r in resp.history)
            _debug_print(debug, f"Redirect history: {chain}")
        _debug_print(
            debug, f"Final URL: {resp.url} | Status: {resp.status_code}")

        # If HTTP error, record and continue to next candidate
        if not resp.ok:
            last_err = requests.HTTPError(
                f"HTTP {resp.status_code} for {resp.url}")
            _debug_print(
                debug, f"Non-OK response. Content-Type: {resp.headers.get('Content-Type')} | Length: {len(resp.text)}")
            # Save body for inspection when debugging
            if debug:
                parsed = urlparse(resp.url)
                fname = f"debug_fetch_{idx}_{parsed.netloc.replace('.', '_')}.html"
                try:
                    with open(fname, 'w', encoding='utf-8') as fh:
                        fh.write(resp.text)
                    _debug_print(debug, f"Saved response body to {fname}")
                except Exception as save_err:
                    _debug_print(
                        debug, f"Failed to save response body: {save_err}")
            continue

        # Basic heuristics to understand page content
        body_lower = resp.text.lower()
        n_tables = body_lower.count("<table")
        n_scripts = body_lower.count("<script")
        _debug_print(
            debug, f"Body length: {len(resp.text)} | <table> tags: {n_tables} | <script> tags: {n_scripts}")
        _debug_print(
            debug, f"Response headers: Content-Type={resp.headers.get('Content-Type')} | Server={resp.headers.get('Server')}")

        # Detect common bot-block/JS-challenge pages (Cloudflare, etc.)
        if ("attention required" in body_lower and "cloudflare" in body_lower) or "cf-browser-verification" in body_lower:
            last_err = RuntimeError(
                f"Access to {url} appears blocked by anti-bot (Cloudflare)")
            _debug_print(debug, f"Detected anti-bot page for {url}; skipping")
            continue

        # Save the page for inspection when debugging
        if debug:
            parsed = urlparse(resp.url)
            fname = f"debug_fetch_{idx}_{parsed.netloc.replace('.', '_')}.html"
            try:
                with open(fname, 'w', encoding='utf-8') as fh:
                    fh.write(resp.text)
                _debug_print(debug, f"Saved response body to {fname}")
            except Exception as save_err:
                _debug_print(
                    debug, f"Failed to save response body: {save_err}")

        # Parse tables using pandas
        html_io = io.StringIO(resp.text)
        tables = []
        for flavor in ("lxml", "bs4"):
            try:
                _debug_print(
                    debug, f"Attempting pandas.read_html with flavor='{flavor}'")
                tables = pd.read_html(html_io, flavor=flavor)
                _debug_print(
                    debug, f"flavor='{flavor}' yielded {len(tables)} tables")
                if tables:
                    break
            except ValueError as e:
                last_err = e
                _debug_print(
                    debug, f"pandas.read_html ValueError for flavor='{flavor}': {e}")
            except Exception as e:
                last_err = e
                _debug_print(
                    debug, f"Unexpected error with flavor='{flavor}': {e}")
            finally:
                html_io.seek(0)

        # If still no tables, attempt Next.js data API fallback
        if not tables:
            try:
                df_next = _try_nextjs_data_api(
                    url,
                    resp.text,
                    headers=headers,
                    debug=debug,
                    desired_count=desired_count,
                )
                if df_next is not None and not df_next.empty:
                    _debug_print(
                        debug, f"Next.js data API extraction produced shape={df_next.shape} with columns={df_next.columns.tolist()}")
                    tables = [df_next]
                else:
                    _debug_print(
                        debug, "Next.js data API: no suitable data found")
            except Exception as e:
                last_err = e
                _debug_print(debug, f"Next.js data API extraction error: {e}")

        # If we have fewer than desired rows, try scraping API endpoints from the JS chunk
        if (not tables) or (tables and tables[0].shape[0] < desired_count):
            try:
                api_df = _try_us500_chunk_api(
                    url, resp.text, headers=headers, debug=debug, desired_count=desired_count)
                if api_df is not None and not api_df.empty:
                    _debug_print(
                        debug, f"Chunk API extraction produced shape={api_df.shape} with columns={api_df.columns.tolist()}")
                    tables = [api_df]
            except Exception as e:
                last_err = e
                _debug_print(debug, f"Chunk API extraction error: {e}")

        # As a last resort, if Selenium is allowed or available, scrape page=1.. via DOM and stitch
        if (use_selenium or _selenium_available()) and ((not tables) or (tables and tables[0].shape[0] < desired_count)):
            try:
                # Heuristic max_pages based on desired_count
                max_pages = max(4, int(math.ceil(desired_count / 50.0)) + 1)
                dom_df = _scrape_us500_with_selenium_pages(
                    url,
                    debug=debug,
                    timeout=selenium_timeout,
                    max_pages=max_pages,
                    target_count=desired_count,
                    headless=selenium_headless,
                )
                if dom_df is not None and not dom_df.empty:
                    _debug_print(
                        debug, f"Selenium DOM multi-page scrape produced shape={dom_df.shape}")
                    tables = [dom_df]
            except Exception as e:
                last_err = e
                _debug_print(
                    debug, f"Selenium DOM multi-page scrape failed: {e}")

        if tables:
            # Found tables for this URL; proceed to select the relevant one below
            break

        # If no tables with requests and selenium is enabled, try JS-rendered fetch
        if use_selenium:
            _debug_print(
                debug, f"No tables found via requests for {url}. Trying Selenium (timeout={selenium_timeout}s)...")
            try:
                html = _fetch_html_with_selenium(
                    url, debug=debug, timeout=selenium_timeout, headless=selenium_headless)
                _debug_print(
                    debug, f"Selenium fetched HTML length: {len(html)}")
                if debug:
                    parsed = urlparse(url)
                    fname = f"debug_fetch_selenium_{idx}_{parsed.netloc.replace('.', '_')}.html"
                    try:
                        with open(fname, 'w', encoding='utf-8') as fh:
                            fh.write(html)
                        _debug_print(
                            debug, f"Saved Selenium page source to {fname}")
                    except Exception as save_err:
                        _debug_print(
                            debug, f"Failed to save Selenium HTML: {save_err}")
                html_io2 = io.StringIO(html)
                for flavor in ("lxml", "bs4"):
                    try:
                        _debug_print(
                            debug, f"Selenium: attempting pandas.read_html with flavor='{flavor}'")
                        tables = pd.read_html(html_io2, flavor=flavor)
                        _debug_print(
                            debug, f"Selenium: flavor='{flavor}' yielded {len(tables)} tables")
                        if tables:
                            break
                    except ValueError as e:
                        last_err = e
                        _debug_print(
                            debug, f"Selenium: ValueError for flavor='{flavor}': {e}")
                    except Exception as e:
                        last_err = e
                        _debug_print(
                            debug, f"Selenium: Unexpected error with flavor='{flavor}': {e}")
                    finally:
                        html_io2.seek(0)
                # If still nothing, try div-based table extraction heuristics (data-label attributes)
                if not tables:
                    _debug_print(
                        debug, "Selenium: attempting div-based table extraction using data-label attributes")
                    try:
                        div_df = _extract_div_based_table(html, debug=debug)
                        if div_df is not None and not div_df.empty:
                            tables = [div_df]
                            _debug_print(
                                debug, f"Selenium: div-based extraction produced shape={div_df.shape} with columns={div_df.columns.tolist()}")
                        else:
                            _debug_print(
                                debug, "Selenium: div-based extraction found no rows")
                    except Exception as e:
                        last_err = e
                        _debug_print(
                            debug, f"Selenium: div-based extraction error: {e}")
                if tables:
                    break
            except Exception as e:
                last_err = e
                _debug_print(debug, f"Selenium fetch failed for {url}: {e}")

    if not tables:
        # Surface the most informative error we saw
        raise RuntimeError(
            f"No HTML tables found on the source page(s). Last error: {last_err or 'unknown'}")

    # Identify the relevant table by presence of required columns (case-insensitive)
    wanted = {"rank", "company", "ticker"}
    for df in tables:
        cols_norm = {str(c).strip().lower() for c in df.columns}
        if wanted.issubset(cols_norm):
            _debug_print(
                debug, f"Selected table with columns: {list(df.columns)} and shape={df.shape}")
            return df

    # If not found directly, try looser matching by picking the widest table
    # as a heuristic best-effort.
    best = max(tables, key=lambda d: d.shape[1])
    _debug_print(
        debug, f"Falling back to widest table with columns: {list(best.columns)} and shape={best.shape}")
    return best


def extract_fortune100_with_tickers(table: pd.DataFrame, top_n: int = DEFAULT_TOP_N, *, debug: bool = False) -> List[FortuneEntry]:
    """
    From the parsed table, extract (rank, company, ticker) for the first top_n rows.

    The function is resilient to different column label variants by normalizing names.
    Rows with missing or blank tickers are dropped (private / non-traded).
    """
    # Normalize column names
    df = table.copy()
    df.columns = [str(c).strip() for c in df.columns]
    _debug_print(debug, f"Normalized source columns: {df.columns.tolist()}")

    # Try to find likely columns
    # We support variants such as 'Rank', 'Company', 'Ticker', case-insensitive and with spaces.
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if "rank" == cl:
            colmap["rank"] = c
        elif cl in ("company", "company name", "name"):
            colmap["company"] = c
        elif cl in ("ticker", "symbol", "ticker symbol", "stock ticker"):
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
        candidates = [c for c in df.columns if any(
            x in c.lower() for x in ("ticker", "symbol"))]
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
    _debug_print(debug, f"After sorting and head({top_n}): {len(slim)} rows")

    # Drop entries without a usable ticker and filter to plausible symbols
    slim["Ticker"] = slim["Ticker"].astype(str).str.strip()
    before_filter = slim.shape[0]
    slim = slim[slim["Ticker"].str.len() > 0]
    slim = slim[~slim["Ticker"].str.contains(
        "^-$|^N/A$", case=False, regex=True)]
    # Keep only plausible ticker patterns (letters/numbers, dot or dash for class shares)
    plausible = slim["Ticker"].str.match(r"^[A-Za-z0-9][A-Za-z0-9\.-]{0,9}$")
    removed = before_filter - int(plausible.sum())
    if debug and removed > 0:
        _debug_print(
            debug, f"Filtered out {removed} rows with implausible tickers")
    slim = slim[plausible]

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
    _debug_print(debug, f"Built {len(entries)} entries with non-empty tickers")

    # Deduplicate by cleaned ticker, keeping the best (lowest) rank if duplicate appears
    seen = {}
    for e in sorted(entries, key=lambda x: x.rank):
        if e.cleaned_ticker not in seen:
            seen[e.cleaned_ticker] = e
    deduped = list(seen.values())
    _debug_print(debug, f"Deduped to {len(deduped)} unique tickers")
    return deduped


def _fetch_html_with_selenium(url: str, *, debug: bool = False, timeout: int = 20, headless: bool = True) -> str:
    """
    Use Selenium (headless Chrome) to render a JS-driven page and return page_source.
    Applies selenium-stealth if available to reduce bot detection. Waits up to `timeout`
    seconds for at least one <table> element to appear; if not, returns whatever was rendered.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.service import Service as ChromeService
    except Exception as e:
        raise RuntimeError(
            "Selenium is not installed. Install with: pip install selenium webdriver-manager selenium-stealth") from e

    # Try to use webdriver-manager to auto-install chromedriver
    driver = None
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36")

        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Apply selenium-stealth if available
        try:
            from selenium_stealth import stealth
            stealth(
                driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
            )
            _debug_print(debug, "Applied selenium-stealth to Chrome driver")
        except Exception as e:
            _debug_print(debug, f"selenium-stealth not applied: {e}")

        driver.set_page_load_timeout(timeout)
        driver.get(url)
        # Wait for a table element to appear
        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            _debug_print(debug, "Detected <table> element on page")
        except Exception as e:
            _debug_print(debug, f"Timed out waiting for <table>: {e}")
        html = driver.page_source
        return html
    finally:
        try:
            if driver is not None:
                driver.quit()
        except Exception:
            pass


def _scrape_us500_with_selenium_pages(base_url: str, *, debug: bool = False, timeout: int = 20, max_pages: int = 4, target_count: int = 100, headless: bool = True) -> Optional[pd.DataFrame]:
    """
    Navigate to ?page=1..N and extract visible grid rows (virtualized) by scrolling
    the grid container to collect rows on each page. Stitch pages until target_count.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.service import Service as ChromeService
        from webdriver_manager.chrome import ChromeDriverManager
    except Exception as e:
        raise RuntimeError(
            "Selenium not available. Install selenium webdriver-manager.") from e

    results: List[Dict[str, str]] = []
    seen_ranks = set()

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36")
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Apply stealth if available
        try:
            from selenium_stealth import stealth
            stealth(
                driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
            )
            _debug_print(
                debug, "Applied selenium-stealth to Chrome driver (DOM scrape)")
        except Exception:
            pass

        for page in range(1, max_pages + 1):
            url = base_url
            sep = '&' if ('?' in url) else '?'
            url = f"{url}{sep}page={page}"
            _debug_print(debug, f"DOM scrape: navigating to {url}")
            driver.set_page_load_timeout(timeout)
            driver.get(url)
            try:
                WebDriverWait(driver, timeout).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div.overflow-auto"))
                )
            except Exception as e:
                _debug_print(
                    debug, f"DOM scrape: grid container not found on page={page}: {e}")
                continue

            container = driver.find_element(
                By.CSS_SELECTOR, "div.overflow-auto")
            # Attempt to locate the relative-positioned items wrapper
            try:
                wrapper = container.find_element(
                    By.CSS_SELECTOR, "div[style*='position: relative']")
            except Exception:
                wrapper = container

            prev_count = -1
            stagnation = 0
            # Scroll and harvest until stable or target reached
            for iter_idx in range(200):
                rows = wrapper.find_elements(
                    By.CSS_SELECTOR, "div[style*='translateY']")
                for row in rows:
                    cells = row.find_elements(By.XPATH, "./div")
                    if len(cells) >= 3:
                        rank_txt = cells[0].text.strip()
                        company_txt = cells[1].text.strip()
                        ticker_txt = cells[2].text.strip()
                        if rank_txt.isdigit() and ticker_txt and company_txt:
                            rank_int = int(rank_txt)
                            if rank_int not in seen_ranks:
                                results.append(
                                    {"Rank": rank_int, "Company": company_txt, "Ticker": ticker_txt})
                                seen_ranks.add(rank_int)
                # Scroll further
                driver.execute_script(
                    "arguments[0].scrollTop = arguments[0].scrollTop + 1000;", container)
                time.sleep(0.15)
                if len(seen_ranks) == prev_count:
                    stagnation += 1
                else:
                    stagnation = 0
                prev_count = len(seen_ranks)
                if stagnation >= 15:
                    break
                if len(seen_ranks) >= target_count:
                    break

            _debug_print(
                debug, f"DOM scrape: after page={page}, collected {len(seen_ranks)} unique ranks")
            if len(seen_ranks) >= target_count:
                break

        if not results:
            return None
        # Build DataFrame
        df = pd.DataFrame(results)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
        df = df.dropna(subset=["Company", "Ticker"], how="any")
        df = df.drop_duplicates(
            subset=["Rank"], keep="first").sort_values("Rank")
        # Keep top target_count
        df = df.head(target_count)
        return df
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def _extract_div_based_table(html: str, *, debug: bool = False) -> Optional[pd.DataFrame]:
    """
    Heuristic parser for "table-like" structures built with DIVs and data-label attributes.
    Looks for elements with data-label in {Rank, Company, Ticker} (case-insensitive),
    groups by a common ancestor element, and builds a DataFrame.
    Returns None if it cannot find enough structure.
    """
    soup = BeautifulSoup(html, "lxml")
    labels_wanted = {"rank", "company", "ticker", "symbol"}
    candidates = soup.find_all(lambda tag: tag.has_attr(
        "data-label") and str(tag.get("data-label")).strip().lower() in labels_wanted)
    _debug_print(
        debug, f"Div-based: found {len(candidates)} elements with matching data-label")
    if not candidates:
        return None

    # Build records by grouping fields under a shared ancestor container per row
    records: List[Dict[str, str]] = []
    seen_containers = set()

    for el in candidates:
        # Walk up a few levels to find a likely row container
        container = el
        for _ in range(5):
            if container is None:
                break
            # Heuristics: a container with multiple data-label children is likely a row
            data_label_children = container.find_all(
                lambda t: t is not container and t.has_attr("data-label"), recursive=True)
            if len(data_label_children) >= 2:
                break
            container = container.parent
        if not container:
            continue
        key = id(container)
        if key in seen_containers:
            continue
        seen_containers.add(key)

        # Extract fields from this container
        fields = {}
        for cell in container.find_all(lambda t: t.has_attr("data-label")):
            label = str(cell.get("data-label")).strip()
            text = cell.get_text(" ", strip=True)
            fields[label] = text

        # Normalize usable subset
        def pick(keys: Tuple[str, ...]) -> Optional[str]:
            for k in keys:
                v = fields.get(k)
                if v:
                    return v
            return None

        rank_val = pick(("Rank", "rank"))
        company_val = pick(("Company", "company", "Name", "name"))
        ticker_val = pick(("Ticker", "ticker", "Symbol", "symbol"))

        if company_val and ticker_val:
            rec = {
                "Rank": rank_val if rank_val is not None else "",
                "Company": company_val,
                "Ticker": ticker_val,
            }
            records.append(rec)

    _debug_print(
        debug, f"Div-based: built {len(records)} potential records from containers")
    if not records:
        return None

    df = pd.DataFrame(records)
    # Coerce rank numeric when possible
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "Rank" in df.columns:
            df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    # Drop duplicates and empties
    df = df.dropna(subset=["Company", "Ticker"], how="any")
    df = df.drop_duplicates(subset=["Company", "Ticker"], keep="first")
    _debug_print(debug, f"Div-based: final DataFrame shape={df.shape}")
    return df


def _try_nextjs_data_api(page_url: str, html_text: str, *, headers: Dict[str, str], debug: bool = False, desired_count: int = DEFAULT_TOP_N) -> Optional[pd.DataFrame]:
    """
    If the page is a Next.js app, infer the buildId from the HTML and query the
    /_next/data/{buildId}/{route}.json endpoint to retrieve structured data.
    Then, search the JSON for a list of objects that looks like the target table
    (fields containing rank/company/ticker or symbol). Returns a DataFrame or None.
    """
    # Look for either _buildManifest.js or _ssgManifest.js to extract buildId
    m = re.search(r"/_next/static/([^/]+)/_buildManifest\.js", html_text)
    if not m:
        m = re.search(r"/_next/static/([^/]+)/_ssgManifest\.js", html_text)
    if not m:
        _debug_print(debug, "Next.js data API: buildId not found in HTML")
        return None
    build_id = m.group(1)
    _debug_print(debug, f"Next.js data API: found buildId={build_id}")

    parsed = urlparse(page_url)
    route_path = parsed.path.strip("/")
    if not route_path:
        route_path = "index"
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    def _sanitize_filename(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", s)[:160]

    def fetch_json(u: str) -> Optional[dict]:
        _debug_print(debug, f"Next.js data API: fetching {u}")
        r = requests.get(u, headers=headers, timeout=REQUESTS_TIMEOUT)
        if not r.ok:
            _debug_print(
                debug, f"Next.js data API: HTTP {r.status_code} for {u}")
            return None
        try:
            payload = r.json()
            if debug:
                try:
                    key = u.split('/_next/data/')[-1]
                    fname = f"debug_next_data_{_sanitize_filename(key)}.json"
                    with open(fname, 'w', encoding='utf-8') as fh:
                        json.dump(payload, fh, indent=2)
                    _debug_print(debug, f"Saved Next.js JSON to {fname}")
                except Exception as save_err:
                    _debug_print(
                        debug, f"Failed to save Next.js JSON: {save_err}")
            return payload
        except Exception as e:
            _debug_print(
                debug, f"Next.js data API: JSON parse error for {u}: {e}")
            return None

    # Fetch base page JSON
    data_url = f"{base_url}/_next/data/{build_id}/{route_path}.json"
    payload = fetch_json(data_url)
    if payload is None:
        return None

    def normalize_records(records: List[Dict]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for rec in records:
            rank = rec.get("rank") or rec.get("Rank")
            company = rec.get("company") or rec.get(
                "Company") or rec.get("name") or rec.get("Name")
            ticker = rec.get("ticker") or rec.get(
                "Ticker") or rec.get("symbol") or rec.get("Symbol")
            if company and ticker:
                out.append({
                    "Rank": rank if rank is not None else "",
                    "Company": company,
                    "Ticker": ticker,
                })
        return out

    all_rows: List[Dict[str, str]] = []
    first_records = _find_records_in_json(payload, debug=debug)
    _debug_print(
        debug, f"Next.js data API: found {len(first_records)} candidate records in JSON")
    all_rows.extend(normalize_records(first_records))

    # If fewer than desired_count, try multiple pagination URL patterns
    page = 2
    max_pages = 20
    page_size_guess = len(all_rows) if all_rows else 50
    while len(all_rows) < desired_count and page <= max_pages:
        added_any = False
        offset = (page - 1) * max(page_size_guess, 1)
        candidates = [
            f"{base_url}/_next/data/{build_id}/{route_path}.json?page={page}",
            f"{base_url}/_next/data/{build_id}/{route_path}/page/{page}.json",
            f"{base_url}/_next/data/{build_id}/{route_path}/page-{page}.json",
            f"{base_url}/_next/data/{build_id}/{route_path}-{page}.json",
            f"{base_url}/_next/data/{build_id}/{route_path}/{page}.json",
            f"{base_url}/_next/data/{build_id}/{route_path}.json?p={page}",
            f"{base_url}/_next/data/{build_id}/{route_path}.json?offset={offset}",
            f"{base_url}/_next/data/{build_id}/{route_path}.json?from={offset}",
            f"{base_url}/_next/data/{build_id}/{route_path}.json?start={offset}",
            f"{base_url}/_next/data/{build_id}/{route_path}.json?skip={offset}",
            f"{base_url}/_next/data/{build_id}/{route_path}.json?offset={offset}&limit={page_size_guess}",
            f"{base_url}/_next/data/{build_id}/{route_path}.json?page={page}&pageSize={page_size_guess}",
            f"{base_url}/_next/data/{build_id}/{route_path}.json?page={page}&limit={page_size_guess}",
            f"{base_url}/_next/data/{build_id}/{route_path}.json?page={page}&perPage={page_size_guess}",
        ]
        for u in candidates:
            payload_var = fetch_json(u)
            if payload_var is None:
                continue
            recs_v = _find_records_in_json(payload_var, debug=debug)
            rows_v = normalize_records(recs_v)
            before = len(all_rows)
            all_rows.extend(rows_v)
            after = len(all_rows)
            _debug_print(
                debug, f"Next.js data API: tried {u} -> added {after - before} rows (total={after})")
            if after > before:
                added_any = True
        if not added_any:
            _debug_print(
                debug, f"Next.js data API: no rows added for page={page}; stopping pagination")
            break
        page += 1

    # Deduplicate by (Company, Ticker)
    seen_keys = set()
    deduped: List[Dict[str, str]] = []
    for r in all_rows:
        key = (r.get("Company"), r.get("Ticker"))
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(r)

    df = pd.DataFrame(deduped)
    _debug_print(
        debug, f"Next.js data API: aggregated {len(all_rows)} rows before dedupe; {len(deduped)} after dedupe")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "Rank" in df.columns:
            df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df = df.dropna(subset=["Company", "Ticker"], how="any")
    df = df.drop_duplicates(subset=["Company", "Ticker"], keep="first")
    # Sort by Rank if available, then Company
    if "Rank" in df.columns:
        df = df.sort_values(["Rank", "Company"], ascending=[True, True])
    return df


def _find_records_in_json(obj, *, debug: bool = False) -> List[Dict]:
    """
    Recursively search the JSON object for arrays of dicts that contain the
    expected fields (rank/company/ticker or symbol). Returns the largest candidate.
    """
    matches: List[List[Dict]] = []

    def looks_like_record(d: Dict) -> bool:
        keys = {str(k).lower() for k in d.keys()}
        has_company = ("company" in keys) or ("name" in keys)
        has_ticker = ("ticker" in keys) or ("symbol" in keys)
        has_rank = ("rank" in keys)
        return has_company and has_ticker or (has_company and has_rank)

    def walk(x):
        if isinstance(x, list):
            if x and isinstance(x[0], dict):
                good = [d for d in x if isinstance(
                    d, dict) and looks_like_record(d)]
                if good:
                    matches.append(good)
            for item in x:
                walk(item)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    walk(obj)
    matches.sort(key=len, reverse=True)
    if matches:
        best = matches[0]
        _debug_print(
            debug, f"Next.js data API: candidate arrays found sizes={list(map(len, matches[:5]))}")
        return best
    return []


def _try_us500_chunk_api(page_url: str, html_text: str, *, headers: Dict[str, str], debug: bool = False, desired_count: int = DEFAULT_TOP_N) -> Optional[pd.DataFrame]:
    """
    Heuristic: fetch the Next.js page chunk for this route and scan for API endpoints
    (e.g., /api/...). Try those endpoints with common pagination params to accumulate
    >= desired_count rows. Returns DataFrame or None.
    """
    parsed = urlparse(page_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    m = re.search(
        r"src=\"(/_next/static/chunks/pages/fortune-500-companies-[^\"]+\.js)\"", html_text)
    if not m:
        _debug_print(debug, "Chunk API: page chunk script not found in HTML")
        return None
    chunk_path = m.group(1)
    chunk_url = base_url + chunk_path
    _debug_print(debug, f"Chunk API: fetching page chunk {chunk_url}")
    r = requests.get(chunk_url, headers=headers, timeout=REQUESTS_TIMEOUT)
    if not r.ok:
        _debug_print(debug, f"Chunk API: HTTP {r.status_code} for {chunk_url}")
        return None
    js = r.text
    if debug:
        try:
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", chunk_path.strip("/"))
            fname = f"debug_chunk_{safe}"
            with open(fname, 'w', encoding='utf-8') as fh:
                fh.write(js)
            _debug_print(debug, f"Chunk API: saved chunk JS to {fname}")
        except Exception as save_err:
            _debug_print(
                debug, f"Chunk API: failed to save chunk JS: {save_err}")

    # Extract candidate API endpoints from JS content
    endpoints: List[str] = []
    for m2 in re.finditer(r"https?://[a-zA-Z0-9\.-]+/api/[a-zA-Z0-9_\-/?=&]+", js):
        endpoints.append(m2.group(0))
    for m2 in re.finditer(r"(?<![a-zA-Z0-9_\-])(/api/[a-zA-Z0-9_\-/]+)", js):
        endpoints.append(base_url + m2.group(1))
    endpoints = sorted(set(endpoints))
    _debug_print(
        debug, f"Chunk API: discovered {len(endpoints)} candidate endpoints")
    if not endpoints:
        return None
    if debug:
        _debug_print(
            debug, f"Chunk API: endpoints sample: {', '.join(endpoints[:3])}")

    def try_fetch_records(u: str) -> List[Dict[str, str]]:
        try:
            rr = requests.get(u, headers=headers, timeout=REQUESTS_TIMEOUT)
            if not rr.ok:
                return []
            payload = rr.json()
        except Exception:
            return []
        recs = _find_records_in_json(payload, debug=debug)
        out: List[Dict[str, str]] = []
        for rec in recs:
            rank = rec.get("rank") or rec.get("Rank")
            company = rec.get("company") or rec.get(
                "Company") or rec.get("name") or rec.get("Name")
            ticker = rec.get("ticker") or rec.get(
                "Ticker") or rec.get("symbol") or rec.get("Symbol")
            if company and ticker:
                out.append({"Rank": rank if rank is not None else "",
                           "Company": company, "Ticker": ticker})
        return out

    all_rows: List[Dict[str, str]] = []
    for ep in endpoints:
        rows0 = try_fetch_records(ep)
        if rows0:
            all_rows.extend(rows0)
            _debug_print(
                debug, f"Chunk API: {ep} yielded {len(rows0)} rows (total={len(all_rows)})")
        if len(all_rows) >= desired_count:
            break
        page = 2
        offset = len(rows0) if rows0 else 50
        params = [
            f"?page={{n}}", f"?p={{n}}", f"?offset={{off}}", f"?start={{off}}", f"?from={{off}}", f"?skip={{off}}",
            f"?page={{n}}&limit={offset}", f"?page={{n}}&pageSize={offset}", f"?page={{n}}&perPage={offset}",
        ]
        while len(all_rows) < desired_count and page <= 10:
            added_any = False
            for p in params:
                url_try = ep + p.format(n=page, off=offset*(page-1))
                rows = try_fetch_records(url_try)
                before = len(all_rows)
                all_rows.extend(rows)
                after = len(all_rows)
                if debug:
                    _debug_print(
                        debug, f"Chunk API: tried {url_try} -> +{after-before} rows (total={after})")
                if after > before:
                    added_any = True
            if not added_any:
                break
            page += 1
        if len(all_rows) >= desired_count:
            break

    if not all_rows:
        return None
    seen = set()
    deduped: List[Dict[str, str]] = []
    for r in all_rows:
        key = (r.get("Company"), r.get("Ticker"))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    df = pd.DataFrame(deduped)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "Rank" in df.columns:
            df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    _debug_print(
        debug, f"Chunk API: aggregated {len(all_rows)} rows; {len(deduped)} unique after dedupe")
    return df


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


def _get_series(raw: pd.DataFrame, field: str, ticker: str) -> Optional[pd.Series]:
    if isinstance(raw.columns, pd.MultiIndex):
        key = (field, ticker)
        if key in raw.columns:
            return raw[key].dropna()
        # Some yfinance returns may invert levels; try alternative
        try:
            return raw[field][ticker].dropna()
        except Exception:
            return None
    else:
        # Single ticker case
        if field in raw.columns:
            return raw[field].dropna()
    return None


def _minmax_normalize(col: pd.Series) -> pd.Series:
    s = col.astype(float).copy()
    if s.dropna().empty:
        return s
    cmin, cmax = s.min(), s.max()
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
        return s * 0.0
    return (s - cmin) / (cmax - cmin)


def compute_additional_vol_measures(
    raw: pd.DataFrame,
    tickers: List[str],
    min_days: int = DEFAULT_MIN_DAYS,
    ewma_lambda: float = 0.94,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Compute additional, widely used daily volatility estimators per ticker over the lookback window:
      - parkinson_vol: range-based using High/Low
      - gk_vol: Garman-Klass using OHLC
      - rs_vol: Rogers-Satchell using OHLC
      - yz_vol: Yang-Zhang using overnight + open-close + RS
      - ewma_vol: RiskMetrics EWMA on log returns (Adj Close if available else Close)
      - mad_vol: Robust MAD-based estimator on log returns
    All measures annualized by sqrt(252). Returns DataFrame with one row per ticker.
    """
    results: List[Dict[str, float]] = []
    ln2 = np.log(2.0)

    for t in tickers:
        rec: Dict[str, float] = {"ticker": t}

        # Gather required series
        s_close = _get_series(raw, "Adj Close", t)
        if s_close is None:
            s_close = _get_series(raw, "Close", t)
        s_open = _get_series(raw, "Open", t)
        s_high = _get_series(raw, "High", t)
        s_low = _get_series(raw, "Low", t)

        # Align to common dates if needed
        frames = {k: v for k, v in {
            "close": s_close, "open": s_open, "high": s_high, "low": s_low}.items() if v is not None}
        if not frames or "close" not in frames:
            continue
        df = pd.DataFrame(frames).dropna(how="any")
        if df.shape[0] < max(5, min_days):
            continue

        # Close-to-close log returns (for EWMA and MAD)
        r_cc = np.log(df["close"] / df["close"].shift(1)).dropna()

        # Parkinson (requires high/low)
        if {"high", "low"}.issubset(df.columns):
            hl = np.log(df["high"] / df["low"]) ** 2
            parkinson_var = (hl.sum()) / (4.0 * ln2 * df.shape[0])
            if np.isfinite(parkinson_var) and parkinson_var >= 0:
                rec["parkinson_vol"] = float(np.sqrt(parkinson_var) * np.sqrt(252.0))

        # Garman-Klass and Rogers-Satchell and Yang-Zhang (require OHLC)
        if {"open", "high", "low", "close"}.issubset(df.columns):
            log_hl = np.log(df["high"] / df["low"]) ** 2
            log_co = np.log(df["close"] / df["open"]) ** 2
            gk_var = 0.5 * log_hl.mean() - (2.0 * ln2 - 1.0) * log_co.mean()
            if np.isfinite(gk_var) and gk_var >= 0:
                rec["gk_vol"] = float(np.sqrt(gk_var) * np.sqrt(252.0))

            # Rogers-Satchell per-day term
            term_rs = np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) + \
                np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"])
            rs_var = term_rs.mean()
            if np.isfinite(rs_var) and rs_var >= 0:
                rec["rs_vol"] = float(np.sqrt(rs_var) * np.sqrt(252.0))

            # Yang-Zhang components
            prev_close = df["close"].shift(1)
            r_o = np.log(df["open"] / prev_close).dropna()  # overnight
            r_c = np.log(df["close"] / df["open"]).dropna()  # open-to-close
            # Recompute RS on aligned subset
            df_rs = df.loc[r_c.index]
            term_rs_d = np.log(df_rs["high"] / df_rs["close"]) * np.log(df_rs["high"] / df_rs["open"]) + \
                np.log(df_rs["low"] / df_rs["close"]) * np.log(df_rs["low"] / df_rs["open"])
            sigma_o2 = float(r_o.var(ddof=1)) if r_o.shape[0] >= 2 else np.nan
            sigma_c2 = float(r_c.var(ddof=1)) if r_c.shape[0] >= 2 else np.nan
            sigma_rs = float(term_rs_d.mean()) if term_rs_d.shape[0] >= 1 else np.nan
            if np.isfinite(sigma_o2) and np.isfinite(sigma_c2) and np.isfinite(sigma_rs):
                k = 0.34
                yz_var = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs
                if yz_var >= 0:
                    rec["yz_vol"] = float(np.sqrt(yz_var) * np.sqrt(252.0))

        # EWMA on close-to-close returns
        if r_cc.shape[0] >= min_days:
            lam = ewma_lambda
            var = float(r_cc.var(ddof=1)) if r_cc.shape[0] >= 2 else float((r_cc.iloc[-1] ** 2))
            for x in r_cc.iloc[-min_days:]:
                var = lam * var + (1.0 - lam) * float(x * x)
            if var >= 0:
                rec["ewma_vol"] = float(np.sqrt(var) * np.sqrt(252.0))

            # Robust MAD-based volatility
            mad = float(np.median(np.abs(r_cc - np.median(r_cc))))
            rec["mad_vol"] = float(1.4826 * mad * np.sqrt(252.0))

        results.append(rec)

    out = pd.DataFrame(results)
    if out.empty:
        return out
    # Ensure consistent column order
    cols = [c for c in [
        "ticker", "parkinson_vol", "gk_vol", "rs_vol", "yz_vol", "ewma_vol", "mad_vol"
    ] if c in out.columns]
    return out[cols]

def attach_company_names(vol_df: pd.DataFrame, entries: List[FortuneEntry]) -> pd.DataFrame:
    """
    Add 'company' and 'rank' columns to the volatility table using the FortuneEntry mapping.
    """
    mapping_name: Dict[str, str] = {
        e.cleaned_ticker: e.company for e in entries}
    mapping_rank: Dict[str, int] = {e.cleaned_ticker: e.rank for e in entries}
    vol_df["company"] = vol_df["ticker"].map(mapping_name)
    vol_df["fortune_rank"] = vol_df["ticker"].map(mapping_rank)
    # Preferred ordering: identifiers first, then metrics if present
    base = ["fortune_rank", "company", "ticker", "n_obs", "daily_std"]
    metrics_order = [
        "cc_vol", "parkinson_vol", "gk_vol", "rs_vol", "yz_vol", "ewma_vol", "mad_vol",
        "norm_avg_vol", "n_measures", "ann_vol"
    ]
    cols = base + [c for c in metrics_order if c in vol_df.columns]
    # Include any other remaining columns
    cols += [c for c in vol_df.columns if c not in cols]

    # Choose ranking metric: prefer composite, then cc_vol, else ann_vol
    rank_metric = None
    for m in ("norm_avg_vol", "cc_vol", "ann_vol"):
        if m in vol_df.columns:
            rank_metric = m
            break
    if rank_metric is None:
        rank_metric = "daily_std"

    return vol_df[cols].sort_values([rank_metric, "fortune_rank"], ascending=[False, True]).reset_index(drop=True)


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
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging for fetching/parsing stage")
    parser.add_argument("--use-selenium", action="store_true",
                        help="Enable Selenium fallback to render JS-driven pages")
    parser.add_argument("--selenium-timeout", type=int, default=20,
                        help="Max seconds to wait for JS-rendered tables")
    parser.add_argument("--non-headless", action="store_true",
                        help="Run Selenium in a visible browser (not headless)")
    parser.add_argument("--source-file", type=str, default=None,
                        help="Local CSV/HTML file containing Rank, Company, Ticker columns to bypass web fetch")
    args = parser.parse_args()

    # --- Step 1: Scrape Fortune 500 page and extract Fortune 100 with tickers
    print(f"[1/5] Fetching Fortune 500 table from: {args.source_url}")
    if args.debug:
        print("[DEBUG] Debug logging enabled")
    try:
        table = fetch_fortune_table(
            args.source_url,
            debug=args.debug,
            use_selenium=args.use_selenium,
            selenium_timeout=args.selenium_timeout,
            source_file=args.source_file,
            desired_count=args.top_n,
            selenium_headless=(not args.non_headless),
        )
    except Exception as e:
        print(
            f"ERROR: Failed to fetch/parse source table: {e}", file=sys.stderr)
        return 2

    print(f"[2/5] Extracting top {args.top_n} entries and cleaning tickers...")
    try:
        entries = extract_fortune100_with_tickers(
            table, top_n=args.top_n, debug=args.debug)
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

    # --- Step 4: Compute volatility measures
    print(
        f"[5/5] Computing volatility measures (close-close, Parkinson, GK, RS, YZ, EWMA, MAD); requiring at least {args.min_days} observations...")
    vol_base = compute_annualized_volatility(price_wide, min_days=args.min_days)
    if vol_base.empty:
        print("ERROR: No tickers had sufficient data to compute volatility.", file=sys.stderr)
        return 2
    # Rename to make explicit (close-close)
    vol_base = vol_base.rename(columns={"ann_vol": "cc_vol"})
    # Additional measures from raw OHLC data
    extra = compute_additional_vol_measures(raw, [e.cleaned_ticker for e in entries], min_days=args.min_days, debug=args.debug)
    # Merge
    vol = vol_base.merge(extra, on="ticker", how="left")

    # Build normalized-averaged composite across available measures
    measure_cols = [c for c in ["cc_vol", "parkinson_vol", "gk_vol", "rs_vol", "yz_vol", "ewma_vol", "mad_vol"] if c in vol.columns]
    if measure_cols:
        normed = vol[measure_cols].apply(_minmax_normalize, axis=0)
        vol["norm_avg_vol"] = normed.mean(axis=1, skipna=True)
        vol["n_measures"] = normed.notna().sum(axis=1)

    vol = attach_company_names(vol, entries)

    # Save results
    out_csv = args.output_csv
    vol.to_csv(out_csv, index=False)
    print(f"\nSaved full results to: {out_csv}")

    # Display top N by volatility (prefer composite if available)
    top_k = min(args.print_top, vol.shape[0])
    rank_metric = next((m for m in ("norm_avg_vol", "cc_vol", "ann_vol") if m in vol.columns), None)
    metric_label = rank_metric if rank_metric else "daily_std"
    print(f"\nTop {top_k} highest-volatility Fortune {args.top_n} *stocks* by {metric_label} (lookback={args.lookback_days} trading days):\n")
    to_show = vol.head(top_k).copy()
    # Build dynamic formatters
    fmts = {
        "fortune_rank": "{:>3d}".format,
        "n_obs": "{:>4d}".format,
        "daily_std": "{:.4f}".format,
    }
    for m in ("cc_vol", "parkinson_vol", "gk_vol", "rs_vol", "yz_vol", "ewma_vol", "mad_vol", "norm_avg_vol"):
        if m in to_show.columns:
            fmts[m] = "{:.4f}".format
    with pd.option_context("display.max_colwidth", 80, "display.width", 160):
        print(to_show.to_string(index=False, formatters=fmts))

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
            print("Skipped due to missing data:", ", ".join(sorted(skipped)))

    missing_tickers = total_requested - len(entries)
    if missing_tickers > 0:
        print(f"Note: {missing_tickers} of the top {total_requested} Fortune entries appear to be private / have no ticker and were excluded.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
