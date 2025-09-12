"""Ticker retrieval utilities for the Highest Volatility package."""

from __future__ import annotations

from dataclasses import dataclass

import json
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


@dataclass
class FortuneTicker:
    """Representation of a single company in the Fortune list."""

    rank: int
    company: str
    ticker: str


# Default source and cache configuration
DEFAULT_SOURCE_URL = "https://us500.com/fortune-500-companies"
CACHE_DIR = Path(".cache") / "tickers"
CACHE_FILE = "fortune.json"
CACHE_EXPIRY = 24 * 60 * 60  # one day
FALLBACK_TICKERS = [
    FortuneTicker(1, "Apple", "AAPL"),
    FortuneTicker(2, "Microsoft", "MSFT"),
    FortuneTicker(3, "Amazon.com", "AMZN"),
    FortuneTicker(4, "Alphabet", "GOOGL"),
    FortuneTicker(5, "Meta Platforms", "META"),
]


def normalize_ticker(ticker: str) -> str:
    """Return a Yahoo Finance compatible ticker symbol."""

    t = ticker.strip().upper()
    return t.replace(".", "-")


def _fetch_with_selenium(source_url: str, top_n: int) -> pd.DataFrame:
    """Fetch the Fortune list via Selenium Stealth.

    The web site backing the list renders its table client side and limits the
    JSON endpoint to the first 50 rows.  When that happens we resort to a
    headless browser that mimics a real user.  ``selenium-stealth`` is used to
    reduce the chance of being blocked.
    """

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium_stealth import stealth

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    try:
        stealth(
            driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )
        driver.get(source_url)
        WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table tbody tr"))
        )
        html = driver.page_source
    finally:
        driver.quit()

    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("table tbody tr")
    records: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        cols = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cols) < 3:
            continue
        try:
            rank = int(cols[0])
        except ValueError:
            continue
        company = cols[1]
        ticker = normalize_ticker(cols[2])
        if ticker in seen:
            continue
        records.append({"rank": rank, "company": company, "ticker": ticker})
        seen.add(ticker)
        if len(records) >= top_n:
            break

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("rank").reset_index(drop=True)
    return df


def fetch_fortune_tickers(
    source_url: str = DEFAULT_SOURCE_URL,
    *,
    top_n: int = 100,
    cache_dir: Path | str = CACHE_DIR,
    cache_expiry: int = CACHE_EXPIRY,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch the Fortune company list and return the first *top_n* rows.

    Results are cached on disk in ``.cache/tickers`` to avoid repeated
    network requests.  The primary data source is ``us500.com`` which renders
    its table via JavaScript.  We parse the ``__NEXT_DATA__`` JSON blob and
    iterate over the paginated JSON API.  Some deployments of the site limit
    that endpoint to the first 50 rows; when fewer than ``top_n`` rows are
    returned we automatically fall back to a headless Selenium session using
    ``selenium-stealth`` to fetch the full table.  If everything fails a small
    built-in fallback list is returned instead.
    """

    cache_path = Path(cache_dir) / CACHE_FILE

    # Attempt to use cached data if still valid
    try:
        if cache_path.exists():
            with cache_path.open() as fh:
                payload = json.load(fh)
            if time.time() - payload.get("timestamp", 0) < cache_expiry:
                table = pd.DataFrame(payload.get("data", []))
                if not table.empty:
                    table = table.rename(columns=str.lower)
                    table["rank"] = table["rank"].astype(int)
                    table["ticker"] = table["ticker"].astype(str).map(normalize_ticker)
                    table = table[table["ticker"].str.fullmatch(r"[A-Z]+[A-Z0-9.-]*")]
                    table = table.drop_duplicates(subset="ticker")
                    return table.sort_values("rank").head(top_n).reset_index(drop=True)
    except Exception:
        pass

    headers = {"User-Agent": "Mozilla/5.0"}
    session = session or requests
    results: list[dict] = []
    total: int | None = None
    build_id: str | None = None
    seen: set[str] = set()

    try:
        # Fetch the initial HTML page which contains the ``__NEXT_DATA__``
        # script.  This gives us the first batch of results as well as the
        # ``buildId`` which is required to access the JSON endpoint for
        # subsequent pages.
        response = session.get(source_url, timeout=30, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        script = soup.find("script", id="__NEXT_DATA__")
        if script and script.string:
            data = json.loads(script.string)
            build_id = data.get("buildId")
            page_data = data["props"]["pageProps"].get("initialResults", [])
            for rec in page_data:
                ticker = rec.get("ticker")
                if not ticker:
                    continue
                t = normalize_ticker(str(ticker))
                if t in seen:
                    continue
                rec["ticker"] = t
                results.append(rec)
                seen.add(t)
            stats = data["props"]["pageProps"].get("stats", {})
            total = stats.get("ffc")

        # Continue fetching more pages via the JSON endpoint.  Each page
        # contains 50 results; we stop once ``top_n`` or the reported total
        # number of companies has been reached.
        page = 2
        while build_id and (total is None or len(results) < total) and len(results) < top_n:
            api_url = (
                f"https://us500.com/_next/data/{build_id}/fortune-500-companies.json?page={page}"
            )
            resp = session.get(api_url, timeout=30, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            page_data = data.get("pageProps", {}).get("initialResults", [])
            new = 0
            for rec in page_data:
                ticker = rec.get("ticker")
                if not ticker:
                    continue
                t = normalize_ticker(str(ticker))
                if t in seen:
                    continue
                rec["ticker"] = t
                results.append(rec)
                seen.add(t)
                new += 1
            if new == 0:
                break
            page += 1
    except Exception:
        results = []

    if results:
        table = pd.DataFrame(results)
        table = table.rename(columns=str.lower)
        expected_cols = ["rank", "company", "ticker"]
        table = table[expected_cols]
        table["rank"] = table["rank"].astype(int)
        table["ticker"] = table["ticker"].astype(str)
        table = table[table["ticker"].str.fullmatch(r"[A-Z]+[A-Z0-9.-]*")]
        table = table.drop_duplicates(subset="ticker")
        table = table.sort_values("rank").reset_index(drop=True)

        if len(table) >= min(top_n, total or len(table)):
            table = table.head(top_n)

            # Store to cache
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("w") as fh:
                    json.dump(
                        {"timestamp": time.time(), "data": table.to_dict(orient="records")},
                        fh,
                    )
            except Exception:
                pass

            return table
        # If we did not get enough rows attempt Selenium fallback when no
        # custom session is provided (tests use a dummy session).
        if session is None:
            try:
                sel_table = _fetch_with_selenium(source_url, top_n)
                if not sel_table.empty:
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with cache_path.open("w") as fh:
                            json.dump(
                                {
                                    "timestamp": time.time(),
                                    "data": sel_table.to_dict(orient="records"),
                                },
                                fh,
                            )
                    except Exception:
                        pass
                    return sel_table.head(top_n)
            except Exception:
                pass

        return table.head(top_n)

    # No results from the JSON endpoint – try Selenium before falling back to
    # the built-in list.
    if session is None:
        try:
            sel_table = _fetch_with_selenium(source_url, top_n)
            if not sel_table.empty:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with cache_path.open("w") as fh:
                        json.dump(
                            {"timestamp": time.time(), "data": sel_table.to_dict(orient="records")},
                            fh,
                        )
                except Exception:
                    pass
                return sel_table.head(top_n)
        except Exception:
            pass

    fallback_df = pd.DataFrame([f.__dict__ for f in FALLBACK_TICKERS])
    return fallback_df.head(top_n)
