"""Ticker retrieval utilities for the Highest Volatility package."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from bs4 import BeautifulSoup


@dataclass
class FortuneTicker:
    """Representation of a single company in the Fortune list."""

    rank: int
    company: str
    ticker: str


DEFAULT_SOURCE_URL = "https://us500.com/fortune-500-companies"
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

    The web site backing the list renders its table client side.  A headless
    browser that mimics a real user is therefore required.  ``selenium-stealth``
    is used to reduce the chance of being blocked.
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
) -> pd.DataFrame:
    """Fetch the Fortune company list and return the first *top_n* rows.

    The list is retrieved exclusively via Selenium Stealth.  If scraping fails
    a small built-in fallback list is returned instead.
    """

    try:
        table = _fetch_with_selenium(source_url, top_n)
        if not table.empty:
            return table.head(top_n)
    except Exception:
        pass

    fallback_df = pd.DataFrame([f.__dict__ for f in FALLBACK_TICKERS])
    return fallback_df.head(top_n)

