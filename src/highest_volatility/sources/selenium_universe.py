"""Selenium Stealth helpers to extract Fortune companies + tickers.

This module centralizes browser setup and scraping utilities so that the
rest of the codebase can build a reliable ticker universe without relying
on JSON endpoints.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
import os
from typing import Iterable, List, Tuple, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium_stealth import stealth


@contextmanager
def chrome() -> Iterable[webdriver.Chrome]:
    """Provision a hardened, headless Chrome driver.

    Uses selenium-stealth to reduce automation fingerprints and
    webdriver-manager for Windows-friendly driver management.
    """

    opts = Options()
    # Headless Chrome (new) works across platforms including Windows 11
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    # Reduce noisy Chrome logs and automation fingerprints
    opts.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    opts.add_experimental_option("useAutomationExtension", False)
    # Optional: speed up by blocking images
    try:
        opts.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
    except Exception:
        pass
    # Silence GPU/WebGL related noise
    opts.add_argument("--disable-webgl")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-notifications")
    opts.add_argument("--log-level=3")

    # A realistic user-agent helps on some sites
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    # Suppress chromedriver logs
    try:
        service = Service(ChromeDriverManager().install(), log_path=os.devnull)
    except Exception:
        service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    stealth(
        driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL",
        fix_hairline=True,
    )

    try:
        yield driver
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def _collect_visible_pairs(
    driver: webdriver.Chrome,
    *,
    row_selector: str,
    name_selector: str,
    ticker_selector: str,
) -> List[Tuple[str, str]]:
    """Collect (company, ticker) from currently visible rows only."""

    pairs: List[Tuple[str, str]] = []
    rows: List = []
    selectors = [s.strip() for s in row_selector.split(",") if s.strip()]
    for sel in selectors:
        try:
            rows.extend(driver.find_elements(By.CSS_SELECTOR, sel))
        except Exception:
            continue
    for r in rows:
        try:
            name = r.find_element(By.CSS_SELECTOR, name_selector).text.strip()
            ticker = r.find_element(By.CSS_SELECTOR, ticker_selector).text.strip()
        except Exception:
            continue
        if not ticker:
            continue
        pairs.append((name, ticker))
    return pairs


def _infinite_scroll_collect(
    driver: webdriver.Chrome,
    *,
    row_selector: str,
    name_selector: str,
    ticker_selector: str,
    min_needed: int,
    max_passes: int = 2000,
    pause: float = 0.35,
    scroll_container_selector: str | None = None,
) -> List[Tuple[str, str]]:
    """Scrolls through a virtualized list/table and accumulates unique pairs.

    Continues until ``min_needed`` unique tickers are observed or ``max_passes``
    scroll steps have been executed without net new rows.
    """

    seen = set()
    out: List[Tuple[str, str]] = []
    last_count = 0
    stagnant_passes = 0

    container = None
    if scroll_container_selector:
        selectors = [s.strip() for s in scroll_container_selector.split(",") if s.strip()]
        for sel in selectors:
            try:
                container = driver.find_element(By.CSS_SELECTOR, sel)
                if container:
                    break
            except Exception:
                container = None

    # Scroll step-wise from top -> bottom to traverse virtualized rows
    # Reset to top first
    if container is not None:
        driver.execute_script("arguments[0].scrollTo(0, 0);", container)
    else:
        driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(pause)

    for _ in range(max_passes):
        # Gather pairs in current viewport
        for name, ticker in _collect_visible_pairs(
            driver,
            row_selector=row_selector,
            name_selector=name_selector,
            ticker_selector=ticker_selector,
        ):
            tkr = ticker.strip().upper()
            if not tkr or tkr in seen:
                continue
            seen.add(tkr)
            out.append((name, tkr))
            if len(out) >= min_needed:
                return out

        # Advance scroll by ~ one viewport height
        if container is not None:
            # Get current scrollTop, clientHeight, scrollHeight
            scroll_top, client_h, scroll_h = driver.execute_script(
                "return [arguments[0].scrollTop, arguments[0].clientHeight, arguments[0].scrollHeight];",
                container,
            )
            step = int(client_h * 0.9) or 300
            next_top = min(scroll_top + step, scroll_h)
            driver.execute_script("arguments[0].scrollTo(0, arguments[1]);", container, next_top)
            at_bottom = next_top >= scroll_h or (scroll_h - next_top) <= 2
        else:
            scroll_top, client_h, scroll_h = driver.execute_script(
                "return [document.documentElement.scrollTop || window.pageYOffset, window.innerHeight, document.body.scrollHeight];"
            )
            step = int(client_h * 0.9) or 300
            next_top = min(scroll_top + step, scroll_h)
            driver.execute_script("window.scrollTo(0, arguments[0]);", next_top)
            at_bottom = next_top >= scroll_h or (scroll_h - next_top) <= 2

        time.sleep(pause)

        if at_bottom:
            stagnant_passes += 1
        else:
            stagnant_passes = 0
        if stagnant_passes >= 3:
            break

        for name, ticker in _collect_visible_pairs(
            driver,
            row_selector=row_selector,
            name_selector=name_selector,
            ticker_selector=ticker_selector,
        ):
            tkr = ticker.strip().upper()
            if not tkr or tkr in seen:
                continue
            seen.add(tkr)
            out.append((name, tkr))
            if len(out) >= min_needed:
                return out

        if len(out) == last_count:
            stagnant_passes += 1
        else:
            stagnant_passes = 0
        last_count = len(out)
        if stagnant_passes >= 2:
            break

    # Final pass at the top to catch early rows if they were virtualized out
    # Final pass at top
    if container is not None:
        driver.execute_script("arguments[0].scrollTo(0, 0);", container)
    else:
        driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(pause)
    for name, ticker in _collect_visible_pairs(
        driver,
        row_selector=row_selector,
        name_selector=name_selector,
        ticker_selector=ticker_selector,
    ):
        tkr = ticker.strip().upper()
        if tkr and tkr not in seen:
            seen.add(tkr)
            out.append((name, tkr))
            if len(out) >= min_needed:
                break

    return out


def fetch_fortune_company_rows(
    first_needed: int,
    *,
    url: str,
    row_selector: str,
    name_selector: str,
    ticker_selector: str,
    wait_selector: str | None = None,
    scroll_container_selector: str | None = None,
) -> List[Tuple[str, str]]:
    """Navigate to a page and extract at least ``first_needed`` (name, ticker) pairs.

    The caller supplies stable CSS selectors for rows and for locating the
    company name and ticker within each row. This function handles headless
    browsing and robust scrolling through virtualized content.
    """

    with chrome() as driver:
        driver.get(url)
        anchor_selector = wait_selector or row_selector
        WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, anchor_selector))
        )

        # Best-effort: dismiss cookie/privacy banners that may block scrolling
        try:
            # OneTrust common id
            el = driver.find_element(By.CSS_SELECTOR, "#onetrust-accept-btn-handler")
            el.click()
            time.sleep(0.2)
        except Exception:
            try:
                # Generic accept button by text (XPath)
                el = driver.find_element(
                    By.XPATH,
                    "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'accept')]",
                )
                el.click()
                time.sleep(0.2)
            except Exception:
                pass

        pairs = _infinite_scroll_collect(
            driver,
            row_selector=row_selector,
            name_selector=name_selector,
            ticker_selector=ticker_selector,
            min_needed=first_needed,
            scroll_container_selector=scroll_container_selector,
        )
        return pairs


def _harvest_grid_rows(
    container, *, max_iters: int = 300, pause: float = 0.12
) -> List[Tuple[str, str, str]]:
    """Iteratively scroll the virtualized grid container and harvest rows.

    Returns a list of tuples (rank, company, ticker) as text values.
    Inspired by the approach in commit 93c2ba0196 (pre-refactor script).
    """
    from selenium.webdriver.common.by import By  # local import

    results: List[Tuple[str, str, str]] = []
    seen_ranks: set[str] = set()

    # Try to locate inner wrapper if present
    try:
        wrapper = container.find_element(By.CSS_SELECTOR, "div[style*='position: relative']")
    except Exception:
        wrapper = container

    prev_count = -1
    stagnation = 0
    for _ in range(max_iters):
        rows = wrapper.find_elements(By.CSS_SELECTOR, "div[style*='translateY']")
        for row in rows:
            cells = row.find_elements(By.XPATH, "./div")
            if len(cells) < 3:
                continue
            rank_txt = cells[0].text.strip()
            company_txt = cells[1].text.strip()
            ticker_txt = cells[2].text.strip()
            if rank_txt.isdigit() and company_txt and ticker_txt:
                if rank_txt not in seen_ranks:
                    seen_ranks.add(rank_txt)
                    results.append((rank_txt, company_txt, ticker_txt))
        # advance scroll
        try:
            scroll_top, client_h, scroll_h = container.get_property("scrollTop"), container.get_property("clientHeight"), container.get_property("scrollHeight")
            step = int((client_h or 800) * 0.9) or 600
            next_top = min((scroll_top or 0) + step, (scroll_h or 0))
            container_parent = container
        except Exception:
            container_parent = None
        if container_parent is not None:
            container_parent.parent.execute_script("arguments[0].scrollTo(0, arguments[1]);", container_parent, next_top)
        else:
            container.parent.execute_script("arguments[0].scrollTop = arguments[0].scrollTop + 1000;", container)

        time.sleep(pause)
        if len(seen_ranks) == prev_count:
            stagnation += 1
        else:
            stagnation = 0
        prev_count = len(seen_ranks)
        if stagnation >= 15:
            break

    return results


def fetch_us500_fortune_pairs(first_needed: int) -> List[Tuple[str, str]]:
    """Fetch Fortune company rows from us500.com using DOM grid harvesting.

    - Iterates page=1..N (up to what's necessary)
    - Scrolls the grid's overflow container to accumulate visible rows
    - Extracts rank/company/ticker columns from the first three cells
    """

    out_pairs: List[Tuple[str, str]] = []
    with chrome() as driver:
        from selenium.webdriver.common.by import By  # local import
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        page = 1
        max_pages = max(4, int((first_needed // 50) + 2))
        while len(out_pairs) < first_needed and page <= max_pages:
            url = f"https://us500.com/fortune-500-companies?page={page}"
            driver.get(url)
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.overflow-auto"))
                )
            except Exception:
                break

            container = driver.find_element(By.CSS_SELECTOR, "div.overflow-auto")
            rows = _harvest_grid_rows(container)
            for rank_txt, company, ticker in rows:
                tkr = ticker.strip()
                if not tkr:
                    continue
                out_pairs.append((company, tkr))
                if len(out_pairs) >= first_needed:
                    break
            page += 1

    # Deduplicate while preserving order
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for name, tkr in out_pairs:
        if tkr not in seen:
            seen.add(tkr)
            deduped.append((name, tkr))
        if len(deduped) >= first_needed:
            break
    return deduped
