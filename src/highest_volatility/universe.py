"""Build a validated ticker universe using Selenium Stealth only.

This module is the single source of truth for collecting the Fortune list
tickers. It intentionally avoids any JSON endpoints and relies on a
headless Selenium browser hardened with selenium-stealth.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import yfinance as yf

from highest_volatility.sources.selenium_universe import (
    fetch_us500_fortune_pairs,
)
from highest_volatility.storage.ticker_cache import (
    load_cached_fortune,
    save_cached_fortune,
)
from highest_volatility.ingest.tickers import normalize_ticker
import re

from highest_volatility.errors import CacheError, HVError, IntegrationError, wrap_error
from highest_volatility.logging import get_logger, log_exception


logger = get_logger(__name__, component="universe")


def _validate_tickers_have_history(
    tickers: List[str], *, min_days: int = 1
) -> List[str]:
    """Return subset of ``tickers`` that have recent price history.

    Downloads a short recent window to quickly filter out symbols that do not
    produce data from Yahoo Finance.
    """

    if not tickers:
        return []

    valid: List[str] = []
    # Batch to avoid overly long URLs
    BATCH = 150
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i : i + BATCH]
        data = yf.download(
            " ".join(batch), period="5d", progress=False, auto_adjust=True
        )
        if data is None or len(data) == 0:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data.to_frame(name="Close")
        counts = close.count(axis=0)
        for tkr, cnt in counts.items():
            if cnt >= min_days:
                valid.append(str(tkr))
    # Preserve input order
    order = {t: idx for idx, t in enumerate(tickers)}
    valid_sorted = sorted(set(valid), key=lambda t: order.get(t, 10_000_000))
    return valid_sorted


def build_universe(
    first_n_fortune: int,
    *,
    validate: bool = True,
    use_ticker_cache: bool = True,
    ticker_cache_days: int = 30,
) -> tuple[list[str], pd.DataFrame]:
    """Build a universe of at least ``first_n_fortune`` public tickers.

    - Scrapes the Fortune listing via Selenium Stealth
    - De-duplicates and normalizes tickers for Yahoo Finance
    - Validates tickers have recent price history
    - Returns a DataFrame with columns: ``rank``, ``company``, ``ticker``
    """

    # Try cached Fortune list first
    fortune_df: pd.DataFrame | None = None
    if use_ticker_cache:
        # Accept a partially smaller cache; we'll filter/validate later.
        min_rows = min(100, first_n_fortune)
        fortune_df = load_cached_fortune(max_age_days=ticker_cache_days, min_rows=min_rows)
        if fortune_df is not None:
            try:
                print(f"      Using cached Fortune list ({len(fortune_df)} rows)")
            except Exception:
                pass
            logger.info({"event": "fortune_cache_used", "rows": len(fortune_df)})

    if fortune_df is None:
        # Fetch via Selenium and then persist to cache
        target_to_fetch = max(first_n_fortune * 2, first_n_fortune + 100)
        try:
            pairs: List[Tuple[str, str]] = fetch_us500_fortune_pairs(target_to_fetch)
        except HVError as error:  # pragma: no cover - defensive
            log_exception(logger, error, event="fortune_scrape_failed")
            raise
        except Exception as exc:  # pragma: no cover - defensive
            error = wrap_error(
                exc,
                IntegrationError,
                message="Failed to scrape Fortune list",
                context={"target": target_to_fetch},
            )
            log_exception(logger, error, event="fortune_scrape_failed")
            raise error
        # Deduplicate and build DataFrame with rank order as encountered
        seen = set()
        recs: List[dict] = []
        rank = 1
        for name, tkr in pairs:
            if tkr in seen:
                continue
            seen.add(tkr)
            recs.append({"rank": rank, "company": name, "ticker": tkr})
            rank += 1
            if len(recs) >= target_to_fetch:
                break
        fortune_df = pd.DataFrame(recs)
        try:
            save_cached_fortune(fortune_df)
            try:
                print(f"      Scraped and cached Fortune list ({len(fortune_df)} rows)")
            except Exception:
                pass
            logger.info({"event": "fortune_cache_saved", "rows": len(fortune_df)})
        except HVError as error:  # pragma: no cover - logging path
            log_exception(logger, error, event="fortune_cache_save_failed")
        except Exception as exc:  # pragma: no cover - defensive
            error = wrap_error(
                exc,
                CacheError,
                message="Failed to persist Fortune cache",
                context={"rows": len(fortune_df)},
            )
            log_exception(logger, error, event="fortune_cache_save_failed")
    # Ensure rank ordering if present
    if "rank" in fortune_df.columns:
        try:
            fortune_df = fortune_df.sort_values("rank").reset_index(drop=True)
        except Exception as exc:  # pragma: no cover - defensive
            error = wrap_error(
                exc,
                IntegrationError,
                message="Failed to sort Fortune list",
            )
            log_exception(logger, error, event="fortune_sort_failed")

    # Carry both raw and normalized tickers for downstream alignment
    try:
        fortune_df = fortune_df.copy()
        fortune_df["ticker"] = fortune_df["ticker"].astype(str).str.strip()
        fortune_df["normalized_ticker"] = fortune_df["ticker"].map(normalize_ticker)
    except Exception as exc:  # pragma: no cover - defensive
        error = wrap_error(
            exc,
            IntegrationError,
            message="Failed to normalize Fortune tickers",
        )
        log_exception(logger, error, event="fortune_normalize_failed")
        fortune_df["normalized_ticker"] = fortune_df["ticker"].map(normalize_ticker)

    # Deduplicate by normalized ticker while preserving order
    tickers: List[str] = []
    companies: List[str] = []
    seen = set()
    for _, row in fortune_df.iterrows():
        name = str(row["company"]).strip()
        tkr = str(row["normalized_ticker"]).strip()
        # Filter to valid Yahoo-style symbols early
        if not re.fullmatch(r"[A-Z]+[A-Z0-9.-]*", tkr):
            continue
        if not tkr or tkr in seen:
            continue
        seen.add(tkr)
        companies.append(name)
        tickers.append(tkr)

    # Optionally validate via recent price history for tradability
    if validate:
        valid = set(_validate_tickers_have_history(tickers))
    else:
        valid = set(tickers)

    final_companies: List[str] = []
    final_tickers: List[str] = []
    for name, tkr in zip(companies, tickers):
        if tkr in valid:
            final_companies.append(name)
            final_tickers.append(tkr)
        if len(final_tickers) >= first_n_fortune:
            break

    # Preserve actual Fortune ranks if available
    try:
        base = fortune_df.drop_duplicates("normalized_ticker", keep="first")
        base = base.set_index("normalized_ticker")
        sel = base.loc[final_tickers, ["rank", "company"]]
        fortune = sel.assign(ticker=sel.index).reset_index(drop=True)
    except Exception as exc:
        # Fallback to enumerated ranks
        error = wrap_error(
            exc,
            IntegrationError,
            message="Failed to align Fortune rankings",
            context={"tickers": len(final_tickers)},
        )
        log_exception(logger, error, event="fortune_alignment_failed")
        ranks = list(range(1, len(final_tickers) + 1))
        fortune = pd.DataFrame({
            "rank": ranks,
            "company": final_companies,
            "ticker": final_tickers,
        })
    return final_tickers, fortune


__all__ = ["build_universe"]
