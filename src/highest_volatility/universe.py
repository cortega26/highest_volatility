"""Build a validated ticker universe using Selenium Stealth only.

This module is the single source of truth for collecting the Fortune list
tickers. It intentionally avoids any JSON endpoints and relies on a
headless Selenium browser hardened with selenium-stealth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import re

import pandas as pd
import yfinance as yf

from highest_volatility.sources.selenium_universe import fetch_us500_fortune_pairs
from highest_volatility.storage.ticker_cache import load_cached_fortune, save_cached_fortune
from highest_volatility.ingest.tickers import normalize_ticker

from highest_volatility.errors import CacheError, HVError, IntegrationError, wrap_error
from highest_volatility.logging import get_logger, log_exception


logger = get_logger(__name__, component="universe")


@dataclass
class FortuneData:
    """Container for intermediate Fortune list state."""

    frame: pd.DataFrame

    def copy(self) -> "FortuneData":
        return FortuneData(self.frame.copy())


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


def _load_cached_fortune(
    *, first_n_fortune: int, ticker_cache_days: int
) -> FortuneData | None:
    """Return cached Fortune data when available."""

    min_rows = min(100, first_n_fortune)
    fortune_df = load_cached_fortune(max_age_days=ticker_cache_days, min_rows=min_rows)
    if fortune_df is None:
        return None

    try:
        print(f"      Using cached Fortune list ({len(fortune_df)} rows)")
    except Exception:  # pragma: no cover - logging only
        pass
    logger.info({"event": "fortune_cache_used", "rows": len(fortune_df)})
    return FortuneData(fortune_df)


def _scrape_fortune(*, first_n_fortune: int) -> FortuneData:
    """Fetch the Fortune list using Selenium Stealth and persist to cache."""

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
        except Exception:  # pragma: no cover - logging only
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

    return FortuneData(fortune_df)


def _normalize_fortune(data: FortuneData) -> FortuneData:
    """Sort by rank when available and attach normalized tickers."""

    fortune_df = data.frame
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

    return FortuneData(fortune_df)


def _deduplicate_symbols(data: FortuneData) -> tuple[list[str], list[str]]:
    """Return ordered companies and tickers using normalized symbols."""

    tickers: List[str] = []
    companies: List[str] = []
    seen = set()
    for _, row in data.frame.iterrows():
        name = str(row["company"]).strip()
        tkr = str(row["normalized_ticker"]).strip()
        if not re.fullmatch(r"[A-Z]+[A-Z0-9.-]*", tkr):
            continue
        if not tkr or tkr in seen:
            continue
        seen.add(tkr)
        companies.append(name)
        tickers.append(tkr)

    return companies, tickers


def _validate_history(tickers: Iterable[str], *, validate: bool) -> set[str]:
    """Return symbols with sufficient history based on ``validate`` flag."""

    if validate:
        return set(_validate_tickers_have_history(list(tickers)))
    return set(tickers)


def _align_ranks(
    data: FortuneData,
    *,
    companies: Iterable[str],
    tickers: Iterable[str],
) -> pd.DataFrame:
    """Align Fortune rankings with the filtered ticker universe."""

    final_tickers = list(tickers)
    final_companies = list(companies)
    try:
        base = data.frame.drop_duplicates("normalized_ticker", keep="first")
        base = base.set_index("normalized_ticker")
        sel = base.loc[final_tickers, ["rank", "company"]]
        fortune = sel.assign(ticker=sel.index).reset_index(drop=True)
    except Exception as exc:
        error = wrap_error(
            exc,
            IntegrationError,
            message="Failed to align Fortune rankings",
            context={"tickers": len(final_tickers)},
        )
        log_exception(logger, error, event="fortune_alignment_failed")
        ranks = list(range(1, len(final_tickers) + 1))
        fortune = pd.DataFrame(
            {
                "rank": ranks,
                "company": final_companies,
                "ticker": final_tickers,
            }
        )
    return fortune


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

    data: FortuneData | None = None
    if use_ticker_cache:
        data = _load_cached_fortune(
            first_n_fortune=first_n_fortune, ticker_cache_days=ticker_cache_days
        )

    if data is None:
        data = _scrape_fortune(first_n_fortune=first_n_fortune)

    data = _normalize_fortune(data)
    companies, tickers = _deduplicate_symbols(data)

    valid = _validate_history(tickers, validate=validate)

    final_companies: List[str] = []
    final_tickers: List[str] = []
    for name, tkr in zip(companies, tickers):
        if tkr in valid:
            final_companies.append(name)
            final_tickers.append(tkr)
        if len(final_tickers) >= first_n_fortune:
            break

    fortune = _align_ranks(
        data,
        companies=final_companies,
        tickers=final_tickers,
    )
    return final_tickers, fortune


__all__ = ["build_universe"]
