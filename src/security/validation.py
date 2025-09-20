"""Input validation helpers shared across security-sensitive boundaries."""

from __future__ import annotations

import re
from typing import List

from highest_volatility.errors import ValidationError

TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,10}$")
INTERVAL_PATTERN = re.compile(r"^[0-9]+(?:wk|mo|[smhdw])$")
FORMAT_CHOICES = {"json", "parquet"}


class SanitizationError(ValidationError):
    """Raised when user supplied data fails validation."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message, context={"field": field} if field else None)


def sanitize_single_ticker(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if not normalized or not TICKER_PATTERN.fullmatch(normalized):
        raise SanitizationError("Ticker contains invalid characters.", field="ticker")
    return normalized


def sanitize_multiple_tickers(raw: str) -> List[str]:
    tickers = [sanitize_single_ticker(part) for part in raw.split(",") if part.strip()]
    if not tickers:
        raise SanitizationError("At least one ticker must be supplied.", field="tickers")
    return tickers


def sanitize_interval(interval: str) -> str:
    normalized = interval.strip().lower()
    if not normalized or not INTERVAL_PATTERN.fullmatch(normalized):
        raise SanitizationError("Interval is invalid.", field="interval")
    return normalized


def sanitize_download_format(fmt: str) -> str:
    normalized = fmt.strip().lower()
    if normalized not in FORMAT_CHOICES:
        raise SanitizationError("Format is invalid.", field="format")
    return normalized


def sanitize_metric(metric: str) -> str:
    normalized = metric.strip().lower()
    if not normalized or not re.fullmatch(r"[a-z0-9_]+", normalized):
        raise SanitizationError("Metric key contains invalid characters.", field="metric")
    return normalized
