"""Volatility related computations."""

from .metrics import (
    TRADING_DAYS_PER_YEAR,
    TRADING_MINUTES_PER_DAY,
    annualized_volatility,
    additional_volatility_measures,
    daily_returns,
    periods_per_year,
    max_drawdown,
    rolling_volatility,
    sharpe_ratio,
)

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "TRADING_MINUTES_PER_DAY",
    "annualized_volatility",
    "additional_volatility_measures",
    "daily_returns",
    "periods_per_year",
    "max_drawdown",
    "rolling_volatility",
    "sharpe_ratio",
]


def __getattr__(name: str):
    if name == "volatility":  # pragma: no cover - defensive guardrail
        raise AttributeError(
            "highest_volatility.compute.volatility was removed; import from "
            "highest_volatility.compute.metrics instead."
        )
    raise AttributeError(f"module 'highest_volatility.compute' has no attribute {name!r}")
