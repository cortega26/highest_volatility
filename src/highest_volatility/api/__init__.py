"""Expose the cache API under the :mod:`highest_volatility` namespace."""

from highest_volatility.app.api import app

__all__ = ["app"]
