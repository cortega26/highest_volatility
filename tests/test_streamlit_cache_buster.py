"""Regression tests for the Streamlit cache-buster wiring."""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

import pandas as pd

from tests.test_streamlit_app_bootstrap import (
    _DummyStreamlit,
    _run_streamlit_entry,
)


class _CachingStreamlit(_DummyStreamlit):
    """Streamlit stub that tracks cache keys for decorated functions."""

    def __init__(self) -> None:
        super().__init__()
        self.cache_keys: dict[str, list[tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]]]] = {}
        self.session_state: dict[str, Any] = {}

    def cache_data(
        self, *decorator_args: Any, **decorator_kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a caching decorator that records invocation keys."""

        del decorator_args, decorator_kwargs

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            storage: dict[
                tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]], Any
            ] = {}

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                key = (args, tuple(sorted(kwargs.items())))
                self.cache_keys.setdefault(func.__name__, []).append(key)
                if key not in storage:
                    storage[key] = func(*args, **kwargs)
                return storage[key]

            return wrapper

        return decorator


def test_cache_buster_participates_in_streamlit_cache(monkeypatch) -> None:
    streamlit_stub = _CachingStreamlit()
    module_globals = _run_streamlit_entry(monkeypatch, streamlit_stub)

    build_cached = module_globals["_build_universe_cached"]
    download_cached = module_globals["_download_prices_cached"]
    app_globals = build_cached.__wrapped__.__globals__
    cache_buster = app_globals["CACHE_BUSTER"]


    build_sig = inspect.signature(build_cached)
    download_sig = inspect.signature(download_cached)
    assert "cache_buster" in build_sig.parameters
    assert build_sig.parameters["cache_buster"].default == cache_buster
    assert "cache_buster" in download_sig.parameters
    assert download_sig.parameters["cache_buster"].default == cache_buster

    build_calls = 0

    def fake_build_universe(top_n: int, validate: bool):
        nonlocal build_calls
        build_calls += 1
        fortune = pd.DataFrame({"rank": [1], "company": ["Alpha"], "ticker": ["AAA"]})
        return ["AAA"], fortune

    download_calls = 0

    def fake_download_price_history(
        tickers: list[str],
        lookback_days: int,
        *,
        interval: str,
        prepost: bool,
        matrix_mode: str,
    ):
        nonlocal download_calls
        download_calls += 1
        return pd.DataFrame({ticker: [] for ticker in tickers})

    app_globals["build_universe"] = fake_build_universe
    app_globals["download_price_history"] = fake_download_price_history

    AnalysisConfig = app_globals["AnalysisConfig"]
    config = AnalysisConfig(
        top_n=1,
        lookback_days=30,
        interval=app_globals["INTERVAL_CHOICES"][0],
        metric_key=app_globals["METRIC_CHOICES"][0],
        min_days=10,
        prepost=False,
        validate_universe=False,
        async_fetch=False,
    )

    universe = app_globals["_load_universe"](config)
    assert build_calls == 1
    universe_key_kwargs = dict(streamlit_stub.cache_keys["_build_universe_cached"][0][1])
    assert universe_key_kwargs["cache_buster"] == cache_buster

    app_globals["_load_universe"](config)
    assert build_calls == 1  # Cached invocation

    build_cached(config.top_n, config.validate_universe, cache_buster="override-token")
    assert build_calls == 2
    override_key_kwargs = dict(streamlit_stub.cache_keys["_build_universe_cached"][-1][1])
    assert override_key_kwargs["cache_buster"] == "override-token"

    prices = app_globals["_download_prices_for"](universe, config)
    assert isinstance(prices, pd.DataFrame)
    assert download_calls == 1
    prices_key_kwargs = dict(streamlit_stub.cache_keys["_download_prices_cached"][0][1])
    assert prices_key_kwargs["cache_buster"] == cache_buster

    app_globals["_download_prices_for"](universe, config)
    assert download_calls == 1  # Cached invocation

    download_cached(
        tuple(universe.tickers),
        config.lookback_days,
        config.interval,
        config.prepost,
        config.matrix_mode,
        cache_buster="override-token",
    )
    assert download_calls == 2
    override_download_kwargs = dict(
        streamlit_stub.cache_keys["_download_prices_cached"][-1][1]
    )
    assert override_download_kwargs["cache_buster"] == "override-token"
