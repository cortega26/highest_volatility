"""Streamlit application entry point for Highest Volatility."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import streamlit as st

from highest_volatility.app.cli import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_DAYS,
    DEFAULT_TOP_N,
    INTERVAL_CHOICES,
    METRIC_CHOICES,
)
from highest_volatility.app.ui_helpers import (
    prepare_metric_table,
    sanitize_price_matrix,
)
from highest_volatility.ingest.prices import download_price_history
from highest_volatility.universe import build_universe


st.set_page_config(page_title="Highest Volatility", layout="wide")

st.title("Highest Volatility Explorer")
st.caption(
    "Interactively explore Fortune-listed tickers using the built-in metrics."
)


with st.sidebar:
    st.header("Configuration")
    top_n = st.slider(
        "Universe size",
        min_value=10,
        max_value=500,
        step=10,
        value=DEFAULT_TOP_N,
        help="Number of Fortune companies to analyse.",
    )
    lookback_days = st.number_input(
        "Lookback window (days)",
        min_value=30,
        max_value=2000,
        value=DEFAULT_LOOKBACK_DAYS,
        step=5,
    )
    interval = st.selectbox("Interval", INTERVAL_CHOICES, index=0)
    metric_key = st.selectbox("Metric", METRIC_CHOICES, index=0)
    min_days = st.number_input(
        "Minimum observations per ticker",
        min_value=10,
        max_value=lookback_days,
        value=DEFAULT_MIN_DAYS,
        step=5,
    )
    prepost = st.checkbox(
        "Include pre/post-market data",
        value=False,
        help="Enable to include extended-hours data for intraday intervals.",
    )
    validate_universe = st.checkbox(
        "Validate tickers via Selenium",
        value=True,
        help="Disable to skip validation for faster startup (uses cached tickers when available).",
    )
    async_fetch = st.checkbox(
        "Fetch prices asynchronously",
        value=False,
        help="Use the experimental async HTTP fetcher instead of batch downloads.",
    )
    run_analysis = st.button("Run analysis", type="primary")


@st.cache_data(show_spinner=False)
def _build_universe_cached(
    top_n: int, validate: bool
) -> tuple[list[str], pd.DataFrame]:
    tickers, fortune = build_universe(top_n, validate=validate)
    return tickers, fortune


@st.cache_data(show_spinner=False)
def _download_prices_cached(
    tickers: Sequence[str],
    lookback_days: int,
    interval: str,
    prepost: bool,
    matrix_mode: str,
) -> pd.DataFrame:
    return download_price_history(
        list(tickers),
        lookback_days,
        interval=interval,
        prepost=prepost,
        matrix_mode=matrix_mode,
    )


def _render() -> None:
    if not run_analysis:
        st.info("Adjust the configuration and click **Run analysis** to fetch data.")
        return

    with st.spinner("Building Fortune universe…"):
        tickers, fortune = _build_universe_cached(top_n, validate_universe)

    if not tickers:
        st.warning("No tickers were returned by the universe builder.")
        return

    with st.spinner("Downloading price history…"):
        prices = _download_prices_cached(
            tuple(tickers),
            lookback_days,
            interval,
            prepost,
            "async" if async_fetch else "batch",
        )

    if prices.empty:
        st.warning("Price history request returned no data for the selected options.")
        return

    filtered_prices, close_only, dropped_short, dropped_duplicate, dropped_empty = (
        sanitize_price_matrix(prices, min_days=min_days, tickers=tickers)
    )

    if close_only.empty:
        st.warning(
            "All tickers were filtered out due to insufficient history or duplicate data."
        )
        return

    dropped_msgs = []
    if dropped_short:
        dropped_msgs.append(
            f"{len(dropped_short)} tickers lacked the required {min_days} observations."
        )
    if dropped_duplicate:
        dropped_msgs.append(
            f"{len(dropped_duplicate)} tickers were removed due to duplicate price series."
        )
    if dropped_empty:
        dropped_msgs.append(
            f"{len(dropped_empty)} tickers contained only empty values after cleaning."
        )
    if dropped_msgs:
        st.info("\n".join(dropped_msgs))

    try:
        table = prepare_metric_table(
            filtered_prices,
            metric_key=metric_key,
            min_days=min_days,
            interval=interval,
            tickers=close_only.columns,
            fortune=fortune,
            close_prices=close_only,
        )
    except Exception as exc:  # pragma: no cover - surface to UI
        st.error(f"Failed to compute metric: {exc}")
        return

    if table.empty:
        st.warning("Metric computation returned no rows for the selected configuration.")
        return

    metric_label = metric_key.replace("_", " ").title()
    st.subheader(f"Top tickers by {metric_label}")
    st.dataframe(table, use_container_width=True)


_render()
