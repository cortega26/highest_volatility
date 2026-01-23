"""Streamlit application entry point for Highest Volatility.

The app maintains an "analysis ready" flag in ``st.session_state`` so that
display-only interactions (such as adjusting chart selections) do not
invalidate previously downloaded data. The flag is cleared whenever the
sidebar configuration changes and is set again when users press
**Run analysis**.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if not __package__:
    # Allow running via ``streamlit run`` without requiring ``pip install -e .``.
    project_src = Path(__file__).resolve().parents[3]
    if project_src.name != "src":
        project_src = project_src / "src"
    sys.path.insert(0, str(project_src))

import pandas as pd
import altair as alt
import streamlit as st

CACHE_BUSTER = "ticker-normalization-v2"

from highest_volatility.app.cli import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_DAYS,
    DEFAULT_TOP_N,
    INTERVAL_CHOICES,
    METRIC_CHOICES,
)
from highest_volatility.app.analysis_state import update_analysis_ready_flag
from highest_volatility.app.ui_helpers import (
    SanitizedPrices,
    prepare_metric_table,
    sanitize_price_matrix,
)
from highest_volatility.compute.metrics import (
    max_drawdown,
    metric_display_name,
    rolling_volatility,
)
from highest_volatility.ingest.prices import download_price_history
from highest_volatility.universe import build_universe


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Immutable container describing the current UI configuration."""

    top_n: int
    lookback_days: int
    interval: str
    metric_key: str
    min_days: int
    prepost: bool
    validate_universe: bool
    async_fetch: bool

    @property
    def matrix_mode(self) -> str:
        """Return the price download mode based on async preference."""

        return "async" if self.async_fetch else "batch"

    def signature(self) -> tuple[object, ...]:
        """Return a hashable representation used for session comparisons."""

        return (
            self.top_n,
            self.lookback_days,
            self.interval,
            self.metric_key,
            self.min_days,
            self.prepost,
            self.validate_universe,
            self.async_fetch,
        )


@dataclass(frozen=True, slots=True)
class UniverseData:
    """Result of building the Fortune universe."""

    tickers: list[str]
    fortune: pd.DataFrame | None

    @property
    def is_empty(self) -> bool:
        return not self.tickers

    def fortune_view(self) -> pd.DataFrame:
        if self.fortune is None or self.fortune.empty:
            return pd.DataFrame()
        columns = [col for col in ("rank", "company", "ticker") if col in self.fortune.columns]
        if not columns:
            return pd.DataFrame()
        return self.fortune.loc[:, columns]


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    return max(min(value, maximum), minimum)


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
    metric_key = st.selectbox(
        "Metric", METRIC_CHOICES, index=0, format_func=metric_display_name
    )
    min_days_label = "Minimum observations per ticker"
    min_days_key = min_days_label
    min_days_max = lookback_days
    min_days_default = _clamp_int(
        DEFAULT_MIN_DAYS,
        minimum=10,
        maximum=min_days_max,
    )
    if min_days_key in st.session_state:
        try:
            existing_min_days = int(st.session_state[min_days_key])
        except (TypeError, ValueError):
            existing_min_days = min_days_default
        st.session_state[min_days_key] = _clamp_int(
            existing_min_days,
            minimum=10,
            maximum=min_days_max,
        )
    min_days = st.number_input(
        min_days_label,
        min_value=10,
        max_value=min_days_max,
        value=min_days_default,
        step=5,
        key=min_days_key,
    )
    prepost = st.checkbox(
        "Include pre/post-market data",
        value=False,
        help="Enable to include extended-hours data for intraday intervals.",
    )
    validate_universe = st.checkbox(
        "Validate tickers via Selenium",
        value=False,
        help=(
            "Enable to revalidate tickers via Yahoo Finance, which fetches fresh data and adds"
            " extra startup time."
        ),
    )
    async_fetch = st.checkbox(
        "Fetch prices asynchronously",
        value=False,
        help="Use the experimental async HTTP fetcher instead of batch downloads.",
    )
    run_analysis = st.button("Run analysis", type="primary")

config = AnalysisConfig(
    top_n=top_n,
    lookback_days=lookback_days,
    interval=interval,
    metric_key=metric_key,
    min_days=min_days,
    prepost=prepost,
    validate_universe=validate_universe,
    async_fetch=async_fetch,
)


@st.cache_data(show_spinner=False)
def _build_universe_cached(
    top_n: int, validate: bool, cache_buster: str = CACHE_BUSTER
) -> tuple[list[str], pd.DataFrame]:
    del cache_buster  # Ensures cache key invalidation when the token changes.
    tickers, fortune = build_universe(top_n, validate=validate)
    return tickers, fortune


@st.cache_data(show_spinner=False)
def _download_prices_cached(
    tickers: Sequence[str],
    lookback_days: int,
    interval: str,
    prepost: bool,
    matrix_mode: str,
    cache_buster: str = CACHE_BUSTER,
) -> pd.DataFrame:
    del cache_buster  # Ensures cache key invalidation when the token changes.
    return download_price_history(
        list(tickers),
        lookback_days,
        interval=interval,
        prepost=prepost,
        matrix_mode=matrix_mode,
    )


def _ensure_analysis_requested(analysis_ready: bool) -> bool:
    if analysis_ready:
        return True
    st.info("Adjust the configuration and click **Run analysis** to fetch data.")
    return False


def _load_universe(config: AnalysisConfig) -> UniverseData | None:
    with st.spinner("Building Fortune universe…"):
        tickers, fortune = _build_universe_cached(
            config.top_n,
            config.validate_universe,
            cache_buster=CACHE_BUSTER,
        )
    universe = UniverseData(list(tickers), fortune)
    if universe.is_empty:
        st.warning("No tickers were returned by the universe builder.")
        return None
    return universe


def _escape_spreadsheet_formula(value: object) -> object:
    """Neutralise spreadsheet formulas by prefixing suspicious values."""

    if not isinstance(value, str) or not value:
        return value
    first_char = value[0]
    if first_char in {"=", "+", "-", "@"}:
        return "'" + value
    if first_char in {"\t", "\n"}:
        return " " + value
    return value


def _normalise_hex_color(value: str | None) -> str | None:
    """Return a normalised 6-digit hex colour (without leading '#')."""

    if not value:
        return None
    colour = value.strip().lstrip("#")
    if len(colour) == 3:
        colour = "".join(component * 2 for component in colour)
    if len(colour) != 6:
        return None
    try:
        int(colour, 16)
    except ValueError:
        return None
    return colour.lower()


def _resolve_highlight_css(get_option=None) -> str:
    """Build a theme-aware CSS snippet for highlighted table rows."""

    option_getter = get_option or st.get_option
    theme_base = (option_getter("theme.base") or "light").lower()
    primary = _normalise_hex_color(option_getter("theme.primaryColor"))
    text_colour_option = option_getter("theme.textColor")
    default_text_colour = "#262730" if theme_base == "light" else "#FAFAFA"
    text_colour = text_colour_option or default_text_colour

    if primary is None:
        background_colour = "#f0f4c3" if theme_base == "light" else "#2c2c2c"
        return f"background-color: {background_colour}; color: {text_colour};"

    red = int(primary[0:2], 16)
    green = int(primary[2:4], 16)
    blue = int(primary[4:6], 16)
    alpha = 0.18 if theme_base == "light" else 0.45
    return f"background-color: rgba({red}, {green}, {blue}, {alpha}); color: {text_colour};"


def _render_universe_section(universe: UniverseData) -> None:
    view = universe.fortune_view()
    if view.empty:
        return
    st.subheader("Fortune universe")
    st.dataframe(view, use_container_width=True)
    escaped_view = view.map(_escape_spreadsheet_formula)
    st.download_button(
        "Download universe (CSV)",
        data=escaped_view.to_csv(index=False),
        file_name="fortune_universe.csv",
        mime="text/csv",
    )


def _download_prices_for(universe: UniverseData, config: AnalysisConfig) -> pd.DataFrame:
    with st.spinner("Downloading price history…"):
        return _download_prices_cached(
            tuple(universe.tickers),
            config.lookback_days,
            config.interval,
            config.prepost,
            config.matrix_mode,
            cache_buster=CACHE_BUSTER,
        )


def _sanitize_prices_for(
    prices: pd.DataFrame, config: AnalysisConfig, universe: UniverseData
) -> SanitizedPrices:
    return sanitize_price_matrix(
        prices,
        min_days=config.min_days,
        tickers=universe.tickers,
    )


def _display_drop_messages(sanitized: SanitizedPrices, config: AnalysisConfig) -> None:
    messages = sanitized.summarize_drops(min_days=config.min_days)
    if messages:
        st.info("\n".join(messages))


def _compute_metric_table(
    config: AnalysisConfig, sanitized: SanitizedPrices, fortune: pd.DataFrame | None
) -> pd.DataFrame:
    try:
        return prepare_metric_table(
            sanitized.filtered,
            metric_key=config.metric_key,
            min_days=config.min_days,
            interval=config.interval,
            tickers=sanitized.available_tickers,
            fortune=fortune,
            close_prices=sanitized.close_only,
        )
    except Exception as exc:  # pragma: no cover - surface to UI
        st.error(f"Failed to compute metric: {exc}")
        return pd.DataFrame()


def _render_metric_table(table: pd.DataFrame, metric_key: str) -> None:
    metric_label = metric_display_name(metric_key)
    st.subheader(f"Top tickers by {metric_label}")
    highlight_n = min(10, len(table))
    highlight_css = _resolve_highlight_css()
    styled_table = table.style.apply(
        lambda row: [highlight_css if row.name < highlight_n else "" for _ in row],
        axis=1,
    )
    st.dataframe(styled_table, use_container_width=True)
    escaped_table = table.map(_escape_spreadsheet_formula)
    st.download_button(
        "Download metric ranking (CSV)",
        data=escaped_table.to_csv(index=False),
        file_name=f"metric_ranking_{metric_key}.csv",
        mime="text/csv",
    )


def _select_visualised_tickers(
    table: pd.DataFrame, sanitized: SanitizedPrices
) -> list[str]:
    available = sanitized.available_tickers
    if not available or "ticker" not in table.columns:
        return []
    default_selection = table["ticker"].head(min(5, len(table))).tolist()
    return st.multiselect(
        "Tickers to visualize",
        options=available,
        default=[ticker for ticker in default_selection if ticker in available],
    )


def _render_price_history(close_only: pd.DataFrame, selected_tickers: list[str]) -> None:
    if not selected_tickers:
        st.info("Select at least one ticker to display the price chart.")
        return
    price_history = (
        close_only[selected_tickers]
        .rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="close")
    )
    price_chart = (
        alt.Chart(price_history)
        .mark_line()
        .encode(x="date:T", y="close:Q", color="ticker:N")
        .properties(height=400)
    )
    st.altair_chart(price_chart, use_container_width=True)


def _render_drawdown_history(close_only: pd.DataFrame, selected_tickers: list[str]) -> None:
    if not selected_tickers:
        st.info("Select at least one ticker to review drawdowns.")
        return
    drawdown = close_only[selected_tickers] / close_only[selected_tickers].cummax() - 1
    drawdown_long = (
        drawdown.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="drawdown")
    )
    drawdown_chart = (
        alt.Chart(drawdown_long)
        .mark_line()
        .encode(x="date:T", y=alt.Y("drawdown:Q", title="Drawdown"), color="ticker:N")
        .properties(height=400)
    )
    st.altair_chart(drawdown_chart, use_container_width=True)
    drawdown_summary = max_drawdown(close_only[selected_tickers])
    st.dataframe(drawdown_summary, use_container_width=True)


def _render_rolling_volatility(close_only: pd.DataFrame, selected_tickers: list[str]) -> None:
    if not selected_tickers:
        st.info("Select at least one ticker to inspect rolling volatility.")
        return
    rolling = rolling_volatility(close_only[selected_tickers])
    rolling_chart = (
        alt.Chart(rolling)
        .mark_line()
        .encode(
            x="date:T",
            y=alt.Y("rolling_volatility:Q", title="Annualized Volatility"),
            color="ticker:N",
            strokeDash=alt.StrokeDash("window:N", title="Window"),
        )
        .properties(height=400)
    )
    st.altair_chart(rolling_chart, use_container_width=True)


def _render_visualisations(close_only: pd.DataFrame, selected_tickers: list[str]) -> None:
    tabs = st.tabs(["Metrics & Prices", "Drawdowns", "Rolling Volatility"])
    with tabs[0]:
        st.markdown("#### Price history")
        _render_price_history(close_only, selected_tickers)
    with tabs[1]:
        st.markdown("#### Drawdown history")
        _render_drawdown_history(close_only, selected_tickers)
    with tabs[2]:
        st.markdown("#### Rolling volatility")
        _render_rolling_volatility(close_only, selected_tickers)


def _render(config: AnalysisConfig, analysis_ready: bool) -> None:
    if not _ensure_analysis_requested(analysis_ready):
        return

    universe = _load_universe(config)
    if universe is None:
        return

    _render_universe_section(universe)

    prices = _download_prices_for(universe, config)
    if prices.empty:
        st.warning("Price history request returned no data for the selected options.")
        return

    sanitized = _sanitize_prices_for(prices, config, universe)
    if not sanitized.has_close_data:
        st.warning(
            "All tickers were filtered out due to insufficient history or duplicate data."
        )
        return

    _display_drop_messages(sanitized, config)

    table = _compute_metric_table(config, sanitized, universe.fortune)
    if table.empty:
        st.warning("Metric computation returned no rows for the selected configuration.")
        return

    _render_metric_table(table, config.metric_key)

    selected_tickers = _select_visualised_tickers(table, sanitized)

    _render_visualisations(sanitized.close_only, selected_tickers)


analysis_ready = update_analysis_ready_flag(
    st.session_state,
    config.signature(),
    run_analysis,
)


_render(config, analysis_ready)
