"""Synthetic benchmark for :func:`additional_volatility_measures`."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from highest_volatility.compute.metrics import additional_volatility_measures, periods_per_year


@dataclass
class BenchmarkResult:
    label: str
    elapsed: float


def _baseline_additional_volatility_measures(
    raw: pd.DataFrame,
    tickers: List[str],
    *,
    min_periods: int = 2,
    interval: str = "1d",
    ewma_lambda: float = 0.94,
) -> pd.DataFrame:
    per_year = periods_per_year(interval)
    ln2 = np.log(2.0)
    results: List[Dict[str, float | str]] = []

    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = set(raw.columns.get_level_values(0))
        lv1 = set(raw.columns.get_level_values(1))
        if ("Open" not in lv0 and "Close" not in lv0) and ("Open" in lv1 or "Close" in lv1):
            raw = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)

    def _get_series(field: str, ticker: str) -> pd.Series | None:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                return raw[field][ticker].dropna()
            return raw[field].dropna()
        except Exception:
            return None

    for ticker in tickers:
        rec: Dict[str, float | str] = {"ticker": ticker}

        s_close = _get_series("Adj Close", ticker)
        if s_close is None:
            s_close = _get_series("Close", ticker)
        s_open = _get_series("Open", ticker)
        s_high = _get_series("High", ticker)
        s_low = _get_series("Low", ticker)

        candidates: Dict[str, pd.Series] = {}
        for key, series in {
            "close": s_close,
            "open": s_open,
            "high": s_high,
            "low": s_low,
        }.items():
            if isinstance(series, pd.Series) and not series.dropna().empty:
                candidates[key] = series.dropna()
        if not candidates or "close" not in candidates:
            continue

        df = pd.concat(candidates, axis=1).dropna(how="any")
        if df.shape[0] < min_periods:
            continue

        r_cc = np.log(df["close"] / df["close"].shift(1)).dropna()

        if {"high", "low"}.issubset(df.columns):
            hl = np.log(df["high"] / df["low"]) ** 2
            var = hl.mean() / (4.0 * ln2)
            if np.isfinite(var) and var >= 0:
                rec["parkinson_vol"] = float(np.sqrt(var * per_year))

        if {"open", "high", "low", "close"}.issubset(df.columns):
            log_hl = np.log(df["high"] / df["low"]) ** 2
            log_co = np.log(df["close"] / df["open"]) ** 2
            gk_var = 0.5 * log_hl.mean() - (2.0 * ln2 - 1.0) * log_co.mean()
            if np.isfinite(gk_var) and gk_var >= 0:
                rec["gk_vol"] = float(np.sqrt(gk_var * per_year))

            term_rs = (
                np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) +
                np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"])
            )
            rs_var = term_rs.mean()
            if np.isfinite(rs_var) and rs_var >= 0:
                rec["rs_vol"] = float(np.sqrt(rs_var * per_year))

            prev_close = df["close"].shift(1)
            r_o = np.log(df["open"] / prev_close).dropna()
            r_c = np.log(df["close"] / df["open"]).dropna()
            df_rs = df.loc[r_c.index]
            term_rs_d = (
                np.log(df_rs["high"] / df_rs["close"]) * np.log(df_rs["high"] / df_rs["open"]) +
                np.log(df_rs["low"] / df_rs["close"]) * np.log(df_rs["low"] / df_rs["open"])
            )
            sigma_o2 = r_o.var(ddof=1)
            sigma_c2 = r_c.var(ddof=1)
            sigma_rs = term_rs_d.mean()
            if (
                np.isfinite(sigma_o2)
                and np.isfinite(sigma_c2)
                and np.isfinite(sigma_rs)
            ):
                k = 0.34
                yz_var = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs
                if yz_var >= 0:
                    rec["yz_vol"] = float(np.sqrt(yz_var * per_year))

        if r_cc.shape[0] >= min_periods:
            var = r_cc.var(ddof=1) if r_cc.shape[0] >= 2 else float(r_cc.iloc[-1] ** 2)
            for x in r_cc.iloc[-min_periods:]:
                var = ewma_lambda * var + (1.0 - ewma_lambda) * (x * x)
            if var >= 0:
                rec["ewma_vol"] = float(np.sqrt(var * per_year))
            mad = np.median(np.abs(r_cc - np.median(r_cc)))
            rec["mad_vol"] = float(1.4826 * mad * np.sqrt(per_year))

        results.append(rec)

    out = pd.DataFrame(results)
    if out.empty:
        return out
    cols = [
        "ticker",
        "parkinson_vol",
        "gk_vol",
        "rs_vol",
        "yz_vol",
        "ewma_vol",
        "mad_vol",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols]


def _generate_prices(n_rows: int, tickers: List[str], seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    data: Dict[tuple[str, str], np.ndarray] = {}
    for ticker in tickers:
        base = 50 + 5 * rng.random()
        close = base + rng.normal(scale=1.0, size=n_rows).cumsum()
        high = close + rng.uniform(0.1, 0.9, size=n_rows)
        low = close - rng.uniform(0.1, 0.9, size=n_rows)
        open_ = close + rng.normal(scale=0.4, size=n_rows)
        adj_close = close * (1 + rng.normal(scale=0.002, size=n_rows))
        frame = {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": adj_close,
        }
        for field, values in frame.items():
            data[(field, ticker)] = values
    columns = pd.MultiIndex.from_tuples(data.keys(), names=["Field", "Ticker"])
    return pd.DataFrame(data, index=index, columns=columns).sort_index(axis=1)


def _timeit(func, *args, repeats: int, **kwargs) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        func(*args, **kwargs)
    return time.perf_counter() - start


def run_benchmark(
    n_rows: int,
    n_tickers: int,
    repeats: int,
    *,
    min_periods: int,
    interval: str,
    ewma_lambda: float,
) -> list[BenchmarkResult]:
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    prices = _generate_prices(n_rows, tickers)
    args = (prices, tickers)
    kwargs = dict(min_periods=min_periods, interval=interval, ewma_lambda=ewma_lambda)

    baseline_time = _timeit(_baseline_additional_volatility_measures, *args, repeats=repeats, **kwargs)
    optimized_time = _timeit(additional_volatility_measures, *args, repeats=repeats, **kwargs)

    return [
        BenchmarkResult("baseline", baseline_time / repeats),
        BenchmarkResult("optimized", optimized_time / repeats),
    ]


def _format_table(results: list[BenchmarkResult]) -> str:
    header = "label\tavg_ms"
    rows = [f"{r.label}\t{r.elapsed * 1_000:.3f}" for r in results]
    return "\n".join([header, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=252 * 4, help="Number of rows in synthetic prices")
    parser.add_argument("--tickers", type=int, default=50, help="Number of tickers to benchmark")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repetitions for timing")
    parser.add_argument("--min-periods", type=int, default=2, dest="min_periods")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--ewma-lambda", type=float, default=0.94, dest="ewma_lambda")
    args = parser.parse_args()

    results = run_benchmark(
        args.rows,
        args.tickers,
        args.repeats,
        min_periods=args.min_periods,
        interval=args.interval,
        ewma_lambda=args.ewma_lambda,
    )

    print(_format_table(results))


if __name__ == "__main__":
    main()
