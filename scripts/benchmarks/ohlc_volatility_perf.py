"""Benchmark :func:`additional_volatility_measures` on a large OHLC matrix."""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from highest_volatility.compute.metrics import additional_volatility_measures, periods_per_year


@dataclass
class PerfResult:
    label: str
    avg_seconds: float
    peak_rss_mb: float


def _baseline_additional_volatility_measures(
    raw: pd.DataFrame,
    tickers: List[str],
    *,
    min_periods: int = 2,
    interval: str = "1d",
    ewma_lambda: float = 0.94,
) -> pd.DataFrame:
    """Reference implementation mirroring the pre-refactor algorithm."""

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
                series = raw[field][ticker]
            else:
                series = raw[field]
        except Exception:
            return None
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        return numeric if not numeric.empty else None

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


def _generate_prices(n_rows: int, tickers: List[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-02", periods=n_rows, freq="T")
    data: Dict[tuple[str, str], np.ndarray] = {}
    for ticker in tickers:
        base = 100 + 20 * rng.random()
        close = base + rng.normal(scale=0.8, size=n_rows).cumsum()
        high = close + rng.uniform(0.05, 0.5, size=n_rows)
        low = close - rng.uniform(0.05, 0.5, size=n_rows)
        open_ = close + rng.normal(scale=0.4, size=n_rows)
        adj_close = close * (1 + rng.normal(scale=0.0005, size=n_rows))
        frame = {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": adj_close,
        }
        for field, values in frame.items():
            data[(field, ticker)] = values.astype(np.float32)
    columns = pd.MultiIndex.from_tuples(data.keys(), names=["Field", "Ticker"])
    return pd.DataFrame(data, index=index, columns=columns).sort_index(axis=1)


def _rss_to_mb(value: float) -> float:
    if sys.platform.startswith("darwin"):
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _measure(
    label: str,
    func_name: str,
    *,
    n_rows: int,
    n_tickers: int,
    repeats: int,
    min_periods: int,
    interval: str,
    ewma_lambda: float,
    seed: int,
) -> PerfResult:
    queue: multiprocessing.Queue[tuple[float, float]] = multiprocessing.Queue()

    def _runner(q: multiprocessing.Queue[tuple[float, float]]) -> None:
        try:
            from resource import RUSAGE_SELF, getrusage
        except ImportError as exc:  # pragma: no cover - Windows fallback
            raise RuntimeError("resource module is required for RSS measurement") from exc

        tickers = [f"T{i:03d}" for i in range(n_tickers)]
        prices = _generate_prices(n_rows, tickers, seed)
        kwargs = dict(min_periods=min_periods, interval=interval, ewma_lambda=ewma_lambda)

        target = (
            _baseline_additional_volatility_measures
            if func_name == "baseline"
            else additional_volatility_measures
        )

        start = time.perf_counter()
        for _ in range(repeats):
            target(prices, tickers, **kwargs)
        elapsed = (time.perf_counter() - start) / max(repeats, 1)
        rss_kb = float(getrusage(RUSAGE_SELF).ru_maxrss)
        q.put((elapsed, rss_kb))

    process = multiprocessing.Process(target=_runner, args=(queue,), daemon=False)
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{label} benchmark failed with exit code {process.exitcode}")
    elapsed, rss_kb = queue.get()
    return PerfResult(label, elapsed, _rss_to_mb(rss_kb))


def run_benchmark(
    *,
    n_rows: int,
    n_tickers: int,
    repeats: int,
    min_periods: int,
    interval: str,
    ewma_lambda: float,
    seed: int,
) -> list[PerfResult]:
    results = [
        _measure(
            "baseline",
            "baseline",
            n_rows=n_rows,
            n_tickers=n_tickers,
            repeats=repeats,
            min_periods=min_periods,
            interval=interval,
            ewma_lambda=ewma_lambda,
            seed=seed,
        ),
        _measure(
            "optimized",
            "optimized",
            n_rows=n_rows,
            n_tickers=n_tickers,
            repeats=repeats,
            min_periods=min_periods,
            interval=interval,
            ewma_lambda=ewma_lambda,
            seed=seed,
        ),
    ]
    return results


def _format_results(results: list[PerfResult]) -> str:
    header = "label\tavg_seconds\tpeak_rss_mb"
    rows = [f"{r.label}\t{r.avg_seconds:.6f}\t{r.peak_rss_mb:.2f}" for r in results]
    return "\n".join([header, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000, help="Number of minute bars")
    parser.add_argument("--tickers", type=int, default=500, help="Number of tickers")
    parser.add_argument("--repeats", type=int, default=1, help="Benchmark repetitions")
    parser.add_argument("--min-periods", type=int, default=2, dest="min_periods")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--ewma-lambda", type=float, default=0.94, dest="ewma_lambda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_benchmark(
        n_rows=args.rows,
        n_tickers=args.tickers,
        repeats=args.repeats,
        min_periods=args.min_periods,
        interval=args.interval,
        ewma_lambda=args.ewma_lambda,
        seed=args.seed,
    )
    print(_format_results(results))


if __name__ == "__main__":
    main()
