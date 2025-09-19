#!/usr/bin/env python3
"""Independent sanity-check script to (re)compute KPIs for a set of tickers.

This bypasses the main pipeline and caches to provide an external cross-check
when results look suspicious. It downloads prices directly from yfinance and
computes volatility measures, then prints the top rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


TRADING_DAYS_PER_YEAR = 252


def _annualized_cc_vol(close: pd.DataFrame, min_days: int) -> pd.DataFrame:
    returns = np.log(close / close.shift(1)).dropna(how="all")
    out: List[Tuple[str, float]] = []
    for t in returns.columns:
        s = returns[t].dropna()
        if s.shape[0] < min_days:
            continue
        vol = float(s.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        out.append((t, vol))
    return pd.DataFrame(out, columns=["ticker", "cc_vol"]).sort_values("cc_vol", ascending=False)


def _parkinson_gk_rs_yz(raw: pd.DataFrame, tickers: Iterable[str], min_days: int) -> pd.DataFrame:
    frames: List[dict] = []
    for t in tickers:
        try:
            df = pd.DataFrame({
                "open": raw["Open"][t],
                "high": raw["High"][t],
                "low": raw["Low"][t],
                "close": raw["Close"][t],
            }).dropna()
        except Exception:
            continue
        if df.shape[0] < min_days:
            continue
        rec: dict = {"ticker": t}
        ln2 = np.log(2.0)
        # Parkinson
        hl = np.log(df["high"] / df["low"]) ** 2
        var_p = (hl.sum()) / (4.0 * ln2 * df.shape[0])
        if np.isfinite(var_p) and var_p >= 0:
            rec["parkinson_vol"] = float(np.sqrt(var_p) * np.sqrt(TRADING_DAYS_PER_YEAR))
        # Garman-Klass / Rogers-Satchell / Yang-Zhang
        log_hl = np.log(df["high"] / df["low"]) ** 2
        log_co = np.log(df["close"] / df["open"]) ** 2
        gk_var = 0.5 * log_hl.mean() - (2.0 * ln2 - 1.0) * log_co.mean()
        if np.isfinite(gk_var) and gk_var >= 0:
            rec["gk_vol"] = float(np.sqrt(gk_var) * np.sqrt(TRADING_DAYS_PER_YEAR))
        term_rs = (
            np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) +
            np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"])
        )
        rs_var = term_rs.mean()
        if np.isfinite(rs_var) and rs_var >= 0:
            rec["rs_vol"] = float(np.sqrt(rs_var) * np.sqrt(TRADING_DAYS_PER_YEAR))
        prev_close = df["close"].shift(1)
        r_o = np.log(df["open"] / prev_close).dropna()
        r_c = np.log(df["close"] / df["open"]).dropna()
        df_rs = df.loc[r_c.index]
        term_rs_d = (
            np.log(df_rs["high"] / df_rs["close"]) * np.log(df_rs["high"] / df_rs["open"]) +
            np.log(df_rs["low"] / df_rs["close"]) * np.log(df_rs["low"] / df_rs["open"])
        )
        sigma_o2 = float(r_o.var(ddof=1)) if r_o.shape[0] >= 2 else np.nan
        sigma_c2 = float(r_c.var(ddof=1)) if r_c.shape[0] >= 2 else np.nan
        sigma_rs = float(term_rs_d.mean()) if term_rs_d.shape[0] >= 1 else np.nan
        if np.isfinite(sigma_o2) and np.isfinite(sigma_c2) and np.isfinite(sigma_rs):
            k = 0.34
            yz_var = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs
            if yz_var >= 0:
                rec["yz_vol"] = float(np.sqrt(yz_var) * np.sqrt(TRADING_DAYS_PER_YEAR))
        frames.append(rec)
    return pd.DataFrame(frames)


def _load_fortune_top(n: int) -> List[str]:
    # Try cache path
    cache_candidates = [
        Path(".cache/tickers/fortune_500.csv"),
        Path("src/.cache/tickers/fortune_500.csv"),
        Path("fortune500_tickers.csv"),
    ]
    df = None
    for p in cache_candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                break
            except Exception:
                continue
    if df is None or df.empty:
        raise SystemExit("Could not find cached Fortune tickers CSV")
    cols = {c.lower(): c for c in df.columns}
    for needed in ("rank", "company", "ticker"):
        if needed not in cols:
            raise SystemExit(f"Ticker CSV missing column: {needed}")
    out = (
        df.rename(columns=str.lower)
        .sort_values("rank")
        .head(n)
        .assign(ticker=lambda x: x["ticker"].astype(str).str.strip().str.upper())
    )
    # Normalize dots to dashes for Yahoo
    out["ticker"] = out["ticker"].str.replace(".", "-", regex=False)
    return out["ticker"].tolist()


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate KPIs independently via yfinance")
    ap.add_argument("--tickers", help="Comma-separated tickers to validate")
    ap.add_argument("--fortune-top", type=int, help="Use top-N Fortune tickers")
    ap.add_argument("--lookback-days", type=int, default=252)
    ap.add_argument("--print-top", type=int, default=10)
    args = ap.parse_args(argv)

    if not args.tickers and not args.fortune_top:
        ap.error("Provide --tickers or --fortune-top")

    if args.tickers:
        tickers = [t.strip().upper().replace(".", "-") for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = _load_fortune_top(args.fortune_top)

    # Download prices directly (no cache), auto-adjusted so Close is adjusted
    end = datetime.utcnow()
    start = end - timedelta(days=args.lookback_days * 2)
    raw = yf.download(
        " ".join(tickers),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise SystemExit("Empty dataset from Yahoo Finance")

    # Build wide close matrix
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.dropna(how="all")

    # Compute core KPIs
    base = _annualized_cc_vol(close, min_days=60)
    extra = pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex) and set(["Open", "High", "Low", "Close"]).issubset(set(raw.columns.get_level_values(0))):
        extra = _parkinson_gk_rs_yz(raw, base["ticker"].tolist(), min_days=60)
    result = base.merge(extra, on="ticker", how="left")

    # Print top
    topk = result.head(args.print_top)
    if not topk.empty:
        print(topk.to_string(index=False))
    else:
        print("No KPIs computed — likely insufficient data for the selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

