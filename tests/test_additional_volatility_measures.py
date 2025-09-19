import numpy as np
import pandas as pd
from highest_volatility.compute.metrics import additional_volatility_measures



def _reference_additional_volatility_measures(
    raw: pd.DataFrame,
    tickers: list[str],
    *,
    min_periods: int = 2,
    interval: str = "1d",
    ewma_lambda: float = 0.94,
) -> pd.DataFrame:
    per_year = 252 if interval == "1d" else 365
    ln2 = np.log(2.0)
    results: list[dict[str, float | str]] = []

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
        rec: dict[str, float | str] = {"ticker": ticker}

        s_close = _get_series("Adj Close", ticker)
        if s_close is None:
            s_close = _get_series("Close", ticker)
        s_open = _get_series("Open", ticker)
        s_high = _get_series("High", ticker)
        s_low = _get_series("Low", ticker)

        candidates: dict[str, pd.Series] = {}
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


def _make_multiindex_price_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=12, freq="B")
    tickers = ["AAA", "BBB"]
    arrays = [[], []]
    data: dict[tuple[str, str], np.ndarray] = {}
    for ticker in tickers:
        base = 100 + 10 * rng.random()
        close = base + rng.normal(scale=1.5, size=len(index)).cumsum()
        high = close + rng.uniform(0.5, 1.5, size=len(index))
        low = close - rng.uniform(0.5, 1.5, size=len(index))
        open_ = close + rng.normal(scale=0.8, size=len(index))
        adj_close = close * (1 + rng.normal(scale=0.01, size=len(index)))
        frame = {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": adj_close,
        }
        for field, values in frame.items():
            key = (field, ticker)
            data[key] = values
    columns = pd.MultiIndex.from_tuples(data.keys(), names=["Field", "Ticker"])
    df = pd.DataFrame(data, index=index, columns=columns).sort_index(axis=1)
    df.iloc[2, df.columns.get_loc(("Low", "AAA"))] = np.nan
    df.iloc[5, df.columns.get_loc(("Open", "BBB"))] = np.nan
    return df


def _make_single_ticker_frame(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-02", periods=10, freq="B")
    base = 200 + rng.random()
    close = base + rng.normal(scale=1.0, size=len(index)).cumsum()
    high = close + rng.uniform(0.2, 0.8, size=len(index))
    low = close - rng.uniform(0.2, 0.8, size=len(index))
    open_ = close + rng.normal(scale=0.5, size=len(index))
    adj_close = close * (1 + rng.normal(scale=0.005, size=len(index)))
    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": adj_close,
        },
        index=index,
    )
    df.loc[index[3], "High"] = np.nan
    return df


def _sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("ticker").reset_index(drop=True)


def test_multi_ticker_matches_reference():
    raw = _make_multiindex_price_frame()
    tickers = ["AAA", "BBB"]
    expected = _sort_frame(
        _reference_additional_volatility_measures(
            raw, tickers, min_periods=3, interval="1d", ewma_lambda=0.94
        )
    )
    result = _sort_frame(
        additional_volatility_measures(
            raw, tickers, min_periods=3, interval="1d", ewma_lambda=0.94
        )
    )
    pd.testing.assert_frame_equal(result, expected)


def test_single_ticker_single_level_columns():
    raw = _make_single_ticker_frame()
    ticker = ["ZZZ"]
    expected = _reference_additional_volatility_measures(
        raw, ticker, min_periods=2, interval="1d", ewma_lambda=0.91
    )
    result = additional_volatility_measures(
        raw, ticker, min_periods=2, interval="1d", ewma_lambda=0.91
    )
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
