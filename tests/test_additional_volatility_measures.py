import numpy as np
import pandas as pd
import pytest

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

    def _field_frame(field: str) -> pd.DataFrame | None:
        if isinstance(raw.columns, pd.MultiIndex):
            if field in raw.columns.get_level_values(0):
                return raw[field]
            return None
        if field in raw.columns:
            return raw[[field]]
        return None

    field_frames: dict[str, pd.DataFrame | None] = {
        "adj_close": _field_frame("Adj Close"),
        "close": _field_frame("Close"),
        "open": _field_frame("Open"),
        "high": _field_frame("High"),
        "low": _field_frame("Low"),
    }

    def _array_for(field: str, ticker: str) -> tuple[np.ndarray, np.ndarray] | None:
        frame = field_frames[field]
        if frame is None:
            return None
        if isinstance(frame, pd.Series):
            series = frame
        else:
            if ticker in frame.columns:
                series = frame[ticker]
            elif frame.shape[1] == 1:
                series = frame.iloc[:, 0]
            else:
                return None
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        mask = np.isfinite(values)
        return values, mask

    for ticker in tickers:
        close_data = _array_for("adj_close", ticker) or _array_for("close", ticker)
        if close_data is None:
            continue
        close_values = close_data[0]

        data_store: dict[str, tuple[np.ndarray, np.ndarray]] = {"close": close_data}
        for key in ("open", "high", "low"):
            arr = _array_for(key, ticker)
            if arr is not None:
                data_store[key] = arr

        masks = [mask for _, mask in data_store.values()]
        if not masks:
            continue
        common_mask = masks[0].copy()
        for mask in masks[1:]:
            common_mask &= mask
        valid_count = int(common_mask.sum())
        if valid_count < min_periods:
            continue

        rec: dict[str, float | str] = {"ticker": ticker}

        close = close_values[common_mask]
        if close.size < 2:
            r_cc = np.array([], dtype=np.float64)
        else:
            r_cc = np.diff(np.log(close))

        if "high" in data_store and "low" in data_store:
            high = data_store["high"][0][common_mask]
            low = data_store["low"][0][common_mask]
            hl = np.log(high / low) ** 2
            if hl.size >= min_periods:
                var = hl.mean() / (4.0 * ln2)
                if np.isfinite(var) and var >= 0:
                    rec["parkinson_vol"] = float(np.sqrt(var * per_year))

        if {"open", "high", "low"}.issubset(data_store):
            open_values = data_store["open"][0][common_mask]
            high = data_store["high"][0][common_mask]
            low = data_store["low"][0][common_mask]
            log_hl = np.log(high / low) ** 2
            log_co = np.log(close / open_values) ** 2
            if log_hl.size >= min_periods and log_co.size >= min_periods:
                gk_var = 0.5 * log_hl.mean() - (2.0 * ln2 - 1.0) * log_co.mean()
                if np.isfinite(gk_var) and gk_var >= 0:
                    rec["gk_vol"] = float(np.sqrt(gk_var * per_year))

                term_rs = (
                    np.log(high / close) * np.log(high / open_values)
                    + np.log(low / close) * np.log(low / open_values)
                )
                rs_var = term_rs.mean()
                if np.isfinite(rs_var) and rs_var >= 0:
                    rec["rs_vol"] = float(np.sqrt(rs_var * per_year))

                if close.size >= 2:
                    prev_close = close[:-1]
                    r_o = np.log(open_values[1:] / prev_close)
                    r_c = np.log(close / open_values)
                    if r_o.size >= 1 and r_c.size >= 1:
                        sigma_o2 = r_o.var(ddof=1)
                        sigma_c2 = r_c.var(ddof=1)
                        sigma_rs = term_rs.mean()
                        if (
                            np.isfinite(sigma_o2)
                            and np.isfinite(sigma_c2)
                            and np.isfinite(sigma_rs)
                        ):
                            k = 0.34
                            yz_var = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs
                            if yz_var >= 0:
                                rec["yz_vol"] = float(np.sqrt(yz_var * per_year))

        if r_cc.size >= min_periods:
            if r_cc.size >= 2:
                var = r_cc.var(ddof=1)
            elif r_cc.size == 1:
                var = float(r_cc[0] ** 2)
            else:
                var = float("nan")
            for x in r_cc[-min_periods:]:
                var = ewma_lambda * var + (1.0 - ewma_lambda) * (x * x)
            if np.isfinite(var) and var >= 0:
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


@pytest.fixture
def multiindex_price_frame() -> pd.DataFrame:
    return _make_multiindex_price_frame()


@pytest.fixture
def misordered_multiindex_price_frame(multiindex_price_frame: pd.DataFrame) -> pd.DataFrame:
    swapped = multiindex_price_frame.swaplevel(0, 1, axis=1)
    return swapped.sort_index(axis=1)


@pytest.fixture
def mixed_dtype_price_frame(multiindex_price_frame: pd.DataFrame) -> pd.DataFrame:
    mixed = multiindex_price_frame.copy()
    open_col = ("Open", "AAA")
    mixed[open_col] = mixed[open_col].map(lambda x: f"{x:.6f}")
    high_col = ("High", "BBB")
    mixed[high_col] = mixed[high_col].astype(object)
    low_col = ("Low", "AAA")
    mixed[low_col] = mixed[low_col].astype(np.float32)
    return mixed


@pytest.fixture
def single_ticker_frame() -> pd.DataFrame:
    return _make_single_ticker_frame()


@pytest.mark.parametrize(
    "fixture_name,tickers,min_periods,interval,ewma_lambda",
    [
        ("multiindex_price_frame", ["AAA", "BBB"], 3, "1d", 0.94),
        ("misordered_multiindex_price_frame", ["AAA", "BBB"], 3, "1d", 0.94),
        ("mixed_dtype_price_frame", ["AAA", "BBB"], 3, "1d", 0.94),
        ("single_ticker_frame", ["ZZZ"], 2, "1d", 0.91),
    ],
)
def test_additional_volatility_measures_matches_reference(
    fixture_name: str,
    tickers: list[str],
    min_periods: int,
    interval: str,
    ewma_lambda: float,
    request: pytest.FixtureRequest,
) -> None:
    raw = request.getfixturevalue(fixture_name)
    expected = _sort_frame(
        _reference_additional_volatility_measures(
            raw, tickers, min_periods=min_periods, interval=interval, ewma_lambda=ewma_lambda
        )
    )
    result = _sort_frame(
        additional_volatility_measures(
            raw, tickers, min_periods=min_periods, interval=interval, ewma_lambda=ewma_lambda
        )
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=True, check_dtype=True)
