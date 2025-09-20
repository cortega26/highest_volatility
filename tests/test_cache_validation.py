import pandas as pd
import pytest

from src.cache import store
from highest_volatility.pipeline import validate_cache
from src.highest_volatility.pipeline import validation


def _make_df():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    return pd.DataFrame({"Adj Close": [1.0, 2.0, 3.0]}, index=idx)


def _make_manifest(df):
    return store.Manifest(
        ticker="ABC",
        interval="1d",
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        rows=len(df),
        source="test",
        version=1,
        updated_at="2020-01-01T00:00:00Z",
    )


def test_validate_cache_ok():
    df = _make_df()
    manifest = _make_manifest(df)
    validate_cache(df, manifest)  # should not raise


def test_validate_cache_weekend_gap_allowed():
    idx = pd.to_datetime(["2020-01-03", "2020-01-06"])
    df = pd.DataFrame({"Adj Close": [1.0, 2.0]}, index=idx)
    manifest = store.Manifest(
        ticker="ABC",
        interval="1d",
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        rows=len(df),
        source="test",
        version=1,
        updated_at="2020-01-01T00:00:00Z",
    )
    validate_cache(df, manifest)


def test_validate_cache_market_holiday_gap_allowed():
    pytest.importorskip("pandas_market_calendars")

    idx = pd.to_datetime(["2024-07-03", "2024-07-05"])
    df = pd.DataFrame({"Adj Close": [1.0, 2.0]}, index=idx)
    manifest = store.Manifest(
        ticker="ABC",
        interval="1d",
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        rows=len(df),
        source="test",
        version=1,
        updated_at="2020-01-01T00:00:00Z",
    )
    validate_cache(df, manifest)


def test_validate_cache_holiday_gap_allowed_without_mcal(monkeypatch):
    monkeypatch.setattr(validation, "mcal", None)
    validation._get_trading_calendar.cache_clear()

    idx = pd.to_datetime(["2024-07-03", "2024-07-05"])
    df = pd.DataFrame({"Adj Close": [1.0, 2.0]}, index=idx)
    manifest = store.Manifest(
        ticker="ABC",
        interval="1d",
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        rows=len(df),
        source="test",
        version=1,
        updated_at="2020-01-01T00:00:00Z",
    )

    try:
        validate_cache(df, manifest)
    finally:
        validation._get_trading_calendar.cache_clear()


@pytest.mark.parametrize(
    "start_date,end_date",
    [
        ("1985-09-26", "1985-09-30"),
        ("1994-04-26", "1994-04-28"),
        ("2001-09-10", "2001-09-17"),
        ("2004-06-10", "2004-06-14"),
        ("2006-12-29", "2007-01-03"),
        ("2012-10-26", "2012-10-31"),
        ("2018-12-04", "2018-12-06"),
    ],
)
def test_validate_cache_special_closures_without_mcal(monkeypatch, start_date, end_date):
    monkeypatch.setattr(validation, "mcal", None)
    validation._get_trading_calendar.cache_clear()

    idx = pd.to_datetime([start_date, end_date])
    df = pd.DataFrame({"Adj Close": [1.0, 2.0]}, index=idx)
    manifest = store.Manifest(
        ticker="ABC",
        interval="1d",
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        rows=len(df),
        source="test",
        version=1,
        updated_at="2020-01-01T00:00:00Z",
    )

    try:
        validate_cache(df, manifest)
    finally:
        validation._get_trading_calendar.cache_clear()


def test_validate_cache_nan():
    df = _make_df()
    df.iloc[1, 0] = None
    manifest = _make_manifest(df)
    with pytest.raises(ValueError):
        validate_cache(df, manifest)


def test_validate_cache_gap():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"])
    df = pd.DataFrame({"Adj Close": [1.0, 2.0, 4.0, 5.0]}, index=idx)
    manifest = store.Manifest(
        ticker="ABC",
        interval="1d",
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        rows=len(df),
        source="test",
        version=1,
        updated_at="2020-01-01T00:00:00Z",
    )
    with pytest.raises(ValueError):
        validate_cache(df, manifest)


def test_validate_cache_gap_detected_without_mcal(monkeypatch):
    monkeypatch.setattr(validation, "mcal", None)
    validation._get_trading_calendar.cache_clear()

    idx = pd.to_datetime(["2024-07-03", "2024-07-08"])
    df = pd.DataFrame({"Adj Close": [1.0, 2.0]}, index=idx)
    manifest = store.Manifest(
        ticker="ABC",
        interval="1d",
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        rows=len(df),
        source="test",
        version=1,
        updated_at="2020-01-01T00:00:00Z",
    )

    try:
        with pytest.raises(ValueError):
            validate_cache(df, manifest)
    finally:
        validation._get_trading_calendar.cache_clear()


def test_validate_cache_range_mismatch():
    df = _make_df()
    manifest = store.Manifest(
        ticker="ABC",
        interval="1d",
        start="2020-01-01",
        end="2020-01-02",
        rows=len(df),
        source="test",
        version=1,
        updated_at="2020-01-01T00:00:00Z",
    )
    with pytest.raises(ValueError):
        validate_cache(df, manifest)


def test_save_cache_uses_validator(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CACHE_ROOT", tmp_path)
    df = _make_df()
    df.iloc[1, 0] = None
    with pytest.raises(ValueError):
        store.save_cache("ABC", "1d", df, "test")
    assert not (tmp_path / "1d" / "ABC.parquet").exists()
