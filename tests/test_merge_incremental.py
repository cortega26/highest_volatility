import pandas as pd

from highest_volatility.cache.merge import merge_incremental


def test_merge_dedup_and_sort():
    existing = pd.DataFrame(
        {"Adj Close": [1, 2]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"])
    )
    new = pd.DataFrame(
        {"Adj Close": [3, 4]}, index=pd.to_datetime(["2020-01-02", "2020-01-03"])
    )
    merged = merge_incremental(existing, new)
    assert list(merged.index) == list(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    assert merged.loc["2020-01-02", "Adj Close"] == 3
