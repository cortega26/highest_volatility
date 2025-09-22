import pandas as pd
from datetime import datetime

from highest_volatility.ingest import downloaders


def test_build_combined_dataframe_trims_naive_start_with_tz_index():
    tz_index = pd.date_range(
        "2024-01-01 09:30",
        periods=4,
        freq="15min",
        tz="US/Eastern",
    )
    frames = {
        "AAA": pd.DataFrame({"Adj Close": [1.0, 1.1, 1.2, 1.3]}, index=tz_index)
    }

    trim_start = datetime(2024, 1, 1, 9, 45)

    combined = downloaders.build_combined_dataframe(frames, trim_start=trim_start)

    assert not combined.empty
    assert combined.index[0] == pd.Timestamp(trim_start, tz=tz_index.tz)
    # Ensure earlier timestamps were trimmed
    assert pd.Timestamp("2024-01-01 09:30", tz=tz_index.tz) not in combined.index


def test_build_combined_dataframe_trim_beyond_range_returns_empty():
    tz_index = pd.date_range(
        "2024-01-01 09:30",
        periods=2,
        freq="15min",
        tz="UTC",
    )
    frames = {
        "AAA": pd.DataFrame({"Adj Close": [1.0, 1.1]}, index=tz_index)
    }

    trim_start = datetime(2024, 1, 1, 10, 0)

    combined = downloaders.build_combined_dataframe(frames, trim_start=trim_start)

    assert combined.empty
