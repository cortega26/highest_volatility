import pandas as pd

from highest_volatility.app.sanitization import sanitize_close


def test_sanitize_close_drops_duplicates_and_short_series():
    idx = pd.date_range("2023-01-01", periods=5)
    close = pd.DataFrame(
        {
            "KEEP": [1, 2, 3, 4, 5],
            "DUPE": [1, 2, 3, 4, 5],
            "SHORT": [1, None, None, None, None],
        },
        index=idx,
    )

    clean, short, duplicate = sanitize_close(close, min_days=3)

    assert list(clean.columns) == ["KEEP"]
    assert short == ["SHORT"]
    assert duplicate == ["DUPE"]
