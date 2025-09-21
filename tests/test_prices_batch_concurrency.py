import time
from concurrent.futures import Future
from typing import List
from unittest.mock import patch

import pandas as pd

from highest_volatility.ingest import downloaders, prices


def _make_chunk_frame(tickers: List[str]) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp.now(tz=None), periods=1, freq="D")
    if len(tickers) == 1:
        return pd.DataFrame({"Adj Close": [1.0]}, index=idx)
    columns = pd.MultiIndex.from_product([["Adj Close"], tickers])
    data = [list(range(1, len(tickers) + 1))]
    return pd.DataFrame(data, index=idx, columns=columns)


def test_batch_download_submits_multiple_chunks():
    tickers = [f"T{i}" for i in range(80)]
    submissions: List[List[str]] = []
    worker_counts: List[int] = []

    def fake_download(tickers_arg, *args, **kwargs):
        symbols = tickers_arg.split(" ") if isinstance(tickers_arg, str) else list(tickers_arg)
        return _make_chunk_frame(symbols)

    class ImmediateExecutor:
        def __init__(self, max_workers: int):
            worker_counts.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            submissions.append(list(args[1]))
            future: Future = Future()
            future.set_result(fn(*args, **kwargs))
            return future

    with patch(
        "highest_volatility.ingest.downloaders.ThreadPoolExecutor",
        side_effect=lambda max_workers: ImmediateExecutor(max_workers),
    ):
        with patch("highest_volatility.ingest.prices.yf.download", side_effect=fake_download):
            df = prices.download_price_history(
                tickers,
                lookback_days=5,
                matrix_mode="batch",
                use_cache=False,
                max_workers=4,
            )

    assert not df.empty
    assert submissions == [tickers[:40], tickers[40:]]
    # The executor is sized to the batch of pending chunks (two workers for the first batch).
    assert worker_counts == [2]


def test_concurrent_batches_reduce_latency():
    tickers = [f"S{i}" for i in range(80)]
    sleep_seconds = 0.2

    def sleepy_download(tickers_arg, *args, **kwargs):
        time.sleep(sleep_seconds)
        symbols = tickers_arg.split(" ") if isinstance(tickers_arg, str) else list(tickers_arg)
        return _make_chunk_frame(symbols)

    with patch("highest_volatility.ingest.prices.yf.download", side_effect=sleepy_download):
        start_serial = time.perf_counter()
        prices.download_price_history(
            tickers,
            lookback_days=5,
            matrix_mode="batch",
            use_cache=False,
            max_workers=1,
        )
        serial_duration = time.perf_counter() - start_serial

    with patch("highest_volatility.ingest.prices.yf.download", side_effect=sleepy_download):
        start_concurrent = time.perf_counter()
        prices.download_price_history(
            tickers,
            lookback_days=5,
            matrix_mode="batch",
            use_cache=False,
            max_workers=4,
        )
        concurrent_duration = time.perf_counter() - start_concurrent

    assert concurrent_duration < serial_duration * 0.75


def test_batch_download_result_dataframe_trims():
    frames = {"AAA": pd.DataFrame({"Adj Close": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-02", "2020-01-04"]))}
    result = downloaders.BatchDownloadResult(frames=frames, failed=[])

    combined = result.to_dataframe(trim_start=pd.Timestamp("2020-01-03"))
    assert list(combined.index) == [pd.Timestamp("2020-01-04")]
    assert ("Adj Close", "AAA") in combined.columns
