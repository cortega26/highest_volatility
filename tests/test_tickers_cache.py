import json
import time
from pathlib import Path

import pandas as pd

from highest_volatility.ingest import tickers


def _build_page(companies, total=None):
    total = total if total is not None else len(companies)
    data = {
        "props": {"pageProps": {"initialResults": companies, "stats": {"ffc": total}}}
    }
    return f"<html><script id='__NEXT_DATA__'>{json.dumps(data)}</script></html>"


class _Resp:
    def __init__(self, text):
        self.text = text
        self.status_code = 200

    def raise_for_status(self):  # pragma: no cover - simple stub
        pass


class DummySession:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get(self, url, timeout=30, headers=None):
        self.calls.append(url)
        page = 1
        if "page=" in url:
            page = int(url.split("page=")[1])
        html = self.pages.get(page)
        if html is None:
            raise AssertionError("unexpected page")
        return _Resp(html)


def test_cached_list_used_when_available(tmp_path):
    pages = {
        1: _build_page(
            [
                {"rank": 1, "company": "A", "ticker": "AAA"},
                {"rank": 2, "company": "B", "ticker": "BBB"},
            ],
            total=3,
        ),
        2: _build_page([{ "rank": 3, "company": "C", "ticker": "CCC" }], total=3),
    }
    session1 = DummySession(pages)

    df1 = tickers.fetch_fortune_tickers(top_n=3, cache_dir=tmp_path, session=session1)
    assert len(df1) == 3
    assert len(session1.calls) == 2

    session2 = DummySession({})
    df2 = tickers.fetch_fortune_tickers(top_n=3, cache_dir=tmp_path, session=session2)
    assert session2.calls == []  # served from cache
    pd.testing.assert_frame_equal(df1, df2)


def test_cached_list_refreshed_after_expiry(tmp_path):
    initial_pages = {1: _build_page([{"rank": 1, "company": "A", "ticker": "AAA"}])}
    session = DummySession(initial_pages)
    df1 = tickers.fetch_fortune_tickers(top_n=1, cache_dir=tmp_path, session=session)
    assert df1.iloc[0]["company"] == "A"

    cache_file = Path(tmp_path) / tickers.CACHE_FILE
    payload = json.loads(cache_file.read_text())
    payload["timestamp"] = time.time() - tickers.CACHE_EXPIRY - 1
    cache_file.write_text(json.dumps(payload))

    updated_pages = {1: _build_page([{"rank": 1, "company": "X", "ticker": "XXX"}])}
    session_new = DummySession(updated_pages)
    df2 = tickers.fetch_fortune_tickers(top_n=1, cache_dir=tmp_path, session=session_new)
    assert session_new.calls  # network was used
    assert df2.iloc[0]["company"] == "X"
