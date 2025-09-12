import json
import time
from pathlib import Path

import pandas as pd

from highest_volatility.ingest import tickers


BUILD_ID = "build123"


def _build_html_page(companies, total=None):
    """Return a minimal HTML document containing the ``__NEXT_DATA__`` blob."""
    total = total if total is not None else len(companies)
    data = {
        "buildId": BUILD_ID,
        "props": {"pageProps": {"initialResults": companies, "stats": {"ffc": total}}},
    }
    return f"<html><script id='__NEXT_DATA__'>{json.dumps(data)}</script></html>"


def _build_json_page(companies):
    """Return the JSON payload served by the ``_next/data`` endpoint."""
    return {"pageProps": {"initialResults": companies}}


class _HTMLResp:
    def __init__(self, text):
        self.text = text
        self.status_code = 200

    def raise_for_status(self):  # pragma: no cover - simple stub
        pass


class _JSONResp:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def json(self):  # pragma: no cover - simple stub
        return self._data

    def raise_for_status(self):  # pragma: no cover - simple stub
        pass


class DummySession:
    """Simple session stub that serves pre-defined HTML/JSON pages."""

    def __init__(self, html_page, json_pages):
        self.html_page = html_page
        self.json_pages = json_pages
        self.calls = []

    def get(self, url, timeout=30, headers=None):
        self.calls.append(url)
        if "fortune-500-companies.json" in url:
            page = int(url.split("page=")[1])
            data = self.json_pages.get(page)
            if data is None:
                raise AssertionError("unexpected page")
            return _JSONResp(data)
        if self.html_page is None:
            raise AssertionError("unexpected page")
        return _HTMLResp(self.html_page)


def test_cached_list_used_when_available(tmp_path):
    html_page = _build_html_page(
        [
            {"rank": 1, "company": "A", "ticker": "AAA"},
            {"rank": 2, "company": "B", "ticker": "BBB"},
        ],
        total=3,
    )
    json_pages = {2: _build_json_page([{ "rank": 3, "company": "C", "ticker": "CCC" }])}
    session1 = DummySession(html_page, json_pages)

    df1 = tickers.fetch_fortune_tickers(top_n=3, cache_dir=tmp_path, session=session1)
    assert len(df1) == 3
    assert len(session1.calls) == 2
    assert session1.calls[1].endswith(
        f"/_next/data/{BUILD_ID}/fortune-500-companies.json?page=2"
    )

    session2 = DummySession(None, {})
    df2 = tickers.fetch_fortune_tickers(top_n=3, cache_dir=tmp_path, session=session2)
    assert session2.calls == []  # served from cache
    pd.testing.assert_frame_equal(df1, df2)


def test_cached_list_refreshed_after_expiry(tmp_path):
    html_page = _build_html_page([{"rank": 1, "company": "A", "ticker": "AAA"}])
    session = DummySession(html_page, {})
    df1 = tickers.fetch_fortune_tickers(top_n=1, cache_dir=tmp_path, session=session)
    assert df1.iloc[0]["company"] == "A"

    cache_file = Path(tmp_path) / tickers.CACHE_FILE
    payload = json.loads(cache_file.read_text())
    payload["timestamp"] = time.time() - tickers.CACHE_EXPIRY - 1
    cache_file.write_text(json.dumps(payload))

    updated_html = _build_html_page([{"rank": 1, "company": "X", "ticker": "XXX"}])
    session_new = DummySession(updated_html, {})
    df2 = tickers.fetch_fortune_tickers(top_n=1, cache_dir=tmp_path, session=session_new)
    assert session_new.calls  # network was used
    assert df2.iloc[0]["company"] == "X"

