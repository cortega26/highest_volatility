from highest_volatility import universe
from highest_volatility.ingest import tickers
from tests.test_tickers_cache import (
    BUILD_ID,
    DummySession,
    _build_html_page,
    _build_json_page,
)


def test_universe_can_fetch_over_200(monkeypatch, tmp_path):
    # Prepare five pages with 50 unique tickers each (total 250)
    first_page = [
        {"rank": i + 1, "company": f"C{i}", "ticker": f"T{i}"} for i in range(50)
    ]
    html_page = _build_html_page(first_page, total=250)
    json_pages = {}
    for page in range(2, 6):
        start = (page - 1) * 50
        comps = [
            {"rank": start + j + 1, "company": f"C{start + j}", "ticker": f"T{start + j}"}
            for j in range(50)
        ]
        json_pages[page] = _build_json_page(comps)
    session = DummySession(html_page, json_pages)

    def fake_fetch(top_n):
        return tickers.fetch_fortune_tickers(
            top_n=top_n, cache_dir=tmp_path, session=session
        )

    monkeypatch.setattr(universe, "fetch_fortune_tickers", fake_fetch)
    tickers_list, _fortune = universe.build_universe(300)
    assert len(tickers_list) >= 200
