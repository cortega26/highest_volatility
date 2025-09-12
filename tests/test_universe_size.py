from highest_volatility import universe
import pandas as pd


def test_universe_can_fetch_over_200(monkeypatch, tmp_path):
    # Provide a fake fetcher that returns 250 unique tickers
    def fake_fetch(top_n):
        data = [
            {"rank": i + 1, "company": f"C{i}", "ticker": f"T{i}"} for i in range(250)
        ]
        df = pd.DataFrame(data)
        return df.head(top_n)

    monkeypatch.setattr(universe, "fetch_fortune_tickers", fake_fetch)
    tickers_list, _fortune = universe.build_universe(300)
    assert len(tickers_list) >= 200
