import asyncio
import sys
from datetime import date, timedelta

sys.path.insert(0, "src")

from datasource.yahoo_http_async import YahooHTTPAsyncDataSource


async def main() -> None:
    ds = YahooHTTPAsyncDataSource()
    start = date.today() - timedelta(days=60)
    end = date.today()
    try:
        df = await ds.get_prices("AAPL", start, end, "30m")
        print("OK rows:", len(df))
    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
