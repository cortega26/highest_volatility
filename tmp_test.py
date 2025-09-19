import sys, asyncio
from datetime import date, timedelta
sys.path.insert(0, 'src')
from datasource.yahoo_http_async import YahooHTTPAsyncDataSource

async def main():
    ds = YahooHTTPAsyncDataSource()
    start = date.today() - timedelta(days=60)
    end = date.today()
    try:
        df = await ds.get_prices('AAPL', start, end, '30m')
        print('OK rows:', len(df))
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
