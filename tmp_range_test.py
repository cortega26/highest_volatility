import asyncio, aiohttp

async def main():
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/AAPL'
    params = {'interval':'30m', 'range':'60d', 'events':'div,splits', 'includeAdjustedClose':'true'}
    async with aiohttp.ClientSession(headers={'User-Agent':'Mozilla/5.0'}) as s:
        async with s.get(url, params=params) as resp:
            print('status', resp.status)
            txt = await resp.text()
            print('len', len(txt))
            print(txt[:200])

asyncio.run(main())
