import asyncio

import aiohttp


async def main() -> None:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
    params = {
        "interval": "30m",
        "range": "60d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
        async with session.get(url, params=params) as resp:
            print("status", resp.status)
            text = await resp.text()
            print("len", len(text))
            print(text[:200])


if __name__ == "__main__":
    asyncio.run(main())
