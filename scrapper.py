import aiohttp
from bs4 import BeautifulSoup


async def fast_scrap(url: str, css_selector: str) -> str:
    """Fast function to scrap a webpage"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            return soup.select_one(css_selector).text

