from __future__ import annotations
from config.settings import TAVILY_API_KEY, WEB_SEARCH_RESULTS

def web_search(query: str) -> list[dict[str, str]]:
    if not TAVILY_API_KEY: return []
    try:
        from tavily import TavilyClient
        return TavilyClient(api_key=TAVILY_API_KEY).search(query=query, max_results=WEB_SEARCH_RESULTS).get("results", [])
    except Exception: return []
