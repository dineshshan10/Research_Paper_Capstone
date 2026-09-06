from __future__ import annotations
from config.settings import TAVILY_API_KEY, WEB_SEARCH_RESULTS

def web_search(query: str, api_key: str | None = None) -> list[dict[str, str]]:
    """Search Tavily using a per-request key or the server environment default."""
    key = api_key or TAVILY_API_KEY
    if not key: return []
    try:
        from tavily import TavilyClient
        return TavilyClient(api_key=key).search(query=query, max_results=WEB_SEARCH_RESULTS).get("results", [])
    except Exception: return []
