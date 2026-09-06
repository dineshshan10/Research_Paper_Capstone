from __future__ import annotations

def rewrite_query(query: str, history: list[dict[str, str]]) -> str:
    """A small deterministic history resolver; avoids an LLM dependency for pronouns."""
    if not history or not any(word in query.lower().split() for word in ("it", "its", "that", "they", "them")):
        return query
    last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
    return f"{query} (Follow-up to: {last_user})" if last_user else query
