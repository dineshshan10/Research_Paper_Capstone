"""Explicit corpus guard for queries that are clearly outside the research-paper scope."""
from __future__ import annotations

EXPLICIT_OUT_OF_CORPUS_TERMS = (
    "deepseek",
    "claude",
    "llama 3",
    "llama3",
    "qwen",
    "grok",
    "venus",
    "mars",
    "weather",
    "stock price",
)


def is_explicitly_out_of_corpus(query: str) -> bool:
    """Return True for subjects the five AI research papers cannot answer."""
    normalized = query.lower()
    return any(term in normalized for term in EXPLICIT_OUT_OF_CORPUS_TERMS)
