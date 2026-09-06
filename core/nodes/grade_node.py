from __future__ import annotations
import re
from config.settings import RELEVANCE_THRESHOLD

OUT_OF_CORPUS = ("deepseek", "claude", "llama 3", "llama3", "qwen", "grok")
STOP_WORDS = {"a", "an", "and", "are", "about", "does", "do", "for", "how", "in", "is", "its", "of", "the", "to", "what", "with", "why"}

def grade_context(query: str, documents) -> bool:
    if any(term in query.lower() for term in OUT_OF_CORPUS): return False
    if not documents: return False
    terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if term not in STOP_WORDS}
    context = set(re.findall(r"[a-z0-9]+", " ".join(d.content for d in documents).lower()))
    # Retrieval has already ranked this context. One meaningful domain-term match is
    # enough to generate a cited response; only clear negatives should take the web path.
    return bool(terms & context)
