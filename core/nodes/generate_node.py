"""Grounded answer generation, with a faithful extractive offline path."""
from __future__ import annotations
import re
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, MAX_OUTPUT_TOKENS
from core.prompts import GROUNDING_PROMPT
from core.schema import RetrievedChunk

def context_text(documents: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] {d.title}, p. {d.page}: {d.content}" for i, d in enumerate(documents, 1))

def _extractive(documents: list[RetrievedChunk]) -> str:
    if not documents: return "I could not find enough relevant material in the indexed research papers."
    sentences: list[str] = []
    for number, doc in enumerate(documents, 1):
        candidate = next((s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.content) if len(s.split()) >= 10), doc.content[:420])
        sentences.append(f"{candidate} [{number}]")
    return "\n\n".join(sentences)

def generate_answer(query: str, documents: list[RetrievedChunk]) -> str:
    if not GEMINI_API_KEY: return _extractive(documents)
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=GROUNDING_PROMPT.format(query=query, context=context_text(documents)), config={"max_output_tokens": MAX_OUTPUT_TOKENS, "temperature": 0.1})
        return response.text or _extractive(documents)
    except Exception:
        return _extractive(documents)
