from __future__ import annotations
from core.retrievers import PaperRetriever

def retrieve_context(retriever: PaperRetriever, query: str, strategy: str, k: int):
    return retriever.retrieve(query, strategy, k)
