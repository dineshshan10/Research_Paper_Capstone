"""Dense, BM25, hybrid RRF, reranked, and MMR retrieval strategies."""
from __future__ import annotations

import re
from typing import Literal
from rank_bm25 import BM25Okapi

from config.settings import MMR_LAMBDA, RRF_K, TOP_K_RETRIEVAL
from core.embeddings import cosine_similarity
from core.schema import RetrievedChunk
from core.vector_store import PaperVectorStore

def tokenize(text: str) -> list[str]: return re.findall(r"[a-z0-9]+", text.lower())

class PaperRetriever:
    def __init__(self, store: PaperVectorStore):
        self.store = store
        self.bm25 = BM25Okapi([tokenize(c.content) for c in store.chunks])
    def dense(self, query: str, k: int) -> list[RetrievedChunk]: return self.store.dense_search(query, k)
    def bm25_search(self, query: str, k: int) -> list[RetrievedChunk]:
        scores = self.bm25.get_scores(tokenize(query)); indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [RetrievedChunk(**self.store.chunks[i].model_dump(), score=float(scores[i]), bm25_score=float(scores[i])) for i in indices]
    def hybrid(self, query: str, k: int) -> list[RetrievedChunk]:
        rankings = [self.dense(query, k * 2), self.bm25_search(query, k * 2)]; fused: dict[str, tuple[float, RetrievedChunk]] = {}
        for result_list in rankings:
            for rank, item in enumerate(result_list, 1):
                score, prior = fused.get(item.id, (0.0, item)); fused[item.id] = (score + 1 / (RRF_K + rank), prior)
        return [item.model_copy(update={"score": score}) for score, item in sorted(fused.values(), reverse=True, key=lambda x: x[0])[:k]]
    def rerank(self, query: str, k: int) -> list[RetrievedChunk]:
        # Lexical cross-encoder fallback is deterministic; optional neural reranker can replace this port.
        query_terms = set(tokenize(query)); candidates = self.hybrid(query, k * 3)
        ranked = sorted(candidates, key=lambda c: (len(query_terms & set(tokenize(c.content))) / max(1, len(query_terms)), c.score), reverse=True)[:k]
        return [c.model_copy(update={"score": float(len(ranked) - i) / len(ranked), "rerank_score": float(len(ranked) - i) / len(ranked)}) for i, c in enumerate(ranked)]
    def mmr(self, query: str, k: int) -> list[RetrievedChunk]:
        candidates = self.dense(query, max(k * 4, 12)); selected: list[RetrievedChunk] = []
        while candidates and len(selected) < k:
            def value(item: RetrievedChunk) -> float:
                diversity = max((len(set(tokenize(item.content)) & set(tokenize(s.content))) / max(1, len(set(tokenize(item.content)))) for s in selected), default=0.0)
                return MMR_LAMBDA * item.score - (1 - MMR_LAMBDA) * diversity
            selected.append(max(candidates, key=value)); candidates = [x for x in candidates if x.id != selected[-1].id]
        return selected
    def retrieve(self, query: str, strategy: Literal['dense','bm25','hybrid','hybrid_rerank','mmr'], k: int = TOP_K_RETRIEVAL) -> list[RetrievedChunk]:
        return {'dense': self.dense, 'bm25': self.bm25_search, 'hybrid': self.hybrid, 'hybrid_rerank': self.rerank, 'mmr': self.mmr}[strategy](query, k)
