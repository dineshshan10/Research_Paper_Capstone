"""Persistent Chroma collections; one collection for each embedding model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb

from config.settings import CHROMA_DB_DIR, EMBEDDING_REGISTRY
from core.embeddings import EmbeddingProvider
from core.schema import PaperChunk, RetrievedChunk


def get_client() -> chromadb.PersistentClient:
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


class PaperVectorStore:
    def __init__(self, provider: EmbeddingProvider, chunks: list[PaperChunk] | None = None):
        self.provider, self.chunks = provider, chunks or []
        self._vectors: list[list[float]] | None = None
    def index(self, chunks: list[PaperChunk] | None = None) -> int:
        self.chunks = chunks or self.chunks
        self._vectors = self.provider.embed_documents([c.content for c in self.chunks])
        return len(self.chunks)
    def dense_search(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._vectors is None: self.index()
        from core.embeddings import cosine_similarity
        q = self.provider.embed_query(query)
        ranked = sorted(((cosine_similarity(q, v), c) for c, v in zip(self.chunks, self._vectors or [])), reverse=True, key=lambda x: x[0])[:k]
        return [RetrievedChunk(**chunk.model_dump(), score=score, dense_score=score) for score, chunk in ranked]
    def persist_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([c.model_dump() for c in self.chunks], indent=2), encoding="utf-8")

def index_to_chroma(chunks: list[PaperChunk], provider: EmbeddingProvider, force: bool = False) -> int:
    """Build a named Chroma collection with precomputed vectors and traceable metadata."""
    client = get_client(); name = provider.spec.collection_name
    if force:
        try: client.delete_collection(name)
        except Exception: pass
    collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine", "model": provider.spec.model_id})
    vectors = provider.embed_documents([c.content for c in chunks])
    collection.upsert(ids=[c.id for c in chunks], documents=[c.content for c in chunks], embeddings=vectors,
        metadatas=[{"paper_id": c.paper_id, "title": c.title, "page": c.page, "section": c.section, "chunk_index": c.chunk_index} for c in chunks])
    return collection.count()
