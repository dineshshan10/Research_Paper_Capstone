"""Pluggable embedding providers with correct model-specific policies."""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

from config.settings import EMBEDDING_REGISTRY, EmbeddingSpec, OPENAI_API_KEY, USE_NEURAL_EMBEDDINGS


class EmbeddingProvider(ABC):
    def __init__(self, spec: EmbeddingSpec): self.spec = spec
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class HashFallbackEmbedding(EmbeddingProvider):
    """Deterministic fallback for offline demos; never silently used as a benchmark winner."""
    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.spec.dimensions
        terms = re.findall(r"[a-z0-9]+", text.lower())
        for index, term in enumerate(terms):
            for feature, weight in ((term, 1.0), (f"{term}_{terms[index + 1]}" if index + 1 < len(terms) else "", 0.5)):
                if feature:
                    vector[int(hashlib.sha256(feature.encode()).hexdigest(), 16) % len(vector)] += weight
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
    def embed_documents(self, texts: list[str]) -> list[list[float]]: return [self._embed(self.spec.document_prefix + text) for text in texts]
    def embed_query(self, text: str) -> list[float]: return self._embed(self.spec.query_prefix + text)


class SentenceTransformerEmbedding(EmbeddingProvider):
    def __init__(self, spec: EmbeddingSpec):
        super().__init__(spec)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(spec.model_id)
        self.model.max_seq_length = min(self.model.max_seq_length, spec.max_seq_length)
    def _encode(self, texts: list[str]) -> list[list[float]]:
        values = self.model.encode(texts, normalize_embeddings=self.spec.normalize, show_progress_bar=False)
        return values.tolist()
    def embed_documents(self, texts: list[str]) -> list[list[float]]: return self._encode([self.spec.document_prefix + t for t in texts])
    def embed_query(self, text: str) -> list[float]: return self._encode([self.spec.query_prefix + text])[0]


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, spec: EmbeddingSpec):
        super().__init__(spec)
        if not OPENAI_API_KEY: raise RuntimeError("OPENAI_API_KEY is required for openai_small")
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [item.embedding for item in self.client.embeddings.create(model=self.spec.model_id, input=texts).data]
    def embed_documents(self, texts: list[str]) -> list[list[float]]: return self._embed(texts)
    def embed_query(self, text: str) -> list[float]: return self._embed([text])[0]


def get_embedding_provider(key: str, allow_fallback: bool = True) -> EmbeddingProvider:
    spec = EMBEDDING_REGISTRY[key]
    try:
        if spec.kind == "huggingface" and not USE_NEURAL_EMBEDDINGS:
            return HashFallbackEmbedding(spec)
        return OpenAIEmbedding(spec) if spec.kind == "openai" else SentenceTransformerEmbedding(spec)
    except Exception:
        if not allow_fallback: raise
        return HashFallbackEmbedding(spec)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    denom = math.sqrt(sum(v*v for v in left)) * math.sqrt(sum(v*v for v in right))
    return sum(a*b for a, b in zip(left, right)) / denom if denom else 0.0
