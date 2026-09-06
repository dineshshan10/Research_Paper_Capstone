"""Shared, serialisable models for the paper-answering system."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class PaperChunk(BaseModel):
    id: str
    content: str
    paper_id: str
    title: str
    page: int = Field(ge=1)
    section: str = ""
    chunk_index: int = Field(ge=0)


class RetrievedChunk(PaperChunk):
    score: float = 0.0
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


class Source(BaseModel):
    number: int
    chunk_id: str
    paper_id: str
    title: str
    page: int
    section: str = ""
    excerpt: str
    score: float
    url: str | None = None


class Answer(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    provenance: Literal["corpus", "web", "unavailable"] = "corpus"
    query: str
    rewritten_query: str | None = None
    retrieval_strategy: str
    latency_ms: float = 0.0
    rewrite_count: int = 0
    inspector: list[RetrievedChunk] = Field(default_factory=list)
    message: str | None = None

