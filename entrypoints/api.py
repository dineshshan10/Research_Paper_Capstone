"""FastAPI adapter for the research-paper answer bot."""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field
from core.graph import get_agent

app = FastAPI(title="Research Paper Answer Bot", version="1.0.0")

class AskRequest(BaseModel):
    query: str = Field(min_length=3)
    thread_id: str = "api"
    strategy: str = "hybrid_rerank"

@app.get("/health")
def health(): return {"status": "ok", "service": "research-paper-answer-bot"}

@app.post("/api/v1/ask")
def ask(request: AskRequest): return get_agent().ask(request.query, request.thread_id, request.strategy).model_dump()

@app.get("/api/v1/sources")
def sources():
    agent = get_agent()
    return {"papers": sorted({c.paper_id: c.title for c in agent.chunks}.items()), "chunks": len(agent.chunks)}
