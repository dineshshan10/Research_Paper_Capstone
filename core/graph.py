"""Public RAG facade. LangGraph is optional at runtime; behaviour is deterministic without keys."""
from __future__ import annotations
from time import perf_counter

from config.settings import DEFAULT_EMBEDDING_KEY, DEFAULT_RETRIEVAL_STRATEGY, MAX_QUERY_REWRITES, TOP_K_CONTEXT
from core.embeddings import get_embedding_provider
from core.ingestion import ingest_corpus
from core.nodes import generate_answer, grade_context, retrieve_context, rewrite_query, web_search
from core.retrievers import PaperRetriever
from core.schema import Answer, Source
from core.vector_store import PaperVectorStore

class ResearchPaperAgent:
    def __init__(self, chunks=None, embedding_key: str = DEFAULT_EMBEDDING_KEY, embedding_provider=None):
        self.chunks = chunks or ingest_corpus()
        self.store = PaperVectorStore(embedding_provider or get_embedding_provider(embedding_key), self.chunks)
        self.retriever = PaperRetriever(self.store)
        self.history: dict[str, list[dict[str, str]]] = {}
    def ask(self, query: str, thread_id: str = "default", strategy: str = DEFAULT_RETRIEVAL_STRATEGY, tavily_api_key: str | None = None) -> Answer:
        started = perf_counter(); history = self.history.setdefault(thread_id, [])
        rewritten = rewrite_query(query, history); docs = retrieve_context(self.retriever, rewritten, strategy, TOP_K_CONTEXT)
        provenance = "corpus"; message = None
        if not grade_context(rewritten, docs):
            web = web_search(query, tavily_api_key); provenance = "web" if web else "unavailable"
            if web:
                answer_text = "\n\n".join(f"{item.get('content', '')} [{i}]" for i, item in enumerate(web, 1))
                sources = [Source(number=i, chunk_id=f"web-{i}", paper_id="web", title=item.get("title", "Web result"), page=0, excerpt=item.get("content", "")[:500], score=float(item.get("score", 0)), url=item.get("url")) for i, item in enumerate(web, 1)]
            else:
                answer_text = "I could not verify that question in the five indexed papers. Configure TAVILY_API_KEY to enable the corrective web-search branch."
                sources = []
            message = "Question is outside the indexed corpus."; docs = []
        else:
            answer_text = generate_answer(rewritten, docs)
            sources = [Source(number=i, chunk_id=d.id, paper_id=d.paper_id, title=d.title, page=d.page, section=d.section, excerpt=d.content[:500], score=d.score) for i, d in enumerate(docs, 1)]
        history.extend([{"role": "user", "content": query}, {"role": "assistant", "content": answer_text}])
        return Answer(answer=answer_text, sources=sources, provenance=provenance, query=query, rewritten_query=rewritten if rewritten != query else None, retrieval_strategy=strategy, latency_ms=(perf_counter()-started)*1000, inspector=docs, message=message)

    def clear_thread(self, thread_id: str) -> None:
        """Remove one conversation without affecting other users' sessions."""
        self.history.pop(thread_id, None)

_agent: ResearchPaperAgent | None = None
def get_agent() -> ResearchPaperAgent:
    global _agent
    if _agent is None: _agent = ResearchPaperAgent()
    return _agent
