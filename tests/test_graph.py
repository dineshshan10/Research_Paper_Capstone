from config.settings import EMBEDDING_REGISTRY
from core.embeddings import HashFallbackEmbedding
from core.graph import ResearchPaperAgent
from core.schema import PaperChunk
from core.vector_store import PaperVectorStore

def test_out_of_corpus_question_is_not_fabricated():
    chunks=[PaperChunk(id="attention-1", content="Transformers use self attention.", paper_id="attention", title="Attention", page=1, chunk_index=0)]
    agent=ResearchPaperAgent(chunks=chunks, embedding_provider=HashFallbackEmbedding(EMBEDDING_REGISTRY["minilm"]))
    answer=agent.ask("What is DeepSeek-R1 training recipe?", "negative")
    assert answer.provenance == "unavailable" and not answer.sources
def test_history_is_isolated_by_thread():
    chunks=[PaperChunk(id="mistral-1", content="Mistral sliding window attention has a fixed context window.", paper_id="mistral", title="Mistral", page=1, chunk_index=0)]
    agent=ResearchPaperAgent(chunks=chunks, embedding_provider=HashFallbackEmbedding(EMBEDDING_REGISTRY["minilm"]))
    agent.ask("Tell me about Mistral", "one")
    assert agent.history["one"] and not agent.history.get("two")

def test_in_corpus_question_is_not_rejected():
    chunks=[PaperChunk(id="mistral-1", content="Mistral uses grouped-query attention and sliding window attention for efficient inference.", paper_id="mistral", title="Mistral", page=1, chunk_index=0)]
    agent=ResearchPaperAgent(chunks=chunks, embedding_provider=HashFallbackEmbedding(EMBEDDING_REGISTRY["minilm"]))
    answer=agent.ask("How does Mistral improve inference efficiency?", "positive")
    assert answer.provenance == "corpus" and answer.sources[0].paper_id == "mistral"

def test_request_tavily_key_reaches_corrective_search(monkeypatch):
    chunks=[PaperChunk(id="attention-1", content="Transformers use self attention.", paper_id="attention", title="Attention", page=1, chunk_index=0)]
    agent=ResearchPaperAgent(chunks=chunks, embedding_provider=HashFallbackEmbedding(EMBEDDING_REGISTRY["minilm"]))
    received = {}
    monkeypatch.setattr("core.graph.web_search", lambda query, key: received.update({"key": key}) or [])
    agent.ask("What is DeepSeek-R1 training recipe?", "web-key", tavily_api_key="session-key")
    assert received["key"] == "session-key"
