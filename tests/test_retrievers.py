from config.settings import EMBEDDING_REGISTRY
from core.embeddings import HashFallbackEmbedding
from core.retrievers import PaperRetriever
from core.schema import PaperChunk
from core.vector_store import PaperVectorStore

def _retriever():
    chunks=[PaperChunk(id="attention-1", content="The Transformer uses multi-head self-attention and positional encodings.", paper_id="attention", title="Attention", page=1, chunk_index=0), PaperChunk(id="mistral-1", content="Mistral uses grouped-query attention and sliding window attention for efficient inference.", paper_id="mistral", title="Mistral", page=1, chunk_index=1), PaperChunk(id="gpt4-1", content="GPT-4 is a large multimodal model evaluated on many benchmarks.", paper_id="gpt4", title="GPT-4", page=1, chunk_index=2)]
    store=PaperVectorStore(HashFallbackEmbedding(EMBEDDING_REGISTRY["minilm"]), chunks); store.index(); return PaperRetriever(store)
def test_all_strategies_return_ranked_results():
    retriever=_retriever()
    for strategy in ("dense", "bm25", "hybrid", "hybrid_rerank", "mmr"):
        results = retriever.retrieve("grouped query attention", strategy, 1)
        assert len(results) == 1 and results[0].id
    assert retriever.bm25_search("grouped query attention", 1)[0].paper_id == "mistral"
