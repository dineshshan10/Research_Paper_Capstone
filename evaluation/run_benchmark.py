"""Reproducible retrieval bake-off; writes markdown results without requiring an LLM."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
from config.settings import EMBEDDING_REGISTRY, RESULTS_DIR, RETRIEVAL_STRATEGIES
from core.embeddings import get_embedding_provider
from core.ingestion import ingest_corpus
from core.retrievers import PaperRetriever
from core.vector_store import PaperVectorStore
from evaluation.metrics import hit_rate_at_k, ndcg_at_k, reciprocal_rank

def evaluate(retriever, cases, strategy):
    rows=[]
    for case in cases:
        started=perf_counter(); docs=retriever.retrieve(case["query"], strategy, 5); papers=[d.paper_id for d in docs]
        rows.append((hit_rate_at_k(set(case["papers"]), papers), reciprocal_rank(set(case["papers"]), papers), ndcg_at_k(set(case["papers"]), papers), (perf_counter()-started)*1000))
    return tuple(sum(row[i] for row in rows)/len(rows) for i in range(4))
def main():
    cases=json.loads((Path(__file__).parent/'golden_qa.json').read_text()); chunks=ingest_corpus(256); RESULTS_DIR.mkdir(exist_ok=True)
    embedding_rows=[]; winner="bge_base"; best=-1
    for key, spec in EMBEDDING_REGISTRY.items():
        provider=get_embedding_provider(key); provider_name=provider.__class__.__name__; store=PaperVectorStore(provider, chunks); started=perf_counter(); store.index(); index_ms=(perf_counter()-started)*1000
        score=evaluate(PaperRetriever(store), cases, "dense"); embedding_rows.append((key, spec.dimensions, *score, index_ms))
        if score[0] > best and provider.__class__.__name__ != "HashFallbackEmbedding": winner, best=key, score[0]
    lines=["# Embedding bake-off", "", "This run uses deterministic fallback vectors when local weights or credentials are unavailable; those rows are not neural-model claims.", "", "| model | dims | Hit@5 | MRR | nDCG@5 | query ms | index ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.1f} | {r[6]:.1f} |" for r in embedding_rows]
    (RESULTS_DIR/'embedding_comparison.md').write_text('\n'.join(lines)+'\n')
    store=PaperVectorStore(get_embedding_provider(winner), chunks); store.index(); retriever=PaperRetriever(store)
    rows=[(s,*evaluate(retriever,cases,s)) for s in RETRIEVAL_STRATEGIES]
    lines=["# Retrieval bake-off", "", f"Production candidate: `{winner}`", "", "| strategy | Hit@5 | MRR | nDCG@5 | query ms |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.1f} |" for r in rows]
    (RESULTS_DIR/'retrieval_comparison.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
if __name__ == '__main__': main()
