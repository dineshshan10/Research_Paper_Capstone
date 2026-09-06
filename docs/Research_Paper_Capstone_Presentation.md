# Research Paper Answer Bot — Presentation Deck

## 1. Problem and goal

Researchers need trustworthy answers across long AI papers. This project delivers a cited RAG assistant over five landmark papers, with retrieval choices justified by a measurable bake-off.

## 2. Corpus and requirements

Five PDFs: Attention, GPT-4, InstructGPT, Mistral 7B, Gemini 1.0. The solution covers local vector indexing, open and commercial embeddings, five retrieval strategies, grounded RAG, testing and explicit sources.

## 3. Architecture

PyMuPDF ingestion → embedding provider → Chroma + BM25 → hybrid RRF/rerank → relevance grader → grounded generator → paper/page citations. Streamlit, API and CLI are adapters around the same core.

## 4. Citation-safe ingestion

Chunks preserve paper title, paper ID, page, section and chunk ID. References are excluded from answer context. This enables each answer citation to map directly to a primary source.

## 5. Embedding bake-off

Candidates are MiniLM, BGE-base, GTE-large and OpenAI text-embedding-3-small. The harness reports Hit@5, MRR, nDCG@5, index time and query latency. It does not claim fallback embeddings are neural results.

## 6. Retrieval bake-off

Dense cosine is the baseline. BM25 handles exact technical vocabulary. Hybrid RRF combines both; hybrid rerank is the production candidate; MMR provides diversity. Results are written to `evaluation/results/`.

## 7. Corrective and conversational RAG

History is isolated per thread. A relevance grader blocks unsupported claims, then uses Tavily when configured or returns a transparent out-of-corpus response. The policy prevents fabricated answers.

## 8. Product demonstration

The Streamlit chat renders citations, source excerpts, provenance, scores and latency in a Retrieval Inspector. FastAPI exposes `/health`, `/api/v1/ask`, and `/api/v1/sources`; a CLI supports terminal demos.

## 9. Quality and reproducibility

The golden QA set has 30 labeled questions, including cross-paper and negative cases. The automated suite tests ingestion, retrieval variants, citation numbering, memory isolation and out-of-corpus refusal.

## 10. Next steps

Warm local models, run the neural/commercial bake-off with configured keys, lock the winning configuration, add a table/figure extraction adapter, and deploy behind authentication.
