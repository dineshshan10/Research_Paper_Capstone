# Research Paper Answer Bot — Design Document

## Architecture

The application uses ports-and-adapters organization: `core/` owns ingestion, embeddings, retrieval,
grounded generation and corrective policy; `entrypoints/` exposes Streamlit, CLI, and FastAPI.
PyMuPDF preserves page metadata from the five primary PDFs. Each answer source therefore includes a
paper title, page number, chunk ID, excerpt and retrieval score.

## Retrieval and evidence policy

The system supports dense, BM25, hybrid reciprocal-rank fusion, hybrid reranking and MMR. The default
is `hybrid_rerank`: it provides semantic recall plus lexical precision for technical terms. The
benchmark writes its evidence to `evaluation/results/`; production defaults should be re-locked after
running the full benchmark with local models available.

Embeddings are provider-owned: query prefixes and normalisation live in the provider spec, not at call
sites. The OpenAI model is included for the required commercial comparison. If weights or credentials
are unavailable, a deterministic hashing fallback keeps the demo and tests operational; benchmark
results should clearly identify that fallback and must not be used to claim a neural-model result.

## Corrective RAG and memory

The grader checks overlap and known out-of-corpus intents. Insufficient corpus evidence enters the
Tavily web branch when configured; otherwise it refuses clearly rather than fabricating an answer.
Conversation history is isolated by `thread_id`, and follow-up prompts are rewritten with the prior
user turn. The rewrite loop is represented by `MAX_QUERY_REWRITES`, preventing unbounded retries.

## Security and limitations

API keys only enter through environment variables. The application never indexes a user upload and
does not provide authentication. PDF extraction is text-only, so figures and complex tables are out of
scope. Offline output is extractive, preserving grounding and citations but not the prose quality of
the optional Gemini synthesis adapter.
