# Research Paper Answer Bot

An evidence-first RAG application over five landmark Generative AI papers: Attention Is All You Need,
GPT-4, InstructGPT, Mistral 7B, and Gemini 1.0. Every response includes citable paper/page sources.

## What is included

- Page-aware PyMuPDF ingestion and persistent Chroma indexing.
- Four embedding configurations: MiniLM, BGE-base, GTE-large, and OpenAI `text-embedding-3-small`.
- Dense, BM25, hybrid RRF, rerank, and MMR retrieval strategies with a reproducible benchmark.
- Grounded Gemini synthesis when configured; faithful local extractive fallback otherwise.
- Corrective out-of-corpus branch (Tavily when configured), thread-isolated conversation memory,
  Streamlit retrieval inspector, FastAPI, CLI, tests, and a golden QA set.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest tests/ -v
```

No key is required for local ingestion, retrieval, tests, CLI, or UI. Set `GEMINI_API_KEY` for
generative synthesis, `OPENAI_API_KEY` to run the commercial embedding arm, and `TAVILY_API_KEY` for
the corrective web-search branch.

## Run all three entrypoints

Activate the virtual environment first:

```bash
cd "/path/to/Research_Paper_Capstone"
source venv/bin/activate
```

### 1. Streamlit research workspace

```bash
streamlit run entrypoints/ui.py --server.port 8501
```

Open `http://localhost:8501`. The workspace includes research chat, corpus, evaluation, and conversation-guide tabs.

### 2. FastAPI service

```bash
uvicorn entrypoints.api:app --reload --port 8000
```

Open the interactive API documentation at `http://localhost:8000/docs`, or submit a cited question:

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"How does Mistral improve inference efficiency?","thread_id":"demo"}'
```

### 3. Interactive terminal CLI

```bash
python cli.py
```

Type a research question at the `You:` prompt. Type `exit` or `quit` to close the session.

### Optional: run the retrieval evaluation

```bash
python -m evaluation.run_benchmark
```

The API exposes `GET /health`, `POST /api/v1/ask`, and `GET /api/v1/sources`.

## Evaluation

`python -m evaluation.run_benchmark` builds the corpus, evaluates all configured embedding and
retrieval variants against `evaluation/golden_qa.json`, and writes Markdown tables to
`evaluation/results/`. Run it with downloaded local models/API credentials before submitting results;
the framework intentionally does not misrepresent its offline fallback as a neural embedding result.

See [the project plan](docs/PROJECT_PLAN.md) and [the design document](docs/DESIGN_DOC.md).
