"""Configuration, paths, and the model registry for the Research Paper Answer Bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "papers"
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
GOLDEN_QA_PATH = EVALUATION_DIR / "golden_qa.json"
RESULTS_DIR = EVALUATION_DIR / "results"

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Paper:
    """One source document in the corpus."""

    paper_id: str
    title: str
    filename: str
    year: int
    topics: tuple[str, ...]

    @property
    def path(self) -> Path:
        return PAPERS_DIR / self.filename


CORPUS: tuple[Paper, ...] = (
    Paper(
        paper_id="attention",
        title="Attention Is All You Need",
        filename="attention_paper.pdf",
        year=2017,
        topics=("transformer", "self-attention", "encoder-decoder", "positional encoding"),
    ),
    Paper(
        paper_id="gpt4",
        title="GPT-4 Technical Report",
        filename="gpt4.pdf",
        year=2023,
        topics=("frontier models", "evaluations", "scaling", "safety"),
    ),
    Paper(
        paper_id="instructgpt",
        title="Training Language Models to Follow Instructions with Human Feedback",
        filename="instructgpt.pdf",
        year=2022,
        topics=("RLHF", "alignment", "instruction following", "reward modeling"),
    ),
    Paper(
        paper_id="mistral",
        title="Mistral 7B",
        filename="mistral_paper.pdf",
        year=2023,
        topics=("efficient inference", "grouped-query attention", "sliding window attention"),
    ),
    Paper(
        paper_id="gemini",
        title="Gemini: A Family of Highly Capable Multimodal Models",
        filename="gemini_paper.pdf",
        year=2023,
        topics=("multimodality", "long context", "benchmarks", "model family"),
    ),
)

PAPERS_BY_ID: dict[str, Paper] = {p.paper_id: p for p in CORPUS}

# ---------------------------------------------------------------------------
# Embedding model registry
#
# Each provider owns its own prefixes, normalization policy, and sequence limit
# so call sites can never get them wrong. The bge models in particular require
# an instruction prefix on QUERIES ONLY -- omitting it silently costs several
# points of retrieval accuracy and looks like the model underperforming.
# ---------------------------------------------------------------------------

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EmbeddingSpec:
    """Everything needed to use one embedding model correctly and consistently."""

    key: str
    model_id: str
    kind: Literal["huggingface", "openai"]
    dimensions: int
    max_seq_length: int
    normalize: bool
    query_prefix: str = ""
    document_prefix: str = ""
    notes: str = ""

    @property
    def collection_name(self) -> str:
        """Chroma collection name -- one isolated collection per embedding model."""
        return f"papers_{self.key}"


EMBEDDING_REGISTRY: dict[str, EmbeddingSpec] = {
    "minilm": EmbeddingSpec(
        key="minilm",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        kind="huggingface",
        dimensions=384,
        max_seq_length=256,
        normalize=True,
        notes="Small/fast tier. 22M params. Included as the speed-vs-quality datapoint.",
    ),
    "bge_base": EmbeddingSpec(
        key="bge_base",
        model_id="BAAI/bge-base-en-v1.5",
        kind="huggingface",
        dimensions=768,
        max_seq_length=512,
        normalize=True,
        query_prefix=BGE_QUERY_PREFIX,
        notes="Expected production default. Query prefix is required; documents get none.",
    ),
    "gte_large": EmbeddingSpec(
        key="gte_large",
        model_id="thenlper/gte-large",
        kind="huggingface",
        dimensions=1024,
        max_seq_length=512,
        normalize=True,
        notes="Largest local model (~1.3GB). Strong quality, heavier RAM/latency.",
    ),
    "openai_small": EmbeddingSpec(
        key="openai_small",
        model_id="text-embedding-3-small",
        kind="openai",
        dimensions=1536,
        max_seq_length=8191,
        normalize=False,
        notes="Commercial benchmark challenger (C3). ~$0.02/1M tokens.",
    ),
}

#: Locked in Phase 6 once the bake-off has run. See docs/DESIGN_DOC.md for rationale.
DEFAULT_EMBEDDING_KEY = "bge_base"

#: Tightest sequence limit across all registered models (256, set by MiniLM).
#: Any chunk longer than this truncates on at least one model in the sweep.
MIN_MODEL_SEQ_LENGTH = min(spec.max_seq_length for spec in EMBEDDING_REGISTRY.values())

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkConfig:
    """Chunking parameters. Sizes are in tokens, not characters."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


DEFAULT_CHUNKING = ChunkConfig()

#: Dimension A (embedding bake-off) runs at this size so that ALL FOUR models see
#: the complete chunk. MiniLM's sentence-transformers config caps at 256 tokens, so
#: comparing models at 512 would silently truncate MiniLM alone and score a config
#: artifact rather than the model. See docs/PROJECT_PLAN.md section 2.4.
FAIR_COMPARISON_CHUNK_SIZE = MIN_MODEL_SEQ_LENGTH  # 256

#: Dimension C (chunk-size sweep) runs on the Dimension A winner only. Ceiling is 512
#: because bge/gte truncate above it; if the winner is openai_small (8191) the sweep
#: may be extended, but cross-model comparison stops being valid past 512.
CHUNK_SIZE_SWEEP: tuple[int, ...] = (256, 384, 512)


def max_safe_chunk_size(*embedding_keys: str) -> int:
    """Largest chunk size that will not truncate on any of the given models.

    Ingestion and the benchmark harness call this to fail loudly instead of
    letting a model silently drop the tail of every chunk.
    """
    keys = embedding_keys or tuple(EMBEDDING_REGISTRY)
    return min(EMBEDDING_REGISTRY[k].max_seq_length for k in keys)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

RetrievalStrategy = Literal["dense", "bm25", "hybrid", "hybrid_rerank", "mmr"]

RETRIEVAL_STRATEGIES: tuple[RetrievalStrategy, ...] = (
    "dense",
    "bm25",
    "hybrid",
    "hybrid_rerank",
    "mmr",
)

#: Locked in Phase 6 once the bake-off has run.
DEFAULT_RETRIEVAL_STRATEGY: RetrievalStrategy = "hybrid_rerank"

TOP_K_RETRIEVAL = 10  # candidates pulled before reranking
TOP_K_CONTEXT = 3     # sources shown to the user and fed to the LLM (C7)
RRF_K = 60            # reciprocal rank fusion constant for hybrid retrieval
MMR_LAMBDA = 0.5      # 0 = max diversity, 1 = max relevance

RERANKER_MODEL_ID = "BAAI/bge-reranker-base"

# ---------------------------------------------------------------------------
# Generation and agentic control
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"
GENERATION_TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 1536

#: Corrective RAG (S3): hard cap so the grade -> rewrite -> retrieve loop cannot spin.
MAX_QUERY_REWRITES = 2
RELEVANCE_THRESHOLD = 0.5
WEB_SEARCH_RESULTS = 3

# ---------------------------------------------------------------------------
# API keys (all optional -- the system degrades rather than crashes)
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


@dataclass(frozen=True)
class Capabilities:
    """Which optional features are available given the configured keys."""

    generation: bool = field(default=False)
    openai_embeddings: bool = field(default=False)
    web_search: bool = field(default=False)


def get_capabilities() -> Capabilities:
    """Report which API-backed features are usable in the current environment.

    Local embeddings, retrieval, and the benchmark harness always work; only
    generation, the OpenAI arm of the bake-off, and corrective web search
    depend on keys.
    """
    return Capabilities(
        generation=bool(GEMINI_API_KEY),
        openai_embeddings=bool(OPENAI_API_KEY),
        web_search=bool(TAVILY_API_KEY),
    )
