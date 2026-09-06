"""Download the three local embedding models before an offline demo."""
from config.settings import EMBEDDING_REGISTRY
from core.embeddings import SentenceTransformerEmbedding

for key, spec in EMBEDDING_REGISTRY.items():
    if spec.kind == "huggingface":
        SentenceTransformerEmbedding(spec)
        print(f"Cached {key}: {spec.model_id}")
