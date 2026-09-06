"""Build the persistent Chroma collection for a selected embedding provider."""
import argparse
from config.settings import DEFAULT_EMBEDDING_KEY
from core.embeddings import get_embedding_provider
from core.ingestion import ingest_corpus, corpus_report
from core.vector_store import index_to_chroma

parser = argparse.ArgumentParser()
parser.add_argument("--embedding", default=DEFAULT_EMBEDDING_KEY)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
chunks = ingest_corpus()
print("Chunks:", corpus_report(chunks))
print("Indexed:", index_to_chroma(chunks, get_embedding_provider(args.embedding), args.force))
