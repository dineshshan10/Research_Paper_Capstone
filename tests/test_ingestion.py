from core.ingestion import ingest_corpus

def test_ingestion_preserves_citable_metadata():
    chunks = ingest_corpus(128)
    assert len(chunks) > 100
    assert {c.paper_id for c in chunks} == {"attention", "gpt4", "instructgpt", "mistral", "gemini"}
    assert all(c.page >= 1 and c.id.startswith(c.paper_id) for c in chunks)
