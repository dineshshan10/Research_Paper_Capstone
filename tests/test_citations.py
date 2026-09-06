from core.nodes.generate_node import generate_answer
from core.schema import RetrievedChunk

def test_offline_answer_maps_to_numbered_sources():
    docs=[RetrievedChunk(id="a", content="The encoder has self-attention and a feed-forward network.", paper_id="attention", title="Attention", page=3, chunk_index=0, score=.8)]
    assert "[1]" in generate_answer("What is in the encoder?", docs)
