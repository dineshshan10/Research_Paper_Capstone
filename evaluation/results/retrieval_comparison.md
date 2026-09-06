# Retrieval bake-off

Production candidate: `bge_base`

| strategy | Hit@5 | MRR | nDCG@5 | query ms |
|---|---:|---:|---:|---:|
| dense | 0.833 | 0.697 | 0.734 | 15.3 |
| bm25 | 0.933 | 0.828 | 0.850 | 0.2 |
| hybrid | 0.933 | 0.878 | 0.892 | 15.6 |
| hybrid_rerank | 0.933 | 0.873 | 0.888 | 16.9 |
| mmr | 0.900 | 0.729 | 0.772 | 29.6 |
