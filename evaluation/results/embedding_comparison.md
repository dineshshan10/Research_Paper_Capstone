# Embedding bake-off

This run uses deterministic fallback vectors when local weights or credentials are unavailable; those rows are not neural-model claims.

| model | dims | Hit@5 | MRR | nDCG@5 | query ms | index ms |
|---|---:|---:|---:|---:|---:|---:|
| minilm | 384 | 0.833 | 0.742 | 0.761 | 7.8 | 90.2 |
| bge_base | 768 | 0.833 | 0.697 | 0.734 | 15.3 | 94.4 |
| gte_large | 1024 | 0.867 | 0.739 | 0.753 | 20.9 | 96.6 |
| openai_small | 1536 | 0.867 | 0.769 | 0.784 | 30.7 | 108.9 |
