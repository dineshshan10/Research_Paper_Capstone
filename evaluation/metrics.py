from __future__ import annotations
from math import log2

def hit_rate_at_k(actual: set[str], predicted: list[str], k: int = 5) -> float: return float(bool(actual & set(predicted[:k])))
def reciprocal_rank(actual: set[str], predicted: list[str]) -> float:
    return next((1 / rank for rank, value in enumerate(predicted, 1) if value in actual), 0.0)
def ndcg_at_k(actual: set[str], predicted: list[str], k: int = 5) -> float:
    # A paper can appear in multiple chunks; count each relevant paper at most once.
    seen: set[str] = set()
    dcg = 0.0
    for rank, value in enumerate(predicted[:k], 1):
        if value in actual and value not in seen:
            dcg += 1 / log2(rank + 1)
            seen.add(value)
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(actual)) + 1))
    return dcg / ideal if ideal else 0.0
