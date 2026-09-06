"""Minimal timing helpers used by all entrypoints."""
from __future__ import annotations
from contextlib import contextmanager
from time import perf_counter

@contextmanager
def timed():
    start = perf_counter(); result = {"ms": 0.0}
    try: yield result
    finally: result["ms"] = (perf_counter() - start) * 1000
