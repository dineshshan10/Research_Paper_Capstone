"""Page-aware PDF ingestion with clean, citation-safe chunks."""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import fitz

from config.settings import CORPUS, DEFAULT_CHUNKING, Paper
from core.schema import PaperChunk


def clean_page_text(text: str) -> str:
    """Remove repeated whitespace, bibliography tails, and empty PDF artifacts."""
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # References are useful bibliographically, but are poor answer context.
    match = re.search(r"(?im)^\s*(references|bibliography)\s*$", text)
    return text[: match.start()].strip() if match else text


def detect_section(text: str, previous: str = "") -> str:
    for line in text.splitlines()[:10]:
        line = line.strip()
        if re.match(r"^(\d+(\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,:()\-]{2,80}$", line):
            return line
    return previous


def _token_windows(text: str, size: int, overlap: int) -> Iterable[str]:
    words = text.split()
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if len(window) >= 30:
            yield " ".join(window)
        if start + size >= len(words):
            break


def ingest_paper(paper: Paper, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[PaperChunk]:
    if not paper.path.exists():
        raise FileNotFoundError(paper.path)
    size = chunk_size or DEFAULT_CHUNKING.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNKING.chunk_overlap
    chunks: list[PaperChunk] = []
    section = ""
    document = fitz.open(paper.path)
    try:
        for page_number, page in enumerate(document, start=1):
            text = clean_page_text(page.get_text("text"))
            if len(text.split()) < 30:
                continue
            section = detect_section(text, section)
            for content in _token_windows(text, size, overlap):
                index = len(chunks)
                chunks.append(PaperChunk(
                    id=f"{paper.paper_id}-p{page_number:03d}-c{index:04d}", content=content,
                    paper_id=paper.paper_id, title=paper.title, page=page_number,
                    section=section, chunk_index=index,
                ))
    finally:
        document.close()
    return chunks


def ingest_corpus(chunk_size: int | None = None) -> list[PaperChunk]:
    return [chunk for paper in CORPUS for chunk in ingest_paper(paper, chunk_size)]


def corpus_report(chunks: list[PaperChunk]) -> dict[str, int]:
    return {paper.paper_id: sum(c.paper_id == paper.paper_id for c in chunks) for paper in CORPUS}
