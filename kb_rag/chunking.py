from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    order: int


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - chunk_overlap)
    return chunks


def iter_chunks(doc_id: str, text: str, chunk_size: int, chunk_overlap: int) -> Iterable[Chunk]:
    parts = split_text(text, chunk_size, chunk_overlap)
    for i, part in enumerate(parts):
        yield Chunk(doc_id=doc_id, chunk_id=f"{doc_id}:{i}", text=part, order=i)


def read_text_from_path(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")
