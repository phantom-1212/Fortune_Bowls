from __future__ import annotations

import io
import os
from typing import Iterable, List, Tuple

from .config import settings

try:
    import pypdf  # type: ignore
except Exception:  # pragma: no cover - optional dependency until installed
    pypdf = None


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf is not installed. Please install pypdf to parse PDFs.")
    reader = pypdf.PdfReader(path)
    texts: List[str] = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def read_file_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        return _read_txt(path)
    if ext == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"Unsupported file type: {ext}")


def simple_tokenize(text: str) -> List[str]:
    return text.split()


def chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    tokens = simple_tokenize(text)
    chunks: List[str] = []
    if not tokens:
        return chunks
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = max(0, end - overlap_tokens)
    return chunks


def load_and_chunk(path: str) -> Tuple[List[str], List[str]]:
    """Returns (chunks, metadata_sources)."""
    text = read_file_text(path)
    chunks = chunk_text(text, settings.max_chunk_tokens, settings.chunk_overlap_tokens)
    sources = [path for _ in chunks]
    return chunks, sources
