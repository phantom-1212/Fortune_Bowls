from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib
import time

from pypdf import PdfReader

from .config import settings


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: Dict


def _stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, str]]:
    if not text:
        return []
    text = text.replace("\n\n", "\n").strip()
    chunks: List[Tuple[int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        # try to end on a sentence boundary if possible
        last_period = chunk.rfind(". ")
        if last_period != -1 and end != n and (end - (start + last_period)) < 200:
            end = start + last_period + 1
            chunk = text[start:end]
        chunks.append((len(chunks), chunk.strip()))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def _extract_pdf_chunks(file_bytes: bytes, source_name: str) -> List[DocumentChunk]:
    reader = PdfReader(BytesIO(file_bytes))
    all_chunks: List[DocumentChunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        for chunk_index, chunk_text in _split_text_into_chunks(
            page_text, settings.chunk_size_chars, settings.chunk_overlap_chars
        ):
            chunk_id = _stable_hash(f"{source_name}|p{page_number}|c{chunk_index}|{chunk_text[:50]}")
            metadata = {
                "source": source_name,
                "page": page_number,
                "chunk_index": chunk_index,
                "created_at": int(time.time()),
                "content_type": "application/pdf",
            }
            all_chunks.append(DocumentChunk(chunk_id=chunk_id, text=chunk_text, metadata=metadata))
    return all_chunks


def _extract_text_chunks(text: str, source_name: str) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    for chunk_index, chunk_text in _split_text_into_chunks(
        text, settings.chunk_size_chars, settings.chunk_overlap_chars
    ):
        chunk_id = _stable_hash(f"{source_name}|c{chunk_index}|{chunk_text[:50]}")
        metadata = {
            "source": source_name,
            "page": None,
            "chunk_index": chunk_index,
            "created_at": int(time.time()),
            "content_type": "text/plain",
        }
        chunks.append(DocumentChunk(chunk_id=chunk_id, text=chunk_text, metadata=metadata))
    return chunks


def load_file_bytes(path: Path) -> Tuple[bytes, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return path.read_bytes(), "application/pdf"
    else:
        return path.read_bytes(), "text/plain"


def extract_chunks_from_path(path: Path) -> List[DocumentChunk]:
    raw, content_type = load_file_bytes(path)
    if content_type == "application/pdf":
        return _extract_pdf_chunks(raw, path.name)
    else:
        text = raw.decode("utf-8", errors="ignore")
        return _extract_text_chunks(text, path.name)


def save_upload_to_disk(filename: str, content: bytes) -> Path:
    dest = Path(settings.uploads_dir) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return dest


def extract_chunks_from_upload(filename: str, content: bytes, content_type: Optional[str]) -> List[DocumentChunk]:
    if (content_type or "").startswith("application/pdf") or filename.lower().endswith(".pdf"):
        return _extract_pdf_chunks(content, filename)
    else:
        text = content.decode("utf-8", errors="ignore")
        return _extract_text_chunks(text, filename)
