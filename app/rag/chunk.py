from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class Chunk:
    text: str
    metadata: dict


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Naive, robust character-based splitter with overlap.
    Attempts to respect paragraph boundaries when possible.
    """
    if not text:
        return []

    # Normalize line endings and collapse large whitespace regions
    normalized = "\n".join(line.rstrip() for line in text.splitlines())

    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]

    # If text is short, return as single chunk
    if sum(len(p) for p in paragraphs) <= chunk_size:
        return [normalized]

    # Build rolling chunks by concatenating paragraphs
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # account for two newlines when joining
        if current_len + para_len <= chunk_size:
            current.append(para)
            current_len += para_len
        else:
            if current:
                chunks.append("\n\n".join(current))
            # Start new chunk with overlap from end of previous chunk
            if chunk_overlap > 0 and chunks:
                overlap_source = chunks[-1]
                overlap = overlap_source[max(0, len(overlap_source) - chunk_overlap) :]
                current = [overlap, para]
                current_len = len(overlap) + para_len
            else:
                current = [para]
                current_len = para_len

    if current:
        chunks.append("\n\n".join(current))

    # If still too large chunks e.g., single massive paragraph, fall back to hard slice
    final_chunks: List[str] = []
    for ch in chunks:
        if len(ch) <= chunk_size + chunk_overlap:
            final_chunks.append(ch)
        else:
            start = 0
            while start < len(ch):
                end = min(start + chunk_size, len(ch))
                final_chunks.append(ch[start:end])
                if end == len(ch):
                    break
                start = max(end - chunk_overlap, 0)

    return final_chunks


def create_chunks(text: str, base_metadata: dict, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    parts = split_text_into_chunks(text, chunk_size, chunk_overlap)
    chunks: List[Chunk] = []
    for i, part in enumerate(parts):
        md = dict(base_metadata)
        md.update({
            "chunk_index": i,
            "chunk_char_count": len(part),
        })
        chunks.append(Chunk(text=part, metadata=md))
    return chunks
