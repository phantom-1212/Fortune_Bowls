from __future__ import annotations

from pathlib import Path

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None  # type: ignore


def pdf_to_text(path: str) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed; install via: pip install pdfplumber")
    text_parts: list[str] = []
    with pdfplumber.open(Path(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)
