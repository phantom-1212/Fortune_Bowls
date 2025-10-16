from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

from pypdf import PdfReader


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def extract_text_from_file(path: Path) -> Tuple[str, dict]:
    """
    Returns (text, metadata) for a supported file type.
    Supported: .txt, .md, .pdf
    """
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = read_text_file(path)
    elif suffix == ".pdf":
        text = read_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    metadata = {
        "source_path": str(path.resolve()),
        "source_name": path.name,
        "file_type": suffix,
    }
    return text, metadata
