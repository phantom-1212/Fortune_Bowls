#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from app.config import settings
from app.ingestion import extract_chunks_from_path
from app.retriever import VectorStore


def collect_files(root: Path) -> List[Path]:
    supported = {".pdf", ".txt", ".md"}
    if root.is_file():
        return [root] if root.suffix.lower() in supported else []
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.suffix.lower() in supported and p.is_file():
            files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument("path", type=str, help="File or directory to ingest")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    store = VectorStore(settings.index_dir, settings.embedding_model)

    files = collect_files(root)
    total_chunks = 0
    for f in files:
        chunks = extract_chunks_from_path(f)
        added, _ = store.add_chunks(chunks)
        total_chunks += added
        print(f"Ingested {f.name}: {added} chunks")

    print(f"Done. Files: {len(files)}, chunks: {total_chunks}")


if __name__ == "__main__":
    main()
