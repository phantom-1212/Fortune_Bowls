from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import faiss
import numpy as np


@dataclass
class Retrieved:
    text: str
    metadata: dict
    score: float


class FAISSVectorStore:
    def __init__(self, index_path: Path, docstore_path: Path, dim: int) -> None:
        self.index_path = index_path
        self.docstore_path = docstore_path
        self.dim = dim

        self.texts: List[str] = []
        self.metadatas: List[dict] = []

        self.index = self._load_or_create_index()
        self._load_docstore()

        # If mismatch, reset to empty safe state
        if self.index.ntotal != len(self.texts):
            # Reset to empty consistent store
            self.index = faiss.IndexFlatIP(self.dim)
            self.texts = []
            self.metadatas = []
            self._persist()

    def _load_or_create_index(self):
        if self.index_path.exists():
            try:
                return faiss.read_index(str(self.index_path))
            except Exception:
                pass
        return faiss.IndexFlatIP(self.dim)

    def _load_docstore(self) -> None:
        if not self.docstore_path.exists():
            return
        try:
            # Simple JSONL: {"text":..., "metadata":{...}}
            import json

            with self.docstore_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    self.texts.append(obj["text"]) 
                    self.metadatas.append(obj.get("metadata", {}))
        except Exception:
            # If corrupted, ignore
            self.texts = []
            self.metadatas = []

    def _append_docstore(self, texts: List[str], metadatas: List[dict]) -> None:
        import json

        self.docstore_path.parent.mkdir(parents=True, exist_ok=True)
        with self.docstore_path.open("a", encoding="utf-8") as f:
            for t, m in zip(texts, metadatas):
                f.write(json.dumps({"text": t, "metadata": m}, ensure_ascii=False) + "\n")

    def _persist(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        # Re-write docstore fully to keep it compact and in sync
        import json
        tmp_path = self.docstore_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for t, m in zip(self.texts, self.metadatas):
                f.write(json.dumps({"text": t, "metadata": m}, ensure_ascii=False) + "\n")
        tmp_path.replace(self.docstore_path)

    def add(self, embeddings: np.ndarray, texts: List[str], metadatas: List[dict]) -> int:
        assert embeddings.shape[1] == self.dim
        assert len(texts) == embeddings.shape[0] == len(metadatas)

        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

        # Append new entries and persist index
        self._append_docstore(texts, metadatas)
        self._persist()

        return len(texts)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Retrieved]:
        if self.index.ntotal == 0:
            return []
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)
        scores = scores[0]
        indices = indices[0]

        results: List[Retrieved] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            text = self.texts[idx]
            md = self.metadatas[idx]
            results.append(Retrieved(text=text, metadata=md, score=float(score)))
        return results
