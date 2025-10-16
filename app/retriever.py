from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings
from .ingestion import DocumentChunk


@dataclass
class SearchResult:
    score: float
    chunk: DocumentChunk


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)
        try:
            self.dimension = self.model.get_sentence_embedding_dimension()  # type: ignore[attr-defined]
        except Exception:
            # Fallback: compute once
            self.dimension = int(self.embed_texts([""]).shape[1])

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=False)
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        return embeddings.astype("float32")

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])


class VectorStore:
    def __init__(self, index_dir: str, embedding_model_name: str) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "embeddings.faiss"
        self.meta_path = self.index_dir / "meta.jsonl"

        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.index = None  # type: ignore
        self.metadata: List[Dict] = []

        self._load_if_exists()

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, ord=2, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def _create_index(self) -> None:
        self.index = faiss.IndexFlatIP(self.embedding_model.dimension)

    def _load_if_exists(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with self.meta_path.open("r", encoding="utf-8") as f:
                self.metadata = [json.loads(line) for line in f]
        else:
            self._create_index()
            self.metadata = []

    def _save(self) -> None:
        if not settings.persist_index:
            return
        faiss.write_index(self.index, str(self.index_path))  # type: ignore[arg-type]
        with self.meta_path.open("w", encoding="utf-8") as f:
            for meta in self.metadata:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def add_chunks(self, chunks: List[DocumentChunk]) -> Tuple[int, int]:
        if not chunks:
            return 0, 0
        texts = [c.text for c in chunks]
        vectors = self.embedding_model.embed_texts(texts)
        vectors = self._l2_normalize(vectors)
        self.index.add(vectors)  # type: ignore[union-attr]
        # Persist metadata aligned with new vectors
        for c in chunks:
            self.metadata.append({
                "chunk_id": c.chunk_id,
                "text": c.text,
                **c.metadata,
            })
        self._save()
        return len(texts), vectors.shape[1]

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        if self.index is None or len(self.metadata) == 0:
            return []
        q = self.embedding_model.embed_text(query)
        q = self._l2_normalize(q)
        distances, indices = self.index.search(q, min(top_k, len(self.metadata)))  # type: ignore[union-attr]
        results: List[SearchResult] = []
        for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            chunk = DocumentChunk(
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                metadata={k: v for k, v in meta.items() if k not in {"chunk_id", "text"}},
            )
            results.append(SearchResult(score=float(score), chunk=chunk))
        return results
