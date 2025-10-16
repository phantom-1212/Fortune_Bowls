from __future__ import annotations

import json
import os
from typing import List, Tuple, Optional

import numpy as np

from .config import settings

try:
    import faiss_cpu as faiss  # type: ignore
except Exception:  # pragma: no cover
    try:
        import faiss  # type: ignore
    except Exception:
        faiss = None


class VectorIndex:
    def __init__(self, dim: int):
        if faiss is None:
            raise RuntimeError("faiss is not installed. Please install faiss-cpu")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts: List[str] = []
        self.sources: List[str] = []

    @staticmethod
    def normalize_rows(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def add(self, embeddings: List[List[float]], texts: List[str], sources: List[str]) -> None:
        arr = np.array(embeddings, dtype="float32")
        arr = self.normalize_rows(arr)
        self.index.add(arr)
        self.texts.extend(texts)
        self.sources.extend(sources)

    def search(self, query_embedding: List[float], top_k: int) -> Tuple[List[str], List[str], List[float]]:
        q = np.array([query_embedding], dtype="float32")
        q = self.normalize_rows(q)
        scores, idxs = self.index.search(q, top_k)
        results_text: List[str] = []
        results_src: List[str] = []
        results_score: List[float] = []
        for i, score in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            results_text.append(self.texts[i])
            results_src.append(self.sources[i])
            results_score.append(float(score))
        return results_text, results_src, results_score


_GLOBAL_INDEX: VectorIndex | None = None
_META_CACHE: Optional[dict] = None
_INDEX_PATH = os.path.join(settings.index_dir, "index.faiss")
_META_PATH = os.path.join(settings.index_dir, "meta.json")


def get_or_create_index(embedding_dim: int) -> VectorIndex:
    global _GLOBAL_INDEX
    if _GLOBAL_INDEX is None:
        _GLOBAL_INDEX = VectorIndex(embedding_dim)
    return _GLOBAL_INDEX


def persist_global_index() -> None:
    """Persist the in-memory FAISS index and metadata to disk."""
    if _GLOBAL_INDEX is None:
        return
    if faiss is None:
        print("[warn] faiss not installed; skipping index persistence")
        return
    os.makedirs(settings.index_dir, exist_ok=True)
    faiss.write_index(_GLOBAL_INDEX.index, _INDEX_PATH)  # type: ignore[arg-type]
    meta = {
        "dim": _GLOBAL_INDEX.dim,
        "texts": _GLOBAL_INDEX.texts,
        "sources": _GLOBAL_INDEX.sources,
    }
    with open(_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def load_persistent_index_if_available() -> bool:
    """Load a persisted index if present. Returns True if loaded."""
    global _GLOBAL_INDEX, _META_CACHE
    if _GLOBAL_INDEX is not None:
        return True
    if not (os.path.isfile(_INDEX_PATH) and os.path.isfile(_META_PATH)):
        return False
    if faiss is None:
        print("[warn] faiss not installed; cannot load persisted index")
        return False
    try:
        with open(_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        dim = int(meta.get("dim"))
        index = faiss.read_index(_INDEX_PATH)  # type: ignore[assignment]
        vi = VectorIndex(dim)
        vi.index = index
        vi.texts = list(meta.get("texts", []))
        vi.sources = list(meta.get("sources", []))
        _GLOBAL_INDEX = vi
        _META_CACHE = meta
        return True
    except Exception as e:  # pragma: no cover
        print(f"[warn] failed to load persisted index: {e}")
        return False


def get_global_index_dim() -> Optional[int]:
    """Return the embedding dimension of the current or persisted index if available."""
    global _META_CACHE
    if _GLOBAL_INDEX is not None:
        return _GLOBAL_INDEX.dim
    # Lazy read meta if exists
    if _META_CACHE is None and os.path.isfile(_META_PATH):
        try:
            with open(_META_PATH, "r", encoding="utf-8") as f:
                _META_CACHE = json.load(f)
        except Exception:
            _META_CACHE = None
    if _META_CACHE and "dim" in _META_CACHE:
        try:
            return int(_META_CACHE["dim"])  # type: ignore[return-value]
        except Exception:
            return None
    return None
