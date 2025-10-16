from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np

try:
    # sentence-transformers is preferred for local embeddings
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


@dataclass
class EmbeddingModel:
    provider: str
    model_name: str

    def embed(self, texts: List[str]) -> np.ndarray:  # (n, d)
        provider = self.provider.lower()
        if provider == "openai":
            return self._embed_openai(texts)
        return self._embed_local(texts)

    # --- Providers ---
    def _embed_local(self, texts: List[str]) -> np.ndarray:
        if SentenceTransformer is None:
            # Tiny deterministic hashing embedding as last resort
            return _hash_embed(texts)
        model = SentenceTransformer(self.model_name)
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.astype(np.float32)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        from httpx import Client

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        # Use text-embedding-3-small by default for cost-efficiency
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"input": texts, "model": model}
        with Client(timeout=60) as client:
            resp = client.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        vectors = [item["embedding"] for item in data["data"]]
        arr = np.array(vectors, dtype=np.float32)
        # normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype(np.float32)


# --- Utilities ---

def _hash_embed(texts: List[str], dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(42)
    # fixed random projection
    projections = rng.standard_normal((dim, 2048)).astype(np.float32)
    out = []
    for t in texts:
        # simple bag-of-chars hashed vector
        vec = np.zeros(2048, dtype=np.float32)
        for ch in t:
            vec[ord(ch) % 2048] += 1.0
        # project
        proj = projections @ vec
        # normalize
        proj /= (np.linalg.norm(proj) + 1e-12)
        out.append(proj)
    return np.vstack(out).astype(np.float32)
