from __future__ import annotations

from typing import Iterable, List

import numpy as np
from fastembed import TextEmbedding


class FastEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = TextEmbedding(model_name)
        # Probe embedding dimension
        sample_vec = list(self.model.embed(["hello world"]))[0]
        self.dim = len(sample_vec)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        # Avoid divide by zero
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def embed_texts(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        embeddings = np.array(list(self.model.embed(texts)), dtype=np.float32)
        if normalize:
            embeddings = self._normalize(embeddings)
        return embeddings

    def embed_query(self, text: str, normalize: bool = True) -> np.ndarray:
        emb = np.array(list(self.model.embed([text])), dtype=np.float32)
        if normalize:
            emb = self._normalize(emb)
        return emb[0]
