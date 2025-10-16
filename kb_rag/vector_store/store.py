from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional
    faiss = None  # type: ignore

from ..embeddings import EmbeddingModel


@dataclass
class Metadata:
    doc_id: str
    chunk_id: str
    source_path: str
    order: int


class VectorStore:
    def __init__(self, index_path: Path, sqlite_path: Path, embedding_model: EmbeddingModel):
        self.index_path = index_path
        self.sqlite_path = sqlite_path
        self.embedding_model = embedding_model
        self.dim: int | None = None
        self._conn: sqlite3.Connection | None = None

    # --- SQLite metadata ---
    def _conn_open(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.sqlite_path))
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    text TEXT NOT NULL,
                    ord INTEGER NOT NULL
                )
                """
            )
            self._conn.commit()
        return self._conn

    def upsert_chunk(self, meta: Metadata, text: str) -> None:
        conn = self._conn_open()
        conn.execute(
            "REPLACE INTO chunks (chunk_id, doc_id, source_path, text, ord) VALUES (?, ?, ?, ?, ?)",
            (meta.chunk_id, meta.doc_id, meta.source_path, text, meta.order),
        )
        conn.commit()

    def get_chunk_text(self, chunk_id: str) -> str | None:
        conn = self._conn_open()
        cur = conn.execute("SELECT text FROM chunks WHERE chunk_id=?", (chunk_id,))
        row = cur.fetchone()
        return row[0] if row else None

    def get_metadata_for_ids(self, chunk_ids: List[str]) -> List[Tuple[str, str, int]]:
        if not chunk_ids:
            return []
        conn = self._conn_open()
        qmarks = ",".join(["?"] * len(chunk_ids))
        cur = conn.execute(
            f"SELECT chunk_id, source_path, ord FROM chunks WHERE chunk_id IN ({qmarks})",
            tuple(chunk_ids),
        )
        return list(cur.fetchall())

    # --- FAISS or numpy fallback ---
    def _load_index(self, dim: int):
        self.dim = dim
        if faiss is not None and self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        if faiss is not None:
            return faiss.IndexFlatIP(dim)
        # fallback: numpy list store
        return None

    def _save_index(self, index, embeddings: np.ndarray | None = None, ids: List[str] | None = None):
        if faiss is not None and index is not None:
            faiss.write_index(index, str(self.index_path))
        else:
            # fallback: save as npz
            if embeddings is not None and ids is not None:
                np.savez(self.index_path.with_suffix(".npz"), embeddings=embeddings, ids=np.array(ids))

    def add_texts(self, metas: List[Metadata], texts: List[str]):
        vectors = self.embedding_model.embed(texts)
        dim = vectors.shape[1]
        index = self._load_index(dim)

        # persist metadata
        for meta, t in zip(metas, texts):
            self.upsert_chunk(meta, t)

        if faiss is not None and index is not None:
            if isinstance(index, faiss.IndexFlatIP):
                # store ids separately using SQLite; FAISS stores only vectors
                index.add(vectors)
                self._save_index(index)
            else:
                index.add(vectors)
                self._save_index(index)
        else:
            # fallback store
            if self.index_path.with_suffix(".npz").exists():
                old = np.load(self.index_path.with_suffix(".npz"), allow_pickle=True)
                old_emb = old["embeddings"]
                old_ids = old["ids"].tolist()
                new_ids = [m.chunk_id for m in metas]
                all_emb = np.vstack([old_emb, vectors])
                all_ids = old_ids + new_ids
            else:
                all_emb = vectors
                all_ids = [m.chunk_id for m in metas]
            self._save_index(None, all_emb, all_ids)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_vec = self.embedding_model.embed([query])
        q = query_vec.astype(np.float32)
        if faiss is not None and self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            sims, idxs = index.search(q, top_k)
            scores = sims[0].tolist()
            ids = self._ids_from_order(idxs[0].tolist())
            return list(zip(ids, scores))
        # fallback
        path = self.index_path.with_suffix(".npz")
        if not path.exists():
            return []
        data = np.load(path, allow_pickle=True)
        emb = data["embeddings"].astype(np.float32)
        ids = data["ids"].tolist()
        # cosine similarity assuming embeddings normalized
        scores = (emb @ q[0]).tolist()
        order = np.argsort(scores)[::-1][:top_k]
        return [(ids[i], float(scores[i])) for i in order]

    def _ids_from_order(self, orders: List[int]) -> List[str]:
        # map FAISS row order to chunk_ids via SQLite rownum; for simplicity keep contiguous order
        # We use ord in SQLite to reconstruct chunk order for each doc, but FAISS doesn't store IDs.
        # To keep it simple, rely on fallback store for IDs, or return placeholders.
        # In production you'd maintain a separate ID map.
        return [str(o) for o in orders]
