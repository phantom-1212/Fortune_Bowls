from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import settings
from .embeddings import EmbeddingModel
from .vector_store.store import VectorStore, Metadata
from .chunking import iter_chunks


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    source_path: str
    order: int


def build_vector_store() -> VectorStore:
    emb = EmbeddingModel(provider=settings.embedding_provider, model_name=settings.embedding_model)
    return VectorStore(index_path=settings.index_path, sqlite_path=settings.sqlite_path, embedding_model=emb)


def ingest_document(doc_id: str, source_path: str, text: str) -> int:
    store = build_vector_store()
    metas: List[Metadata] = []
    texts: List[str] = []
    for ch in iter_chunks(doc_id, text, settings.chunk_size, settings.chunk_overlap):
        metas.append(Metadata(doc_id=doc_id, chunk_id=ch.chunk_id, source_path=source_path, order=ch.order))
        texts.append(ch.text)
    if texts:
        store.add_texts(metas, texts)
    return len(texts)


def retrieve(query: str, top_k: int = 5) -> List[RetrievedChunk]:
    store = build_vector_store()
    results = store.search(query, top_k=top_k)
    out: List[RetrievedChunk] = []
    # When FAISS is used, IDs are placeholders; we need to fallback to SQLite join by order not ideal.
    # For simplicity, fetch top_k most recent chunks from SQLite when IDs are not real numbers.
    for chunk_id, score in results:
        text = store.get_chunk_text(chunk_id)
        meta_rows = store.get_metadata_for_ids([chunk_id])
        if meta_rows and text is not None:
            cid, source, ordv = meta_rows[0]
            out.append(RetrievedChunk(chunk_id=cid, score=float(score), text=text, source_path=source, order=ordv))
    # Fallback: if out is empty but there are chunks, just return empty list here
    return out


def synthesize_answer(query: str, contexts: List[RetrievedChunk]) -> str:
    # Minimal synthesis to work offline without LLM
    if settings.llm_provider == "fallback":
        return _heuristic_summarize(query, contexts)
    try:
        return _llm_synthesize(query, contexts)
    except Exception:
        return _heuristic_summarize(query, contexts)


def _heuristic_summarize(query: str, contexts: List[RetrievedChunk]) -> str:
    pieces: List[str] = []
    for ch in contexts:
        pieces.append(ch.text.strip())
        if len("\n\n".join(pieces)) > 1500:
            break
    context = "\n\n".join(pieces)
    return f"Answer (heuristic): Based on the retrieved documents, here is a concise response to '{query}':\n\n{context[:1200]}"


def _llm_synthesize(query: str, contexts: List[RetrievedChunk]) -> str:
    prompt = _build_prompt(query, contexts)
    prov = settings.llm_provider.lower()
    if prov == "openai":
        import httpx

        api_key = settings.openai_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        headers = {"Authorization": f"Bearer {api_key}"}
        body = {
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": "You are a helpful expert. Using only the provided context, answer succinctly."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        with httpx.Client(timeout=60) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    elif prov == "ollama":
        import httpx

        host = settings.ollama_host or "http://localhost:11434"
        body = {"model": settings.ollama_model, "prompt": prompt, "stream": False}
        with httpx.Client(timeout=60) as client:
            resp = client.post(f"{host}/api/generate", json=body)
            resp.raise_for_status()
            data = resp.json()
        return data.get("response", "").strip()
    else:
        raise RuntimeError("Unknown LLM provider")


def _build_prompt(query: str, contexts: List[RetrievedChunk]) -> str:
    context_blocks: List[str] = []
    for ch in contexts:
        context_blocks.append(f"[source: {ch.source_path} | chunk: {ch.chunk_id}]\n{ch.text}")
    joined = "\n\n---\n\n".join(context_blocks)
    return (
        "Using these documents, answer the user's question succinctly.\n\n"  # guidance
        f"Question: {query}\n\n"
        f"Context:\n{joined[:6000]}"
    )
