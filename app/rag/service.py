from __future__ import annotations

from typing import List, Sequence

from openai import OpenAI

from app.config import Settings
from app.rag.chunk import Chunk, create_chunks
from app.rag.embeddings import FastEmbedder
from app.rag.extract import extract_text_from_file
from app.rag.vectorstore import FAISSVectorStore, Retrieved


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = FastEmbedder(settings.embedding_model_name)
        self.vstore = FAISSVectorStore(
            index_path=settings.faiss_index_path,
            docstore_path=settings.docstore_path,
            dim=self.embedder.dim,
        )
        # OpenAI client may require API key via env or settings
        self.llm = OpenAI()

    # --------------------- Ingestion ---------------------
    def ingest_files(self, file_paths: List[str]) -> dict:
        all_chunks: List[Chunk] = []
        for path_str in file_paths:
            text, base_md = extract_text_from_file(self.settings.uploads_dir / path_str if not path_str.startswith("/") else path_str)  # type: ignore
            chunks = create_chunks(
                text,
                base_metadata=base_md,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            return {"chunks_added": 0, "files_count": len(file_paths)}

        texts = [c.text for c in all_chunks]
        metadatas = [c.metadata for c in all_chunks]
        embs = self.embedder.embed_texts(texts)
        added = self.vstore.add(embs, texts, metadatas)
        return {"chunks_added": added, "files_count": len(file_paths)}

    # --------------------- Retrieval ---------------------
    def retrieve(self, query: str, top_k: int) -> List[Retrieved]:
        top_k = max(1, min(self.settings.top_k_max, top_k or self.settings.top_k_default))
        q = self.embedder.embed_query(query)
        return self.vstore.search(q, top_k)

    # --------------------- Synthesis ---------------------
    def synthesize(self, query: str, contexts: List[Retrieved]) -> str:
        if not contexts:
            return "I couldn't find relevant information in the knowledge base. Please ingest documents first."

        context_text = "\n\n".join(
            f"[Source: {c.metadata.get('source_name','unknown')} | Score: {c.score:.3f}]\n{c.text}"
            for c in contexts
        )

        system_prompt = (
            "You are a helpful assistant. Using ONLY the provided context, answer the user's question succinctly. "
            "If the answer is not present in the context, say you don't know. Include 1-2 brief citations in parentheses using the source_name."
        )

        user_prompt = (
            f"Question: {query}\n\nContext:\n{context_text}\n\nAnswer succinctly:"
        )

        try:
            resp = self.llm.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            # Fallback: simple extractive summary
            return (
                "(LLM unavailable) Top matches:\n\n" + "\n\n".join(c.text[:500] for c in contexts)
            )

    def answer(self, query: str, top_k: int) -> dict:
        retrieved = self.retrieve(query, top_k)
        answer = self.synthesize(query, retrieved)
        sources = [
            {
                "source_name": r.metadata.get("source_name"),
                "source_path": r.metadata.get("source_path"),
                "score": r.score,
                "chunk_index": r.metadata.get("chunk_index"),
            }
            for r in retrieved
        ]
        return {"answer": answer, "sources": sources}
