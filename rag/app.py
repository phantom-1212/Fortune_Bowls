from __future__ import annotations

import os
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .config import settings
from .schemas import IngestResponse, QueryRequest, QueryAnswer
from .parser import load_and_chunk
from .embeddings import embed_texts
from .index import (
    get_or_create_index,
    load_persistent_index_if_available,
    persist_global_index,
    get_global_index_dim,
)
from .llm import synthesize_answer

app = FastAPI(title="RAG Search API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.on_event("startup")
async def on_startup() -> None:
    os.makedirs(settings.storage_dir, exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.index_dir, exist_ok=True)
    # Load index if exists
    load_persistent_index_if_available()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

# Serve the minimal frontend at /ui
if os.path.isdir("/workspace/frontend"):
    app.mount("/ui", StaticFiles(directory="/workspace/frontend", html=True), name="frontend")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    all_chunks: List[str] = []
    all_sources: List[str] = []
    saved_files: List[str] = []

    for f in files:
        # Save to uploads dir with a UUID prefix to avoid collisions
        filename = f"{uuid.uuid4().hex}_{os.path.basename(f.filename)}"
        save_path = os.path.join(settings.upload_dir, filename)
        with open(save_path, "wb") as out:
            content = await f.read()
            out.write(content)
        saved_files.append(save_path)

        # Load and chunk file
        chunks, sources = load_and_chunk(save_path)
        all_chunks.extend(chunks)
        all_sources.extend(sources)

    if not all_chunks:
        return IngestResponse(files=saved_files, chunks_indexed=0)

    # Embed and index
    embeddings = embed_texts(all_chunks)
    dim = len(embeddings[0])

    existing_dim = get_global_index_dim()
    if existing_dim is not None and existing_dim != dim:
        raise HTTPException(
            status_code=400,
            detail=(
                "Embedding dimension mismatch. Ensure consistent EMBEDDING_MODEL across ingestions."
            ),
        )

    index = get_or_create_index(dim)
    index.add(embeddings, all_chunks, all_sources)

    # Persist to disk
    persist_global_index()

    return IngestResponse(files=saved_files, chunks_indexed=len(all_chunks))


@app.post("/query", response_model=QueryAnswer)
async def query(body: QueryRequest) -> QueryAnswer:
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    top_k = body.top_k or settings.top_k

    # Ensure index exists
    dim = get_global_index_dim()
    if dim is None:
        raise HTTPException(status_code=400, detail="No index available. Ingest documents first.")

    # Embed query and search
    q_emb = embed_texts([body.query])[0]
    index = get_or_create_index(len(q_emb))
    texts, sources, _scores = index.search(q_emb, top_k)

    if not texts:
        return QueryAnswer(answer="I don't know.", sources=[], matched_chunks=0)

    answer = synthesize_answer(body.query, texts)
    return QueryAnswer(answer=answer, sources=sources, matched_chunks=len(texts))
