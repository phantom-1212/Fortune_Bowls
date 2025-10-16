from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import settings, UPLOADS_DIR
from .chunking import read_text_from_path
from .pdf_loader import pdf_to_text
from .rag import ingest_document, retrieve, synthesize_answer

app = FastAPI(title="KB RAG")

static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def home():
    html = (Path(static_dir / "index.html").read_text() if (static_dir / "index.html").exists() else "<h1>KB RAG</h1>")
    return HTMLResponse(content=html)


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    counts = []
    for f in files:
        filename = Path(f.filename).name
        dest = UPLOADS_DIR / filename
        content = await f.read()
        dest.write_bytes(content)

        text = ""
        if filename.lower().endswith(".pdf"):
            text = pdf_to_text(str(dest))
        else:
            text = read_text_from_path(str(dest))
        doc_id = filename
        n = ingest_document(doc_id=doc_id, source_path=str(dest), text=text)
        counts.append({"file": filename, "chunks": n})
    return JSONResponse({"ingested": counts})


@app.post("/query")
async def query(q: str = Form(...), top_k: int = Form(5)):
    ctx = retrieve(q, top_k=top_k)
    answer = synthesize_answer(q, ctx)
    return JSONResponse({
        "query": q,
        "answer": answer,
        "contexts": [
            {"chunk_id": c.chunk_id, "score": c.score, "source_path": c.source_path, "order": c.order} for c in ctx
        ],
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("kb_rag.app:app", host=settings.host, port=settings.port, reload=False)
