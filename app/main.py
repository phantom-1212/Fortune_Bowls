from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.rag.service import RAGService
from app.routers.ingest import router as ingest_router
from app.routers.query import router as query_router


app = FastAPI(title="Knowledge-base Search Engine (RAG)")

# CORS for local dev / simple demos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend assets
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "frontend" / "templates"
STATIC_DIR = BASE_DIR / "frontend" / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def on_startup() -> None:
    settings = get_settings()
    # Initialize RAG service once and store on app state
    app.state.rag_service = RAGService(settings)


# Dependency to access the RAG service from routers
async def get_service(request: Request) -> RAGService:
    return request.app.state.rag_service


# Routers
app.include_router(ingest_router, prefix="/api", tags=["ingest"], dependencies=[])
app.include_router(query_router, prefix="/api", tags=["query"], dependencies=[])


# Root page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


# Health check
@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
