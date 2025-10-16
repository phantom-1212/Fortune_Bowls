from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, UploadFile

from app.config import get_settings
from app.rag.service import RAGService

router = APIRouter()


def _safe_filename(name: str) -> str:
    # Basic sanitization and uniqueness
    base = Path(name).name.replace("..", "_").replace("/", "_")
    return base


@router.post("/ingest")
async def ingest_files(
    files: List[UploadFile] = File(...),
) -> dict:
    settings = get_settings()
    saved: List[str] = []
    for f in files:
        filename = _safe_filename(f.filename or "upload")
        out_path = settings.uploads_dir / filename
        # If exists, add numeric suffix
        counter = 1
        while out_path.exists():
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            out_path = settings.uploads_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        with out_path.open("wb") as w:
            w.write(await f.read())
        saved.append(out_path.name)

    # Defer to app.state.rag_service
    from fastapi import Request

    # Access request to get app state
    # Workaround: Create a dummy request dependency is not provided; use global app import
    from app.main import app  # type: ignore

    service: RAGService = app.state.rag_service
    stats = service.ingest_files(saved)
    return {"saved_files": saved, **stats}
