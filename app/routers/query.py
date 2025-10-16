from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings
from app.rag.service import RAGService

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list


@router.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest) -> QueryResponse:
    from app.main import app  # type: ignore

    service: RAGService = app.state.rag_service
    res = service.answer(req.question, (req.top_k or 0))
    return QueryResponse(**res)
