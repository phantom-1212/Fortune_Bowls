from typing import List, Optional
from pydantic import BaseModel


class IngestResponse(BaseModel):
    files: List[str]
    chunks_indexed: int


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class QueryAnswer(BaseModel):
    answer: str
    sources: List[str]
    matched_chunks: int
