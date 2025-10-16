from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    source: str = Field(..., description="Document source or filename")
    page: Optional[int] = Field(None, description="Page number if PDF")
    chunk_index: Optional[int] = Field(None, description="Chunk index within the source/page")
    score: float = Field(..., description="Similarity score (cosine)")
    text: str = Field(..., description="Chunk text")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    include_sources: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceChunk]] = None


class IngestResponse(BaseModel):
    num_documents: int
    num_chunks: int
    documents: List[Dict[str, Any]]
