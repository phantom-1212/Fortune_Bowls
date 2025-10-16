from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    # Data & persistence
    data_dir: Path = Path("data")
    faiss_index_path: Path = data_dir / "index.faiss"
    docstore_path: Path = data_dir / "chunks.jsonl"
    uploads_dir: Path = data_dir / "uploads"

    # Embeddings
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Retrieval
    top_k_default: int = 5
    top_k_max: int = 20

    # LLM
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
