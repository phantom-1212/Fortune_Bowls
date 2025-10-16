from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class Settings:
    data_dir: str = os.getenv("DATA_DIR", "data")
    index_dir: str = os.getenv("INDEX_DIR", "data/index")
    uploads_dir: str = os.getenv("UPLOADS_DIR", "data/uploads")

    # Embeddings
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # LLM provider: 'openai' | 'transformers'
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")

    # OpenAI
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Local transformers fallback
    transformer_model: str = os.getenv("TRANSFORMER_MODEL", "google/flan-t5-base")

    # RAG params
    top_k: int = int(os.getenv("TOP_K", "5"))
    chunk_size_chars: int = int(os.getenv("CHUNK_SIZE_CHARS", "1000"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))

    # Persistence
    persist_index: bool = os.getenv("PERSIST_INDEX", "true").lower() == "true"

    def ensure_dirs(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        Path(self.uploads_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
