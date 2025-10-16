import os
from typing import Optional


class Settings:
    """Application settings loaded from environment variables with sensible defaults."""

    def __init__(self) -> None:
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.storage_dir: str = os.getenv("STORAGE_DIR", "/workspace/storage")
        self.upload_dir: str = os.path.join(self.storage_dir, "uploads")
        self.index_dir: str = os.path.join(self.storage_dir, "index")
        self.model_provider: str = os.getenv("MODEL_PROVIDER", "openai")  # or "ollama"
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.max_chunk_tokens: int = int(os.getenv("MAX_CHUNK_TOKENS", "500"))
        self.chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
        self.top_k: int = int(os.getenv("TOP_K", "5"))


settings = Settings()
