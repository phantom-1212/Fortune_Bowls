import os
from pathlib import Path
from dataclasses import dataclass


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Ensure runtime directories exist
for d in (DATA_DIR, UPLOADS_DIR, VECTOR_STORE_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    # Embeddings
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")  # "openai" or "local"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")  # "openai", "ollama", "fallback"
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    ollama_host: str | None = os.getenv("OLLAMA_HOST")  # e.g. http://localhost:11434
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Vector Store
    prefer_faiss: bool = os.getenv("PREFER_FAISS", "true").lower() in {"1", "true", "yes"}
    index_path: Path = VECTOR_STORE_DIR / "index.faiss"
    fallback_index_path: Path = VECTOR_STORE_DIR / "index.npz"
    sqlite_path: Path = VECTOR_STORE_DIR / "metadata.db"

    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))


settings = Settings()
