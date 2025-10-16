from __future__ import annotations

from typing import Iterable, List

from .config import settings

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None


def embed_texts(texts: List[str]) -> List[List[float]]:
    if settings.model_provider == "openai":
        return _embed_openai(texts)
    elif settings.model_provider == "ollama":
        return _embed_ollama(texts)
    else:
        raise ValueError(f"Unknown model provider: {settings.model_provider}")


def _embed_openai(texts: List[str]) -> List[List[float]]:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in resp.data]


def _embed_ollama(texts: List[str]) -> List[List[float]]:
    import requests

    # Ollama embeddings endpoint
    out: List[List[float]] = []
    for t in texts:
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": settings.ollama_embedding_model, "prompt": t},
            timeout=60,
        )
        r.raise_for_status()
        out.append(r.json()["embedding"])  # type: ignore
    return out
