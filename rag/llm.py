from __future__ import annotations

from typing import List

from .config import settings

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None


SYSTEM_PROMPT = (
    "You are a helpful assistant. Using the provided context chunks, "
    "answer the user's question succinctly. If the answer is not present "
    "in the context, say you don't know."
)


def synthesize_answer(question: str, context_chunks: List[str]) -> str:
    if settings.model_provider == "openai":
        return _synthesize_openai(question, context_chunks)
    elif settings.model_provider == "ollama":
        return _synthesize_ollama(question, context_chunks)
    else:
        raise ValueError(f"Unknown model provider: {settings.model_provider}")


def _build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {question}\n"
        f"Answer:"
    )


def _synthesize_openai(question: str, context_chunks: List[str]) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")
    client = OpenAI(api_key=settings.openai_api_key)
    prompt = _build_prompt(question, context_chunks)
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    return resp.choices[0].message.content or ""


def _synthesize_ollama(question: str, context_chunks: List[str]) -> str:
    import requests

    prompt = _build_prompt(question, context_chunks)
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")
