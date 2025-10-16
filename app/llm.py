from __future__ import annotations

from typing import List, Dict, Tuple
import os

from .config import settings
from .retriever import SearchResult


SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant that answers questions using only the provided context. "
    "Be concise and precise. If the answer is not contained in the context, say you don't know. "
    "Cite sources inline like [1], [2] corresponding to the provided context snippets."
)


def _build_context(snippets: List[Tuple[str, str]]) -> str:
    # snippets: list of (label, text)
    parts = []
    for label, text in snippets:
        parts.append(f"[{label}]\n{text}")
    return "\n\n".join(parts)


class LLMProvider:
    def __init__(self) -> None:
        self.provider = settings.llm_provider.lower().strip()
        self._client = None
        self._pipeline = None

        if self.provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
                self._client = OpenAI()
            except Exception as e:
                raise RuntimeError("Failed to initialize OpenAI client. Set OPENAI_API_KEY.") from e
        else:
            # transformers fallback
            try:
                from transformers import pipeline  # type: ignore
                self._pipeline = pipeline(
                    "text2text-generation", model=settings.transformer_model
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed to initialize transformers pipeline. Install transformers/torch or use OpenAI."
                ) from e

    def generate(self, question: str, results: List[SearchResult]) -> str:
        # Prepare labeled snippets up to max_context_chars
        snippets: List[Tuple[str, str]] = []
        total = 0
        for i, r in enumerate(results, start=1):
            label = str(i)
            source = r.chunk.metadata.get("source")
            page = r.chunk.metadata.get("page")
            header = f"Source: {source}"
            if page:
                header += f", p.{page}"
            snippet_text = f"{header}\n\n{r.chunk.text}"
            if total + len(snippet_text) > settings.max_context_chars:
                break
            snippets.append((label, snippet_text))
            total += len(snippet_text)

        context_block = _build_context(snippets)
        user_prompt = (
            "Using only the context, answer the question succinctly and cite sources.\n\n"
            f"Question: {question}\n\nContext:\n{context_block}"
        )

        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        else:
            # transformers
            output = self._pipeline(
                f"{SYSTEM_INSTRUCTIONS}\n\n{user_prompt}",
                max_new_tokens=400,
                do_sample=False,
            )[0]["generated_text"]
            return output
