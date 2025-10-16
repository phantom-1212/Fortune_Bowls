# Knowledge-base Search Engine (RAG)

Backend: FastAPI with FAISS vector search and LLM synthesis (OpenAI or Ollama)

## Features
- Upload .txt / .md / .pdf documents and index them
- FAISS cosine-similarity search over embeddings
- LLM synthesis using retrieved chunks with simple, grounded prompt
- Minimal web UI for ingestion and querying
- Persistent index on disk

## Quickstart

1) Python setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set OPENAI_API_KEY in your environment if using OpenAI
export OPENAI_API_KEY=...  # or set in shell profile
```

2) Run API

```bash
uvicorn rag.app:app --host 0.0.0.0 --port 8000 --reload
```

3) Open UI

Visit http://localhost:8000/ and upload documents, then ask questions.

## Environment variables
- `MODEL_PROVIDER`: `openai` or `ollama` (default `openai`)
- `OPENAI_API_KEY`: OpenAI API key for embeddings and chat
- `OPENAI_MODEL`: chat model (default `gpt-4o-mini`)
- `EMBEDDING_MODEL`: embedding model (default `text-embedding-3-small`)
- `OLLAMA_MODEL`: ollama local model (default `llama3.1:8b`)
- `STORAGE_DIR`: path to storage (uploads + index)

## API
- `POST /ingest`: multipart form with `files`[] . Returns number of chunks indexed
- `POST /query`: `{ "query": "...", "top_k": 5 }` → `{ answer, sources[], matched_chunks }`
- `GET /health`: health check

## Docker

```bash
docker build -t rag-demo .
docker run -it --rm -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY rag-demo
```

For Ollama provider, ensure Ollama is running locally:

```bash
MODEL_PROVIDER=ollama OLLAMA_MODEL=llama3.1:8b docker run -it --rm -p 8000:8000 \
  -e MODEL_PROVIDER=ollama -e OLLAMA_MODEL=llama3.1:8b rag-demo
```

## Notes
- Mixing different embedding models in the same index is not supported; keep it consistent.
- PDF parsing uses `pypdf` which may not extract images or scanned text (use OCR if needed).
