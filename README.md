# Knowledge-base Search Engine (RAG)

This repository contains a FastAPI-based Retrieval-Augmented Generation (RAG) service that indexes documents (TXT/PDF/MD), retrieves relevant chunks via FAISS, and synthesizes succinct answers using an LLM (OpenAI or Ollama). A minimal frontend is included.

## What’s included
- Backend API: `FastAPI` with endpoints for ingestion and query
- Document parsing: `.txt`, `.md`, `.pdf` using `pypdf`
- Vector store: `FAISS` (cosine similarity via normalized inner product) with on-disk persistence
- Embeddings: OpenAI or Ollama
- LLM synthesis: OpenAI Chat Completions or Ollama `generate`
- Minimal frontend: upload and query UI
- Dockerfile for containerized deployment

## Quickstart

1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure environment

```bash
cp .env.example .env
export OPENAI_API_KEY=...   # if using OpenAI
# or set MODEL_PROVIDER=ollama and ensure Ollama is running locally
```

3) Run the API server

```bash
uvicorn rag.app:app --host 0.0.0.0 --port 8000 --reload
```

4) Open the UI

- Visit `http://localhost:8000/ui` to upload documents and ask questions
- Health check at `http://localhost:8000/health`

## API
- `POST /ingest` (multipart) → `{ files: string[], chunks_indexed: number }`
- `POST /query` (JSON: `{ query: string, top_k?: number }`) → `{ answer, sources[], matched_chunks }`
- `GET /health` → `{ status: "ok" }`

## Docker

```bash
docker build -t rag-demo .
docker run -it --rm -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY rag-demo
```

For Ollama:

```bash
docker run -it --rm -p 8000:8000 \
  -e MODEL_PROVIDER=ollama -e OLLAMA_MODEL=llama3.1:8b rag-demo
```

## Notes
- Keep the embedding model consistent across ingestions; otherwise, you’ll need a new index.
- For OCR/scanned PDFs, integrate an OCR step (not included).
