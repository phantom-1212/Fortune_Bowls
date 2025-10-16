# Knowledge-base Search & Synthesis (RAG)

Backend service to ingest PDFs/text documents, search semantically with embeddings, and synthesize answers via an LLM with citations. Includes a minimal web UI and a CLI for bulk ingestion.

## Features

- Document ingestion for PDF and text files with smart chunking
- Embeddings via `sentence-transformers` and FAISS vector store (cosine similarity)
- RAG: top-k retrieval + LLM synthesis with inline citations
- Providers: OpenAI (default) or local `transformers` fallback (FLAN-T5)
- REST API (FastAPI) + simple HTML UI
- CLI for bulk ingestion from a directory

## Quickstart

1) Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure LLM provider (OpenAI recommended):

```bash
export OPENAI_API_KEY=YOUR_KEY
# optional overrides
export LLM_PROVIDER=openai           # or: transformers
export OPENAI_MODEL=gpt-4o-mini
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

3) Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/` for the minimal UI.

4) Ingest bulk documents (optional):

```bash
python scripts/ingest.py path/to/docs
```

5) Query via API:

```bash
curl -sS -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the policy on returns?", "top_k": 5, "include_sources": true}' | jq
```

## API

- `POST /ingest/files`: multipart form upload of one or more files (pdf, txt, md)
- `POST /query`: body `{ question: string, top_k?: number, include_sources?: boolean }`

## How it works

- Ingestion extracts text (by page for PDFs), splits into overlapping chunks, embeds with Sentence-Transformers, normalizes vectors and stores in FAISS (IP == cosine).
- Queries embed the question, perform ANN search, and pass top-k chunks to the LLM with clear instructions to answer succinctly with citations like `[1]`, `[2]`.

## Notes

- Local `transformers` fallback uses `google/flan-t5-base` and will download model weights on first run. For best quality, use OpenAI.
- Index and metadata persist under `data/index/`. Uploads saved in `data/uploads/`.

## Docker

Build and run with Docker:

```bash
docker build -t rag-kb .
docker run --rm -it -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/data:/app/data \
  rag-kb
```

Then open `http://localhost:8000/`.

## Demo video

Record a short screencast showing:
- Starting the server
- Ingesting a sample PDF
- Asking a question and seeing the answer + citations

## License
MIT
