# Knowledge-base Search Engine (RAG)

A minimal Retrieval-Augmented Generation (RAG) reference implementation.
- Backend: FastAPI
- Embeddings: `fastembed` with `BAAI/bge-small-en-v1.5`
- Vector store: FAISS (inner product, normalized vectors)
- Ingestion: PDF/TXT/MD with chunking
- LLM: OpenAI for synthesis
- UI: Minimal HTML page for upload and query

## Features
- Upload multiple documents, ingest them into a FAISS index
- Ask questions; retrieve top-k chunks and synthesize an answer
- Persisted index and docstore under `data/`

## Quickstart

### 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set environment variables (or `.env`) for OpenAI if you want synthesis:
```bash
export OPENAI_API_KEY=sk-...
# Optional overrides
export EMBEDDING_MODEL_NAME="BAAI/bge-small-en-v1.5"
export OPENAI_MODEL="gpt-4o-mini"
```

### 2. Run
```bash
./scripts/dev.sh
# App at http://localhost:8000
```

### 3. Demo
- Open the root page.
- Upload `examples/example.txt`.
- Ask: "What does this system do?"

### API

- POST `/api/ingest`
  - Form-data: `files`: multiple file uploads (pdf, txt, md)
  - Response: `{ saved_files: [...], chunks_added, files_count }`

- POST `/api/query`
  - JSON: `{ "question": "...", "top_k": 5 }`
  - Response: `{ "answer": "...", "sources": [ { source_name, source_path, score, chunk_index }, ...] }`

## Implementation Notes
- Text extraction uses `pypdf` for PDFs and UTF-8 read for text/markdown.
- Chunking uses simple paragraph-aware splitter with overlap to preserve context.
- Embeddings are L2-normalized; FAISS `IndexFlatIP` computes cosine similarity as inner product.
- If OpenAI is unavailable, the system falls back to showing top passages.

## Docker
```bash
docker build -t rag-app .
docker run -it --rm -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY rag-app
```

## License
MIT
