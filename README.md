# Knowledge-base Search & Synthesis (RAG)

Backend service and minimal UI to ingest documents (text/PDF), retrieve relevant chunks via embeddings, and synthesize concise answers using an LLM or a heuristic fallback.

## Features
- Upload and index `.txt` and `.pdf` files
- Embeddings via local `sentence-transformers` or OpenAI (optional)
- Vector search with FAISS (if available) or NumPy fallback, plus SQLite metadata
- REST API with FastAPI
- Minimal HTML UI for upload and questions
- CLI for ingestion and querying

## Quick Start

### 1) Install Python packages
If virtualenv creation fails in your environment, install user-local:

```bash
pip3 install --user fastapi==0.115.0 uvicorn[standard]==0.30.6 httpx==0.27.2 python-multipart==0.0.17 numpy==2.1.2 pdfplumber==0.11.4
# Optional for local embeddings and FAISS
pip3 install --user sentence-transformers==3.1.1
pip3 install --user faiss-cpu  # version varies by platform; optional
```

### 2) Run the API

```bash
python3 -m uvicorn kb_rag.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/` and use the UI to upload documents and ask questions.

### 3) CLI Usage

```bash
python3 -m kb_rag.cli ingest path/to/file1.txt path/to/file2.pdf
python3 -m kb_rag.cli query "What is FastAPI?" --top-k 5
```

## Configuration
Configure via environment variables:

- `EMBEDDING_PROVIDER`: `local` (default) or `openai`
- `EMBEDDING_MODEL`: default `sentence-transformers/all-MiniLM-L6-v2`
- `LLM_PROVIDER`: `openai`, `ollama`, or `fallback` (uses heuristic summarization)
- `OPENAI_API_KEY`: required if using OpenAI
- `OPENAI_MODEL`: default `gpt-4o-mini`
- `OLLAMA_HOST`: e.g. `http://localhost:11434`
- `OLLAMA_MODEL`: default `llama3.1:8b-instruct-q4_K_M`
- `CHUNK_SIZE`: default `1000`
- `CHUNK_OVERLAP`: default `200`

## Notes
- If FAISS is not installed or available for your platform, the system uses a NumPy `.npz` store as a fallback. Retrieval remains functional though not as efficient.
- PDF extraction uses `pdfplumber`. For scanned PDFs, OCR is not included by default.

## Prompting Guidance
Use a concise instruction for synthesis, for example:

> Using these documents, answer the user's question succinctly.

## Deliverables
- Backend API and minimal web UI
- CLI
- This README
- Demo: run locally and record a short video showing ingestion and querying

## License
MIT
