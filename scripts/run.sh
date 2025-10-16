#!/usr/bin/env bash
set -euo pipefail
export STORAGE_DIR=${STORAGE_DIR:-/workspace/storage}
export MODEL_PROVIDER=${MODEL_PROVIDER:-openai}
export OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}
export EMBEDDING_MODEL=${EMBEDDING_MODEL:-text-embedding-3-small}
uvicorn rag.app:app --host 0.0.0.0 --port 8000 --reload
