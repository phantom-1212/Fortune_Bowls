# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY README.md .

# Create non-root user
RUN useradd -ms /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV LLM_PROVIDER=openai \
    OPENAI_MODEL=gpt-4o-mini \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    DATA_DIR=data \
    INDEX_DIR=data/index \
    UPLOADS_DIR=data/uploads

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
