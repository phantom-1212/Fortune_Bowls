# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY rag ./rag
COPY frontend ./frontend
COPY scripts ./scripts

ENV STORAGE_DIR=/data
RUN mkdir -p /data /data/uploads /data/index && chmod -R 777 /data

EXPOSE 8000

CMD ["bash", "/app/scripts/run.sh"]
