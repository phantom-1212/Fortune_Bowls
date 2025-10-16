from __future__ import annotations

from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .schemas import QueryRequest, QueryResponse, IngestResponse, SourceChunk
from .ingestion import extract_chunks_from_path, extract_chunks_from_upload, save_upload_to_disk
from .retriever import VectorStore
from .llm import LLMProvider

app = FastAPI(title="Knowledge-base Search & Synthesis (RAG)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global components
vector_store = VectorStore(settings.index_dir, settings.embedding_model)
llm = LLMProvider()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(files: List[UploadFile] = File(...)):
    total_chunks = 0
    documents_info = []
    for f in files:
        content = await f.read()
        # persist upload
        saved_path = save_upload_to_disk(f.filename, content)
        chunks = extract_chunks_from_upload(f.filename, content, f.content_type)
        added, dim = vector_store.add_chunks(chunks)
        total_chunks += added
        documents_info.append({
            "filename": f.filename,
            "path": str(saved_path),
            "num_chunks": added,
            "embedding_dim": dim,
        })
    return IngestResponse(num_documents=len(files), num_chunks=total_chunks, documents=documents_info)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    top_k = req.top_k or settings.top_k
    results = vector_store.search(req.question, top_k)
    answer = llm.generate(req.question, results)

    sources = None
    if req.include_sources:
        sources = [
            SourceChunk(
                source=r.chunk.metadata.get("source"),
                page=r.chunk.metadata.get("page"),
                chunk_index=r.chunk.metadata.get("chunk_index"),
                score=r.score,
                text=r.chunk.text,
                metadata=r.chunk.metadata,
            )
            for r in results
        ]
    return QueryResponse(answer=answer, sources=sources)


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(
        content=f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>RAG: Knowledge-base Search</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, sans-serif; margin: 2rem; }}
    .card {{ max-width: 900px; margin: 0 auto; border: 1px solid #e0e0e0; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
    h1 {{ font-size: 1.5rem; margin: 0 0 1rem; }}
    h2 {{ font-size: 1.1rem; margin-top: 1.5rem; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
    textarea {{ width: 100%; min-height: 120px; }}
    pre {{ white-space: pre-wrap; background: #fafafa; padding: 1rem; border-radius: 8px; border: 1px solid #eee; }}
    .sources {{ font-size: 0.9rem; color: #444; }}
    .src {{ margin-bottom: 0.5rem; }}
    .muted {{ color: #666; }}
    input[type='number'] {{ width: 80px; }}
    .btn {{ background: #111827; color: white; border: none; padding: 0.6rem 1rem; border-radius: 8px; cursor: pointer; }}
    .btn:disabled {{ opacity: 0.6; cursor: not-allowed; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>Knowledge-base Search & Synthesis</h1>

    <h2>1) Ingest documents (PDF or text)</h2>
    <form id=\"ingestForm\">
      <input type=\"file\" id=\"files\" multiple accept=\".pdf,.txt,.md\" />
      <button type=\"submit\" class=\"btn\">Upload & Embed</button>
      <span id=\"ingestStatus\" class=\"muted\"></span>
    </form>

    <h2>2) Ask a question</h2>
    <form id=\"queryForm\">
      <textarea id=\"question\" placeholder=\"Ask about your documents...\"></textarea>
      <div>
        <label>top_k: <input type=\"number\" id=\"topk\" value=\"5\" min=\"1\" max=\"20\" /></label>
        <label><input type=\"checkbox\" id=\"incsrc\" checked /> include sources</label>
      </div>
      <button type=\"submit\" class=\"btn\">Ask</button>
    </form>

    <h2>Answer</h2>
    <div id=\"answer\" class=\"muted\">No answer yet.</div>

    <h2>Sources</h2>
    <div id=\"sources\" class=\"sources\"></div>

    <p class=\"muted\">Backend health: <code id=\"health\">checking...</code></p>
  </div>

<script>
async function checkHealth() {{
  try {{
    const res = await fetch('/health');
    const j = await res.json();
    document.getElementById('health').textContent = j.status;
  }} catch (e) {{ document.getElementById('health').textContent = 'error'; }}
}}
checkHealth();

const ingestForm = document.getElementById('ingestForm');
const queryForm = document.getElementById('queryForm');

ingestForm.addEventListener('submit', async (e) => {{
  e.preventDefault();
  const status = document.getElementById('ingestStatus');
  const files = document.getElementById('files').files;
  if (!files.length) {{ status.textContent = 'Pick files first.'; return; }}
  status.textContent = 'Uploading...';
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  const res = await fetch('/ingest/files', {{ method: 'POST', body: fd }});
  const j = await res.json();
  status.textContent = `Ingested ${j.num_documents} docs, ${j.num_chunks} chunks.`;
}});

queryForm.addEventListener('submit', async (e) => {{
  e.preventDefault();
  const q = document.getElementById('question').value.trim();
  const topk = parseInt(document.getElementById('topk').value) || 5;
  const incsrc = document.getElementById('incsrc').checked;
  if (!q) return;
  document.getElementById('answer').textContent = 'Thinking...';
  document.getElementById('sources').innerHTML = '';
  const res = await fetch('/query', {{
    method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ question: q, top_k: topk, include_sources: incsrc }})
  }});
  const j = await res.json();
  document.getElementById('answer').innerHTML = `<pre>${j.answer}</pre>`;
  if (j.sources) {{
    const cont = document.getElementById('sources');
    for (const s of j.sources) {{
      const div = document.createElement('div');
      const loc = s.page ? `${{s.source}} (p.${{s.page}})` : s.source;
      div.className = 'src';
      div.innerHTML = `<strong>[${{s.chunk_index}}] ${{loc}}</strong> · score=${{s.score.toFixed(3)}}<br/><pre>${{s.text}}</pre>`;
      cont.appendChild(div);
    }}
  }}
}});
</script>
</body>
</html>
        """
    )
