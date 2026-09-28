# Frontend, Deployment & Documentation Implementation Plan (Plan 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a responsive web UI (Indexing, Querying, and Evaluation sections) served by the FastAPI app, a Docker/Railway deployment setup, and complete documentation.

**Architecture:** Plain HTML, CSS, and JavaScript in `app/static/`, with no build step and no framework. The FastAPI app serves `index.html` at `/` and the assets under `/static`. A single Dockerfile, used by Railway, runs `uvicorn app.main:create_app --factory` with the embedding model baked into the image. Docs are `README.md` plus `docs/*.md`.

**Tech Stack:** HTML/CSS/vanilla JS, FastAPI `StaticFiles`, Docker, Railway (`railway.json`). Builds on Plans 1 and 2, which must be complete.

**Spec:** `requirements.md` (repo root)

## Global Constraints

- The web interface has two main sections, **Indexing** and **Querying**, plus an evaluation metrics display panel.
- Indexing has a file upload area (drag and drop, multiple files, and a folder picker).
- Querying has a chat-like interface. Each answer shows the source code snippets it used.
- Responsive: two columns at 900px and wider, one column below that. No horizontal scrolling at 360px width.
- Model output and file content are rendered with `textContent` only, never `innerHTML` (XSS safety).
- Deployment target: Railway via Dockerfile. Chroma persists to `CHROMA_DIR=/data/chroma` on a Railway volume mounted at `/data`. `ANTHROPIC_API_KEY` is set as a Railway variable. The server listens on `$PORT`.
- No new Python dependencies.

## Review Focus

- A request that fails with a 4xx, 5xx, or network error shows the API's `detail` message in the UI; the page never gets stuck on "Thinking…". Covered by the `api()` helper in `app.js`, manual check in Task 1 Step 4.
- An LLM answer containing `<script>` or other HTML is displayed as text (test: `test_frontend_never_uses_innerhtml`, Task 1).
- Uploading a folder keeps relative paths such as `src/auth.py` rather than bare file names. Covered by `f.webkitRelativePath` in `app.js`, manual check in Task 1 Step 4.
- Evaluation can take minutes with the judge enabled, so the Run button is disabled while it runs and a progress message is shown. Manual check in Task 1 Step 4.
- The container starts without an API key; `/health` works and `/query` returns a clear 502. Smoke-tested in Task 2 Step 3.

---

### Task 1: Web frontend

**Files:**
- Create: `app/static/index.html`, `app/static/style.css`, `app/static/app.js`
- Modify: `app/main.py`: serve the static files
- Test: `tests/test_frontend.py`

**Interfaces:**
- Consumes: `GET /health` → `{indexed_chunks}`; `POST /index/files` (multipart `files`); `POST /query` `{question, k}` → `{answer, sources[]}`; `POST /evaluate` `{k, use_judge}` → report dict (see Plan 2 Task 3).
- Produces: `GET /` returns `index.html`; `GET /static/*` serves the assets.

- [ ] **Step 1: Write the failing tests** (`tests/test_frontend.py`)

```python
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakeLLM

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"


def test_index_page_served(tmp_path):
    client = TestClient(create_app(str(tmp_path), FakeLLM()))
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    for text in ("Indexing", "Querying", "Evaluation", 'name="viewport"'):
        assert text in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_frontend_never_uses_innerhtml():
    assert "innerHTML" not in (STATIC / "app.js").read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_frontend.py -v`
Expected: FAIL (404 on `/`, file not found for `app.js`).

- [ ] **Step 3: Create the static files**

`app/static/index.html`:
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Codebase RAG</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header class="topbar">
    <h1>Codebase RAG</h1>
    <span id="status" class="status" aria-live="polite">Loading…</span>
  </header>

  <main class="layout">
    <section class="panel" id="indexing" aria-labelledby="indexing-title">
      <h2 id="indexing-title">Indexing</h2>
      <label class="dropzone" id="dropzone" for="file-input">
        <strong>Drop code files here</strong>
        <span>or click to choose files</span>
        <input type="file" id="file-input" multiple hidden>
      </label>
      <div class="row">
        <label class="button secondary" for="folder-input">Choose folder</label>
        <input type="file" id="folder-input" webkitdirectory multiple hidden>
        <button id="index-btn" class="button" disabled>Index files</button>
      </div>
      <ul id="file-list" class="file-list"></ul>
      <p id="index-result" class="message" aria-live="polite"></p>
    </section>

    <section class="panel" id="querying" aria-labelledby="querying-title">
      <h2 id="querying-title">Querying</h2>
      <div id="chat" class="chat" aria-live="polite">
        <p class="hint">Ask a question about the indexed code, e.g. “How are passwords hashed?”</p>
      </div>
      <form id="ask-form" class="ask">
        <label class="sr-only" for="question">Question</label>
        <textarea id="question" rows="2" placeholder="Ask about the code…" required></textarea>
        <label class="k-select">Top K
          <select id="query-k"><option>3</option><option selected>5</option><option>10</option></select>
        </label>
        <button class="button" type="submit">Ask</button>
      </form>
    </section>

    <section class="panel wide" id="evaluation" aria-labelledby="evaluation-title">
      <h2 id="evaluation-title">Evaluation</h2>
      <p class="hint">Runs the 14-question ground-truth suite against the bundled sample codebase.</p>
      <div class="row">
        <label class="k-select">K
          <select id="eval-k"><option>3</option><option selected>5</option><option>10</option></select>
        </label>
        <label class="check"><input type="checkbox" id="use-judge" checked> LLM-as-judge</label>
        <button id="eval-btn" class="button">Run evaluation</button>
      </div>
      <p id="eval-status" class="message" aria-live="polite"></p>
      <div id="metrics" class="metrics"></div>
      <div class="table-wrap"><table id="eval-table" hidden>
        <thead><tr><th>Question</th><th>P@K</th><th>R@K</th><th>RR</th><th>Judge</th></tr></thead>
        <tbody></tbody>
      </table></div>
    </section>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>
```

`app/static/style.css`:
```css
:root {
  --bg: #f6f7f9; --panel: #ffffff; --text: #1d2330; --muted: #5d6677; --border: #dde1e8;
  --accent: #3b5bdb; --accent-text: #ffffff; --user: #e7ecff; --code-bg: #f1f3f6; --error: #c92a2a;
  --radius: 10px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12151b; --panel: #1b1f27; --text: #e6e9ef; --muted: #9aa3b2; --border: #2c323d;
    --accent: #748ffc; --accent-text: #0d1020; --user: #26304d; --code-bg: #11141a; --error: #ff8787;
  }
}
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 16px; border-bottom: 1px solid var(--border); background: var(--panel); }
.topbar h1 { font-size: 1.15rem; margin: 0; }
.status { color: var(--muted); font-size: .9rem; }
.layout { display: grid; grid-template-columns: 1fr; gap: 16px; padding: 16px; max-width: 1280px; margin: 0 auto; }
@media (min-width: 900px) { .layout { grid-template-columns: minmax(300px, 1fr) 2fr; } .wide { grid-column: 1 / -1; } }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; min-width: 0; }
.panel h2 { margin: 0 0 12px; font-size: 1.05rem; }
.dropzone { display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 28px 12px; border: 2px dashed var(--border); border-radius: var(--radius); cursor: pointer; text-align: center; color: var(--muted); }
.dropzone.drag, .dropzone:hover { border-color: var(--accent); color: var(--text); }
.row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-top: 12px; }
.button { display: inline-block; border: 0; border-radius: 8px; padding: 8px 14px; background: var(--accent); color: var(--accent-text); font: inherit; cursor: pointer; }
.button.secondary { background: transparent; color: var(--accent); border: 1px solid var(--accent); }
.button:disabled { opacity: .5; cursor: not-allowed; }
.file-list { list-style: none; padding: 0; margin: 10px 0 0; max-height: 160px; overflow: auto; font: 13px ui-monospace, monospace; color: var(--muted); }
.message { min-height: 1.5em; margin: 10px 0 0; color: var(--muted); }
.message.error, .bubble.error { color: var(--error); }
.chat { display: flex; flex-direction: column; gap: 10px; height: 420px; overflow-y: auto; padding: 4px; }
.bubble { padding: 10px 12px; border-radius: var(--radius); border: 1px solid var(--border); white-space: pre-wrap; overflow-wrap: anywhere; }
.bubble.user { align-self: flex-end; background: var(--user); max-width: 85%; }
.sources { margin-top: 8px; display: flex; flex-direction: column; gap: 6px; white-space: normal; }
.sources summary { cursor: pointer; font: 13px ui-monospace, monospace; }
pre { margin: 6px 0 0; padding: 10px; background: var(--code-bg); border-radius: 8px; overflow-x: auto; font: 12.5px/1.45 ui-monospace, monospace; }
.ask { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; align-items: flex-end; }
.ask textarea { flex: 1 1 240px; resize: vertical; padding: 8px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font: inherit; }
.k-select, .check { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }
select { font: inherit; padding: 4px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg); color: var(--text); }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-top: 12px; }
.metric { border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 12px; }
.metric .label { color: var(--muted); font-size: .85rem; }
.metric .value { font-size: 1.4rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.table-wrap { overflow-x: auto; margin-top: 12px; }
table { width: 100%; border-collapse: collapse; font-size: .9rem; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
.hint { color: var(--muted); margin: 0; }
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
```

`app/static/app.js`:
```js
// Vanilla UI for the Codebase RAG API. All dynamic text is inserted as text nodes, never parsed as HTML.
const $ = (sel) => document.querySelector(sel);
let selectedFiles = [];

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new Error("Network error: the API is unreachable.");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    throw new Error(`${res.status}: ${detail}`);
  }
  return body;
}

function showMessage(node, text, isError = false) {
  node.textContent = text;
  node.classList.toggle("error", isError);
}

async function refreshStatus() {
  try {
    const health = await api("/health");
    $("#status").textContent = `${health.indexed_chunks} chunks indexed`;
  } catch {
    $("#status").textContent = "API unreachable";
  }
}

// ---------- Indexing ----------
function setFiles(files) {
  selectedFiles = [...files];
  const list = $("#file-list");
  list.replaceChildren(...selectedFiles.map((f) => el("li", {}, f.webkitRelativePath || f.name)));
  $("#index-btn").disabled = selectedFiles.length === 0;
  showMessage($("#index-result"), selectedFiles.length ? `${selectedFiles.length} file(s) selected` : "");
}

$("#file-input").addEventListener("change", (e) => setFiles(e.target.files));
$("#folder-input").addEventListener("change", (e) => setFiles(e.target.files));
const dropzone = $("#dropzone");
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  setFiles(e.dataTransfer.files);
});

$("#index-btn").addEventListener("click", async () => {
  const button = $("#index-btn");
  const form = new FormData();
  for (const f of selectedFiles) form.append("files", f, f.webkitRelativePath || f.name);
  button.disabled = true;
  showMessage($("#index-result"), "Indexing…");
  try {
    const r = await api("/index/files", { method: "POST", body: form });
    let text = `Indexed ${r.files_indexed} file(s) into ${r.chunks_indexed} chunks (${r.total_chunks} total).`;
    if (r.skipped.length) text += ` Skipped: ${r.skipped.map((s) => `${s.path} (${s.reason})`).join(", ")}`;
    showMessage($("#index-result"), text);
    refreshStatus();
  } catch (err) {
    showMessage($("#index-result"), err.message, true);
  } finally {
    button.disabled = selectedFiles.length === 0;
  }
});

// ---------- Querying ----------
function renderSources(sources) {
  const box = el("div", { class: "sources" });
  for (const s of sources) {
    const summary = el("summary", {}, `${s.id}  ·  lines ${s.start_line}-${s.end_line}  ·  score ${s.score.toFixed(3)}`);
    box.append(el("details", {}, summary, el("pre", {}, el("code", {}, s.content))));
  }
  return box;
}

$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("#question").value.trim();
  if (!question) return;
  const chat = $("#chat");
  chat.querySelector(".hint")?.remove();
  chat.append(el("div", { class: "bubble user" }, question));
  const reply = el("div", { class: "bubble" }, "Thinking…");
  chat.append(reply);
  chat.scrollTop = chat.scrollHeight;
  $("#question").value = "";
  try {
    const r = await api("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k: Number($("#query-k").value) }),
    });
    reply.replaceChildren(r.answer);
    if (r.sources.length) reply.append(renderSources(r.sources));
  } catch (err) {
    reply.classList.add("error");
    reply.replaceChildren(err.message);
  }
  chat.scrollTop = chat.scrollHeight;
});

$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask-form").requestSubmit(); }
});

// ---------- Evaluation ----------
const pct = (x) => `${(x * 100).toFixed(1)}%`;

function metric(label, value) {
  return el("div", { class: "metric" }, el("div", { class: "label" }, label), el("div", { class: "value" }, value));
}

function renderReport(report) {
  const k = report.k;
  const cards = [
    metric(`Precision@${k}`, pct(report.retrieval.precision_at_k)),
    metric(`Recall@${k}`, pct(report.retrieval.recall_at_k)),
    metric("MRR", report.retrieval.mrr.toFixed(3)),
  ];
  const g = report.generation;
  if (g) {
    cards.push(
      metric("Judge overall", pct(g.overall)),
      metric("Faithfulness", `${g.faithfulness.toFixed(2)} / 5`),
      metric("Relevance", `${g.relevance.toFixed(2)} / 5`),
      metric("Correctness", `${g.correctness.toFixed(2)} / 5`),
    );
  }
  $("#metrics").replaceChildren(...cards);

  const rows = report.results.map((row) => {
    const judge = row.judge ? pct(row.judge.overall) : row.error ? "error" : "—";
    const judgeCell = el("td", { class: "num" }, judge);
    if (row.judge) judgeCell.title = row.judge.reasoning;
    if (row.error) judgeCell.title = row.error;
    return el("tr", {},
      el("td", {}, row.question),
      el("td", { class: "num" }, pct(row.precision_at_k)),
      el("td", { class: "num" }, pct(row.recall_at_k)),
      el("td", { class: "num" }, row.reciprocal_rank.toFixed(2)),
      judgeCell);
  });
  $("#eval-table tbody").replaceChildren(...rows);
  $("#eval-table").hidden = false;
}

$("#eval-btn").addEventListener("click", async () => {
  const button = $("#eval-btn");
  const useJudge = $("#use-judge").checked;
  button.disabled = true;
  showMessage($("#eval-status"), useJudge ? "Running… the LLM judge can take a few minutes." : "Running…");
  try {
    const report = await api("/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ k: Number($("#eval-k").value), use_judge: useJudge }),
    });
    renderReport(report);
    const g = report.generation;
    const errors = report.results.filter((r) => r.error).length;
    let text = `Evaluated ${report.num_examples} questions.`;
    if (useJudge && !g) text += " LLM judge unavailable (see row tooltips); retrieval metrics only.";
    else if (errors) text += ` ${errors} judge error(s).`;
    showMessage($("#eval-status"), text);
  } catch (err) {
    showMessage($("#eval-status"), err.message, true);
  } finally {
    button.disabled = false;
  }
});

refreshStatus();
```

Modify `app/main.py`: add these imports:
```python
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
```
Add the module constant `STATIC_DIR = Path(__file__).resolve().parent / "static"`, and inside `create_app`, before `return app`:
```python
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")
```

- [ ] **Step 4: Run tests, then check manually in the browser**

Run: `uv run pytest -v`. Expected: all pass.

Then run `CHROMA_DIR=$(mktemp -d) uv run uvicorn app.main:create_app --factory --port 8765` and open http://localhost:8765 . Check:
1. Choosing the folder `eval/sample_repo` lists paths like `sample_repo/auth.py`, and indexing reports the chunk count.
2. Asking a question without an API key shows a red 502 error bubble, not a stuck "Thinking…".
3. Evaluation with the judge unchecked fills in the metric cards and the table.
4. At a narrow window width (about 360px) the layout is a single column with no horizontal page scroll.

- [ ] **Step 5: Commit**

```bash
git add app/static app/main.py tests/test_frontend.py
git commit -m "feat: add responsive web UI for indexing, querying and evaluation"
```

---

### Task 2: Docker + Railway deployment config

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `railway.json`

**Interfaces:**
- Consumes: `uv.lock`, `app/`, `eval/`, and the factory `app.main:create_app`.
- Produces: an image that listens on `$PORT` (default 8000), with `CHROMA_DIR=/data/chroma` and a health check at `/health`.

- [ ] **Step 1: Create the files**

`Dockerfile`:
```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY eval ./eval

# Bake the all-MiniLM-L6-v2 ONNX embedding model into the image so the first request is fast.
RUN uv run --no-sync python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warmup'])"

ENV CHROMA_DIR=/data/chroma PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uv run --no-sync uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT}"]
```

`.dockerignore`:
```
.git
.venv
data
__pycache__
.pytest_cache
tests
docs
```

`railway.json`:
```json
{
  "$schema": "https://railway.com/railway.schema.json",
  "build": { "builder": "DOCKERFILE", "dockerfilePath": "Dockerfile" },
  "deploy": { "healthcheckPath": "/health", "healthcheckTimeout": 120, "restartPolicyType": "ON_FAILURE" }
}
```

- [ ] **Step 2: Build the image**

Run: `docker build -t rag-evaluation .`
Expected: the build succeeds. (If the Docker daemon isn't running, report that and skip to Step 4; don't fake the result.)

- [ ] **Step 3: Smoke-test the container without an API key**

```bash
docker run -d --rm -p 8766:8000 --name rag-smoke rag-evaluation
sleep 8
curl -s localhost:8766/health                                   # {"status":"ok","indexed_chunks":0}
curl -s -F "files=@eval/sample_repo/auth.py;filename=auth.py" localhost:8766/index/files
curl -s -o /dev/null -w "%{http_code}\n" -H 'Content-Type: application/json' \
  -d '{"question":"how are passwords hashed?"}' localhost:8766/query   # 502 (no key)
curl -s -H 'Content-Type: application/json' -d '{"use_judge":false}' localhost:8766/evaluate | head -c 300
docker stop rag-smoke
```
Expected: the outputs shown in the comments. The evaluate call returns JSON containing `"retrieval"`.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore railway.json
git commit -m "build: add Dockerfile and Railway deployment config"
```

---

### Task 3: README and complete documentation

**Files:**
- Create: `README.md`, `docs/architecture.md`, `docs/api.md`, `docs/evaluation.md`, `docs/deployment.md`

**Interfaces:**
- Consumes: the final code in `app/`, `eval/`, `Dockerfile`, and `railway.json`. Every endpoint, env var, and number in the docs must match the code. Read the code; don't copy from this plan blindly.

- [ ] **Step 1: Write `README.md`** with these sections, in this order:
  1. Title plus a one-paragraph summary: a codebase Q&A RAG system with evaluation, built for "Lab 04: RAG System with Evaluation".
  2. **Features**: code-aware chunking (Python by function, class, and method via `ast`; other languages by 40-line windows with 10-line overlap); persistent ChromaDB; free local embeddings (all-MiniLM-L6-v2 via ONNX); grounded answers with citations from Claude (`claude-opus-5`); Precision@K, Recall@K, MRR, and an LLM-as-judge (faithfulness, relevance, correctness); a web UI.
  3. **Architecture**: the INDEXING, QUERYING, and EVALUATION ASCII diagram from `requirements.md`, adapted to name the real modules (`chunker.py` → `store.py` → `rag.py` → `llm.py`; `evaluation.py` + `metrics.py`).
  4. **Quickstart**: `uv sync`; `export ANTHROPIC_API_KEY=...`; `uv run uvicorn app.main:create_app --factory --reload`; open http://localhost:8000; note that the first run downloads the ~80 MB embedding model.
  5. **Configuration** table: `ANTHROPIC_API_KEY` (required for /query and the judge), `LLM_MODEL` (default `claude-opus-5`), `CHROMA_DIR` (default `./data/chroma`), `PORT` (Docker, default 8000).
  6. **API summary**: one line per endpoint (`GET /`, `GET /health`, `POST /index/files`, `POST /query`, `POST /evaluate`) with a `curl` example each, plus a link to `docs/api.md` and the auto-generated `/docs` (Swagger).
  7. **Evaluation**: how to run it (UI button or `curl -X POST localhost:8000/evaluate`), what each metric means in one line, and the dataset location. Include the **measured** retrieval numbers from running `curl -s -X POST localhost:8000/evaluate -H 'Content-Type: application/json' -d '{"use_judge": false}'` locally at k=5. Run it and paste the real numbers; never invent them. State that the judge scores need an API key.
  8. **Testing**: `uv run pytest`; tests never call the real API.
  9. **Deployment**: a short summary plus a link to `docs/deployment.md`. Leave a clearly marked `Live URL:` line saying the URL is added after deploying (the implementer cannot deploy without Railway credentials).
  10. **Project structure** tree.
  11. **Design decisions & limitations**: ChromaDB's built-in ONNX MiniLM instead of the sentence-transformers package (same model, no torch, smaller image); re-indexing replaces a file's chunks; class attributes between methods fall into `<module>`; path-level ground truth for non-Python files; extension challenges (hybrid BM25, reranking, caching, multiple codebases) not implemented, with a one-line note on where each would plug in (`store.search`, `rag.retrieve`, `VectorStore` collection per repo).
  12. **License**: Apache-2.0 (see `LICENSE`).

- [ ] **Step 2: Write `docs/architecture.md`**: the responsibility and public interface of each module (`chunker`, `store`, `llm`, `rag`, `metrics`, `evaluation`, `main`); the step-by-step data flow for indexing, querying, and evaluation; the chunk id scheme (`path::symbol`, `<module>`, `Class.method`, `L{start}-{end}`, `#n` suffixes); the two Chroma collections (`code` and `eval`); the prompts used (summarize `ANSWER_SYSTEM` and `JUDGE_SYSTEM`); error handling (`LLMError` → 502; judge errors per row; skipped uploads).

- [ ] **Step 3: Write `docs/api.md`**: for each endpoint, the method and path, request format (multipart or JSON schema with limits: 1 MB per file, question 1–2000 chars, k 1–20), a full example response copied from a real local call (the `/query` example may show the 502 error body if there's no key; say so), and the status codes (200, 422, 502).

- [ ] **Step 4: Write `docs/evaluation.md`**: the dataset format; the relevance matching rule (exact id or path); formulas for Precision@K, Recall@K, and MRR with a worked example; LLM-as-judge criteria, the 1–5 scale, and `overall = sum / 15`; the judge's structured JSON output; how to add examples (and that `tests/test_dataset.py` validates the ground truth); how to interpret the results.

- [ ] **Step 5: Write `docs/deployment.md`**: Railway steps (a new project from the GitHub repo, which auto-detects `railway.json`/Dockerfile; add a volume mounted at `/data`; set `ANTHROPIC_API_KEY`; generate a domain; the health check is `/health`); a local Docker run (`docker build`, then `docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... -v rag-data:/data rag-evaluation`); a Render alternative (Docker runtime, disk at `/data`); cost and latency notes (the judge makes 2 Claude calls per example, 28 per full run).

- [ ] **Step 6: Verify the docs against the code**

Run: `grep -rn "claude-opus-5\|CHROMA_DIR\|LLM_MODEL\|/index/files\|/evaluate" README.md docs/*.md | head -40` and spot-check each claim against `app/`. Confirm every relative link resolves: `ls docs/architecture.md docs/api.md docs/evaluation.md docs/deployment.md LICENSE`.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/architecture.md docs/api.md docs/evaluation.md docs/deployment.md
git commit -m "docs: add README and complete project documentation"
```
