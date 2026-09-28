# Codebase RAG System

A retrieval-augmented Q&A system for source code: upload files, ask questions
about them, and get answers grounded in the actual code with citations. Built
for "Lab 04: RAG System with Evaluation," it also ships a full evaluation
framework — retrieval metrics (Precision@K, Recall@K, MRR) and an
LLM-as-judge for generation quality — against a bundled 14-question dataset.

## Features

- **Code-aware chunking** — Python files are split by function, class, and
  method using the `ast` module (with a `<module>` chunk for top-level code
  between them); other languages fall back to 40-line windows with 10-line
  overlap.
- **Persistent vector store** — ChromaDB, backed by local disk, so indexed
  code survives restarts.
- **Free local embeddings** — `all-MiniLM-L6-v2` via ONNX (ChromaDB's default
  embedding function), no API key or GPU required.
- **Grounded answers with citations** — Claude (`claude-opus-5`) answers
  using only the retrieved snippets and cites them by chunk id.
- **Retrieval metrics** — Precision@K, Recall@K, and MRR (Mean Reciprocal
  Rank) computed against hand-labeled ground truth.
- **LLM-as-judge** — Claude scores each generated answer on faithfulness,
  relevance, and correctness (1-5 each).
- **Web UI** — a single-page interface for indexing files, asking questions,
  and running the evaluation suite.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Codebase RAG System                       │
├─────────────────────────────────────────────────────────────┤
│                                                                │
│  INDEXING                                                     │
│  ────────                                                     │
│  Code Files → chunker.py → store.py (embed + persist)        │
│                                                                │
│  QUERYING                                                     │
│  ────────                                                     │
│  Question → rag.py (retrieve via store.py) → llm.py (answer) │
│                                                                │
│  EVALUATION                                                   │
│  ──────────                                                   │
│  Dataset → evaluation.py → rag.py + metrics.py + llm.py      │
│            (retrieval metrics + LLM-as-judge scores)          │
│                                                                │
└─────────────────────────────────────────────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for module responsibilities,
data flow, and the chunk id scheme.

## Quickstart

```bash
uv sync
export ANTHROPIC_API_KEY=sk-...          # required for /query and the judge
uv run uvicorn app.main:create_app --factory --reload
```

Open http://localhost:8000. The first run downloads the ~80 MB
`all-MiniLM-L6-v2` embedding model (cached afterwards).

## Configuration

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes, for `/query` and judge scoring | — | Claude API key (`anthropic.Anthropic()` reads it from the environment) |
| `LLM_MODEL` | No | `claude-opus-5` | Model used for answers and judging |
| `CHROMA_DIR` | No | `./data/chroma` | Directory for the persistent ChromaDB store |
| `PORT` | No (Docker only) | `8000` | Port the container listens on |

## API summary

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Liveness + indexed chunk count |
| `POST` | `/index/files` | Upload and index code files |
| `POST` | `/query` | Ask a question about indexed code |
| `POST` | `/evaluate` | Run the retrieval + generation evaluation suite |

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/index/files \
  -F "files=@app/chunker.py"

curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How are passwords hashed?", "k": 5}'

curl -X POST http://localhost:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"use_judge": false}'
```

Full request/response schemas, limits, and status codes: [`docs/api.md`](docs/api.md).
The auto-generated Swagger UI is available at `/docs` while the server runs.

## Evaluation

Run it from the UI's Evaluation panel, or:

```bash
curl -X POST http://localhost:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"use_judge": false}'
```

`/evaluate` re-indexes the bundled sample repo (`eval/sample_repo`, into a
separate `eval` collection) and runs the 14-question dataset at
`eval/dataset.json` through it.

- **Precision@K** — fraction of the top-K retrieved chunks that are relevant.
- **Recall@K** — fraction of the ground-truth items found somewhere in the
  top-K.
- **MRR** — average of 1/rank of the first relevant chunk across questions.
- **LLM-as-judge** — Claude scores faithfulness, relevance, and correctness
  (1-5 each) for every generated answer; requires `ANTHROPIC_API_KEY`.

Measured locally (k=5, `use_judge: false`, no API key needed for retrieval-only
metrics), against the 5-file sample repo indexed fresh:

```json
{
  "k": 5,
  "num_examples": 14,
  "retrieval": {
    "precision_at_k": 0.2286,
    "recall_at_k": 1.0,
    "mrr": 0.9643
  },
  "generation": null
}
```

Judge scores (`generation`) require `ANTHROPIC_API_KEY`; without one, judge
calls fail per-row with an `error` field and `generation` stays `null` if no
row succeeds. Details, formulas, and worked examples: [`docs/evaluation.md`](docs/evaluation.md).

## Testing

```bash
uv run pytest
```

56 tests, none of which call the real Anthropic API — LLM calls are faked or
injected via a fake client in fixtures.

## Deployment

Docker-based deploy to Railway (auto-detects `railway.json` and the
`Dockerfile`), with a Docker-run and Render alternative documented in
[`docs/deployment.md`](docs/deployment.md).

**Live URL:** _pending — not yet deployed (no Railway credentials available
in this environment). Will be added here after deployment._

## Project structure

```
app/
  main.py          FastAPI app factory and endpoints
  chunker.py        Code-aware chunking (Python ast / line windows)
  store.py           ChromaDB vector store wrapper
  llm.py              Claude client: answers + LLM-as-judge
  rag.py              RAG pipeline: index / retrieve / query
  metrics.py         Precision@K, Recall@K, MRR
  evaluation.py     Evaluation runner
  static/               Web UI (index.html, style.css, app.js)
eval/
  dataset.json         14-example evaluation dataset
  sample_repo/         5-file sample codebase used for evaluation
tests/                    pytest suite (56 tests, no live API calls)
docs/
  architecture.md      Module responsibilities and data flow
  api.md                  Endpoint reference
  evaluation.md         Metrics, formulas, judge criteria
  deployment.md         Railway / Docker / Render instructions
Dockerfile
railway.json
```

## Design decisions & limitations

- **No authentication or rate limiting**: the API (`/query`, `/evaluate`) is
  wide open — anyone with the URL can spend the configured
  `ANTHROPIC_API_KEY`'s credits (a single `/evaluate` run with the judge
  enabled makes 28 Claude calls). If deploying publicly, set a spend limit on
  the API key, avoid sharing the URL publicly, or put the service behind
  auth.
- **Embedding truncation**: `all-MiniLM-L6-v2` only embeds the first ~256
  word pieces of each chunk, so long functions/windows are retrieved mostly
  by their beginning (signature, docstring) rather than their full body.
- **ChromaDB's built-in ONNX MiniLM** embedding function is used instead of
  the `sentence-transformers` package — same model, no `torch` dependency,
  and a smaller container image.
- **Re-indexing a file replaces its chunks**: `RAGPipeline.index_files`
  deletes all existing chunks for a path before adding the new ones, so
  uploading an updated file doesn't leave stale chunks behind.
- **Class attributes between methods** (module-level code outside any
  function or class body, and any class body content not captured as a
  method) fall into the `<module>` chunk for that file, not into a specific
  class chunk.
- **Ground truth for non-Python files** is path-level (e.g. `"utils.ts"`)
  rather than symbol-level, since only Python is parsed by `ast`; TypeScript
  chunks are line windows (`utils.ts::L1-40`) whose boundaries don't align
  with meaningful symbols.
- **Extension challenges** from the lab spec were not implemented. Where
  each would plug in:
  - *Hybrid search (BM25 + vector)*: `VectorStore.search` in `app/store.py`.
  - *Reranking*: `RAGPipeline.retrieve` in `app/rag.py`, after the vector
    search call.
  - *Caching*: `VectorStore.search` (query cache) or `ClaudeLLM._complete`
    (answer cache) in `app/store.py` / `app/llm.py`.
  - *Multiple codebases*: one `VectorStore` collection per repo, keyed by
    repo name, instead of the single `"code"` collection created in
    `create_app`.

## License

Apache-2.0 — see [LICENSE](LICENSE).
