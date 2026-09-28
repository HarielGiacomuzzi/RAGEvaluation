# Deployment

The app is a single Docker image (`Dockerfile`) with no external services
beyond the Anthropic API and a local ChromaDB directory that needs to persist
across restarts/deploys.

## Railway

1. Create a new Railway project from this GitHub repository. Railway
   auto-detects `railway.json` and builds with the Dockerfile
   (`"builder": "DOCKERFILE"`, `"dockerfilePath": "Dockerfile"`).
2. Add a **volume** mounted at `/data`. The image sets `CHROMA_DIR=/data/chroma`,
   so indexed chunks persist there across deploys and restarts; without the
   volume, every deploy starts with an empty index. The image also sets
   `ANONYMIZED_TELEMETRY=False` to disable ChromaDB's telemetry.
3. Set the **`ANTHROPIC_API_KEY`** environment variable (required for `/query`
   and judge scoring; the app starts and serves `/health` without it, but
   `/query` returns `502` until it's set).
4. Optionally set `LLM_MODEL` to override the default (`claude-opus-5`).
5. Generate a public domain for the service.
6. Railway's health check is configured in `railway.json`:
   `healthcheckPath: /health`, `healthcheckTimeout: 120` seconds,
   `restartPolicyType: ON_FAILURE`.

`PORT` is set by Railway automatically at runtime; the Dockerfile's `CMD`
reads it (`--port ${PORT}`, defaulting to `8000` if unset).

**Warning: no authentication or rate limiting.** `/query` and `/evaluate` are
unauthenticated — anyone with the deployed URL can spend the configured
`ANTHROPIC_API_KEY`'s credits (a full `/evaluate` run with the judge makes 28
Claude calls, 4 concurrent). Set a spend limit on the API key, don't share
the URL publicly, or put the service behind auth before deploying it
publicly.

**Live URL:** _pending — this app has not yet been deployed (no Railway
credentials available in this environment). Add the deployed URL here once
available._

## Local Docker

```bash
docker build -t rag-evaluation .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-... \
  -v rag-data:/data \
  rag-evaluation
```

The named volume `rag-data` persists `CHROMA_DIR` (`/data/chroma`) across
container restarts. The image bakes the `all-MiniLM-L6-v2` ONNX embedding
model in at build time (`RUN uv run --no-sync python -c "..."` in the
Dockerfile), so the first real request doesn't pay the ~80 MB download.

## Render (alternative)

Render supports the same Dockerfile directly:

1. New Web Service → connect the repo → runtime **Docker** (Render detects
   `Dockerfile` automatically; no `railway.json` equivalent is needed).
2. Add a **disk** mounted at `/data` (equivalent to the Railway volume above)
   so `CHROMA_DIR=/data/chroma` persists.
3. Set `ANTHROPIC_API_KEY` (and optionally `LLM_MODEL`) as environment
   variables.
4. Set the health check path to `/health`.
5. Render provides `PORT` automatically at runtime, same as Railway; the
   Dockerfile's `CMD` already reads it.

## Cost and latency notes

- **Retrieval-only evaluation** (`use_judge: false`) makes no LLM calls at
  all — embeddings are local (ONNX), so a full 14-example run costs nothing
  and completes in a few seconds.
- **Full evaluation** (`use_judge: true`, the default) makes **2 Claude
  calls per example** — one to generate the answer, one to judge it — so a
  full run over the 14-example dataset makes **28 Claude calls**, executed
  4 at a time (`JUDGE_WORKERS = 4` in `app/evaluation.py`).
- Each `/query` call makes exactly **1 Claude call**.
- Every Claude call sets `fallbacks="default"` with the
  `server-side-fallback-2026-07-01` beta, so a transient outage on the
  primary model is retried server-side against a fallback model rather than
  failing the request outright — this can add latency on a fallback but
  improves reliability.
- Embeddings (indexing and query-time search) never call the Anthropic API;
  they run locally via ChromaDB's ONNX `all-MiniLM-L6-v2`, so indexing cost
  and latency scale with local CPU, not API usage.
