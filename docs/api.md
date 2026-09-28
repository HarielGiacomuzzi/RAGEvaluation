# API Reference

Base URL: `http://localhost:8000` (local) or the Railway domain once deployed.
Interactive Swagger docs are auto-generated at `/docs`.

All JSON request bodies use pydantic models with `str_strip_whitespace` on
string fields. Validation failures return `422 Unprocessable Entity` with
FastAPI's standard error body.

## `GET /`

Serves the web UI (`app/static/index.html`). Not part of the JSON API; not
listed in `/docs`.

**Status codes:** `200`.

## `GET /health`

Liveness check and current chunk count for the `code` collection (the one
used by `/index/files` and `/query`, not the `eval` collection).

```bash
curl http://localhost:8000/health
```

Response (captured locally after indexing 5 files):

```json
{
  "status": "ok",
  "indexed_chunks": 22
}
```

**Status codes:** `200`.

## `POST /index/files`

Uploads one or more files (`multipart/form-data`) and indexes them into the
`code` collection.

**Request:** multipart form, field name `files`, repeated per file.

**Limits:**
- Each file must be **1 MB or smaller** (`MAX_FILE_BYTES = 1_000_000` bytes);
  larger files are skipped, not rejected.
- Each file must decode as **UTF-8 text**; files that don't are skipped, not
  rejected.
- Skipped files never fail the request — they're reported in the `skipped`
  array of the response.
- Re-uploading a file with the same path replaces its previously indexed
  chunks (no duplication).

```bash
curl -X POST http://localhost:8000/index/files \
  -F "files=@eval/sample_repo/auth.py" \
  -F "files=@eval/sample_repo/inventory.py" \
  -F "files=@eval/sample_repo/orders.py" \
  -F "files=@eval/sample_repo/pricing.py" \
  -F "files=@eval/sample_repo/utils.ts"
```

Response (real local call, 5 files):

```json
{
  "files_indexed": 5,
  "chunks_indexed": 22,
  "total_chunks": 22,
  "skipped": []
}
```

If a file were too large or not UTF-8, `skipped` would instead contain
entries like `{"path": "big.bin", "reason": "file larger than 1 MB"}`.

**Status codes:** `200` (even when some files are skipped), `422` (malformed
multipart request).

## `POST /query`

Asks a question about the currently indexed code (the `code` collection) and
returns a grounded, cited answer with the source chunks used.

**Request (JSON):**

| Field | Type | Constraints |
|---|---|---|
| `question` | string | 1-2000 characters, whitespace-stripped |
| `k` | integer | 1-20, default 5 |

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How are user passwords hashed?", "k": 5}'
```

**Without `ANTHROPIC_API_KEY` set** (real local response, status `502`):

```json
{
  "detail": "LLM request failed: \"Could not resolve authentication method. Expected one of api_key, auth_token, or credentials to be set. Or for one of the `X-Api-Key` or `Authorization` headers to be explicitly omitted\""
}
```

Any `LLMError` — missing credentials, an Anthropic API error, or the model
refusing to answer — is surfaced this way: `502` with `detail` set to the
error message.

**With a valid key**, a successful response has this shape (from
`app/main.py`'s `source_dict` and `QueryResult`):

```json
{
  "answer": "Passwords are hashed with PBKDF2-HMAC-SHA256 [auth.py::hash_password]...",
  "sources": [
    {
      "id": "auth.py::hash_password",
      "path": "auth.py",
      "symbol": "hash_password",
      "start_line": 5,
      "end_line": 12,
      "language": "python",
      "content": "def hash_password(password: str) -> tuple[bytes, bytes]:\n    ...",
      "score": 0.8134
    }
  ]
}
```

`score` is cosine similarity (higher is better), rounded to 4 decimal places.
If nothing has been indexed yet, `answer` is a fixed message
("No code has been indexed yet. Upload files in the Indexing section
first.") and `sources` is `[]` — no LLM call is made in that case, so no key
is required.

**Status codes:** `200`, `422` (invalid `question`/`k`), `502` (LLM error).

## `POST /evaluate`

Re-indexes the bundled `eval/sample_repo` into a separate `eval` collection
and runs the dataset at `eval/dataset.json` through retrieval (and, unless
disabled, generation + judging).

**Request (JSON, optional body — omitting it uses the defaults):**

| Field | Type | Constraints |
|---|---|---|
| `k` | integer | 1-20, default 5 |
| `use_judge` | boolean | default `true` |

```bash
curl -X POST http://localhost:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"use_judge": false}'
```

Response (real local call, `k=5`, `use_judge: false`, no API key present):

```json
{
  "k": 5,
  "num_examples": 14,
  "retrieval": {
    "precision_at_k": 0.2286,
    "recall_at_k": 1.0,
    "mrr": 0.9643
  },
  "generation": null,
  "results": [
    {
      "question": "How are user passwords hashed?",
      "relevant": ["auth.py::hash_password"],
      "retrieved": [
        "auth.py::hash_password",
        "auth.py::verify_password",
        "auth.py::<module>",
        "auth.py::generate_token",
        "inventory.py::Inventory.low_stock_items"
      ],
      "precision_at_k": 0.2,
      "recall_at_k": 1.0,
      "reciprocal_rank": 1.0
    }
  ]
}
```

(`results` truncated to one row above for brevity; a real call returns all 14.)

With `use_judge: true` and a valid `ANTHROPIC_API_KEY`, each row additionally
gets `answer` (the generated answer) and `judge`
(`{faithfulness, relevance, correctness, reasoning, overall}`), and the
top-level `generation` object holds the mean of those across judged rows plus
`judged` and `errors` counts. If a row's judge call fails (e.g. no API key),
that row gets an `error` string instead of `judge`, and `generation` is
computed only from rows that succeeded (`null` if none did).

**Status codes:** `200`, `422` (invalid `k`).

## Status code summary

| Code | Meaning | Where |
|---|---|---|
| `200` | Success | All endpoints |
| `422` | Request validation failed (bad `question`, `k`, or malformed body) | `/query`, `/evaluate`, `/index/files` |
| `502` | The LLM call failed (missing credentials, Anthropic API error, or model refusal) | `/query` |
