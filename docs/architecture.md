# Architecture

## Modules

### `app/chunker.py` — code-aware chunking

Splits source text into `Chunk` objects. Public interface:

- `Chunk` (frozen dataclass): `path`, `symbol`, `start_line`, `end_line`
  (1-based, inclusive), `content`, `language`; `.id` returns `f"{path}::{symbol}"`.
- `detect_language(path) -> str` — maps a file extension to a language name
  (`.py` → `python`, `.ts`/`.tsx` → `typescript`, etc.), `"text"` if unknown.
- `chunk_file(path, text) -> list[Chunk]` — the entry point. Python files are
  parsed with `ast`; any other language, or a Python file that fails to parse
  (`SyntaxError`, `ValueError`, `RecursionError`), falls back to fixed-size
  line windows.

Python chunking (`_chunk_python`) walks top-level `ast` nodes:

- A top-level function or async function becomes one chunk, symbol = its name.
- A top-level class becomes a chunk for its header (from the `class` line —
  or its decorators — up to just before the first method, trailing blank
  lines stripped), symbol = the class name; each method inside it becomes its
  own chunk with symbol `ClassName.method_name`.
- Any line not covered by a function/class/method chunk (imports, module
  constants, code between methods that isn't itself a method) is collected
  into a single chunk with symbol `<module>`, inserted first.

Everything else (`_chunk_lines`) uses fixed windows: `WINDOW = 40` lines per
chunk, `OVERLAP = 10` lines between consecutive windows, symbol `L{start}-{end}`
(1-based line range). A window with no non-blank lines is skipped.

`_dedupe` appends `#2`, `#3`, ... to the symbol of any chunk whose symbol
already occurred in that file (e.g. two overloaded top-level functions with
the same name), so ids stay unique per file.

### `app/store.py` — vector store

Wraps a single ChromaDB collection.

- `SearchResult` (frozen dataclass): `chunk: Chunk`, `score: float` (cosine
  similarity, higher is better — Chroma returns cosine *distance*, so the
  store computes `1 - distance`).
- `VectorStore(persist_dir, collection="code")` — opens (or creates) a
  `chromadb.PersistentClient` collection at `persist_dir` with
  `hnsw:space: cosine`. Embeddings are produced by ChromaDB's default
  embedding function (`all-MiniLM-L6-v2`, ONNX, downloaded on first use).
- `.add(chunks)` — upserts chunks by id. The embedded document text is
  `"# {path} :: {symbol}\n{content}"`, so the embedding sees the file path
  and symbol name, not just the raw code body. Full chunk data is stored in
  Chroma metadata for retrieval.
- `.delete_path(path)` — deletes every chunk whose metadata `path` matches
  (used before re-adding a file's chunks).
- `.count()` — number of chunks in the collection.
- `.search(query, k)` — embeds `query`, returns up to `min(k, count())`
  `SearchResult`s ordered by similarity; returns `[]` if the collection is
  empty.

### `app/llm.py` — Claude client

- `DEFAULT_MODEL = "claude-opus-5"`, overridable via the `LLM_MODEL` env var.
- `ANSWER_SYSTEM` — instructs Claude to answer using only the given snippets
  and to cite snippet ids in square brackets (e.g. `[auth.py::hash_password]`),
  and to say plainly when the snippets don't answer the question.
- `JUDGE_SYSTEM` — instructs Claude to score `faithfulness`, `relevance`, and
  `correctness` from 1 (very poor) to 5 (excellent) and explain briefly.
- `ClaudeLLM(client=None, model=None)` — the Anthropic client is created
  lazily (`anthropic.Anthropic()`) on first use, so the app can start without
  credentials; only `/query` and judge calls need `ANTHROPIC_API_KEY`.
  - `.answer(question, chunks) -> str` — formats chunks as `<snippet id=... lines=...>` blocks and asks Claude for a grounded answer.
  - `.judge(question, answer, chunks, reference) -> JudgeScore` — asks Claude
    for structured JSON (`output_config.format = json_schema`, schema
    `JUDGE_SCHEMA`) and validates it into a `JudgeScore` pydantic model,
    clamping each criterion to `[1, 5]`.
  - `JudgeScore.overall` — `(faithfulness + relevance + correctness) / 15`.
- Every Claude call passes `betas=["server-side-fallback-2026-07-01"]` and
  `fallbacks="default"`, enabling Anthropic's server-side fallback so a
  transient model outage is retried against a fallback model automatically
  instead of failing the request.
- Raises `LLMError` (a `RuntimeError` subclass) on any `anthropic.AnthropicError`,
  on a `TypeError` from the SDK finding no credentials, or when the model's
  `stop_reason` is `"refusal"`.

### `app/rag.py` — pipeline

- `IndexResult(files_indexed, chunks_indexed)`, `QueryResult(answer, sources)`.
- `RAGPipeline(store, llm)`:
  - `.index_files(files: list[tuple[path, text]]) -> IndexResult` — for each
    file, chunks it, deletes any existing chunks for that path, then adds the
    new chunks (re-indexing replaces, never accumulates duplicates).
  - `.retrieve(question, k=5) -> list[SearchResult]` — thin wrapper over
    `store.search`.
  - `.query(question, k=5) -> QueryResult` — retrieves, and if nothing is
    indexed returns a fixed `NO_INDEX_ANSWER` message with no sources;
    otherwise asks the LLM to answer using the retrieved chunks.

### `app/metrics.py` — retrieval metrics

Pure functions over lists of chunk ids. Ground-truth items in the dataset are
either a full chunk id (`"auth.py::hash_password"`) or a bare file path
(`"utils.ts"`).

- `matches(chunk_id, relevant_item)` — true if the chunk id equals the item,
  or if the chunk id's path segment (`chunk_id.split("::", 1)[0]`) equals the
  item — this is what makes path-level ground truth work.
- `precision_at_k(retrieved, relevant, k)` — fraction of the top-k retrieved
  ids that match any relevant item.
- `recall_at_k(retrieved, relevant, k)` — fraction of relevant items matched
  by at least one of the top-k retrieved ids.
- `reciprocal_rank(retrieved, relevant)` — `1 / rank` of the first matching
  retrieved id, `0.0` if none match.
- All three raise `ValueError` if `k <= 0`.

### `app/evaluation.py` — evaluation runner

- `EvalExample(question, relevant, reference_answer)`.
- `load_dataset(path=eval/dataset.json)` — parses the JSON dataset.
- `load_sample_repo(root=eval/sample_repo)` — reads every file under the
  sample repo as `(relative_path, text)` tuples, sorted.
- `_evaluate_example(pipeline, ex, k, use_judge)` — retrieves for the
  question, computes precision/recall/reciprocal-rank against `ex.relevant`;
  if `use_judge`, also generates an answer and asks the judge, storing
  `judge` (scores + `overall`) or `error` (the `LLMError` message) on the row.
- `run_evaluation(pipeline, examples, k=5, use_judge=True) -> dict` — runs all
  examples concurrently (`ThreadPoolExecutor`, `JUDGE_WORKERS = 4`), then
  averages retrieval metrics across all rows and, if any row was judged,
  averages `faithfulness`, `relevance`, `correctness`, `overall` across judged
  rows only (plus `judged` and `errors` counts). Raises `ValueError` if the
  example list is empty.

### `app/main.py` — FastAPI app

`create_app(chroma_dir=None, llm=None)` builds two independent
`RAGPipeline`s, each backed by its own ChromaDB collection in the same
`chroma_dir`:

- `pipeline` — collection `"code"`, used by `/index/files` and `/query`.
- `eval_pipeline` — collection `"eval"`, used by `/evaluate`; kept separate
  so running an evaluation never mixes with or clobbers whatever the user has
  indexed for querying.

Endpoints and their error handling:

- `POST /index/files` — reads each upload; files over `MAX_FILE_BYTES`
  (1,000,000 bytes) or that fail UTF-8 decoding are recorded in `skipped`
  with a reason and excluded from indexing (never raise an error). Indexing
  itself runs in a threadpool (`run_in_threadpool`) since Chroma calls are
  synchronous.
- `POST /query` — catches `LLMError` and re-raises as `HTTPException(502)`
  with the error message as `detail`.
- `GET /health` — always 200; reports the `"code"` collection's chunk count.
- `POST /evaluate` — re-indexes the bundled sample repo into `eval_pipeline`
  (idempotent — re-indexing replaces each file's chunks) and runs
  `run_evaluation`. Per-row judge errors are captured in each row's `error`
  field, not raised as HTTP errors.
- `GET /` — serves `app/static/index.html`; `/static/*` serves the rest of
  the UI.

## Data flow

**Indexing**: client POSTs files (multipart) to `/index/files` → each file's
bytes are size/UTF-8 checked → accepted files go to
`RAGPipeline.index_files` → `chunk_file` splits each file into `Chunk`s →
`VectorStore.delete_path` clears any prior chunks for that path →
`VectorStore.add` embeds and upserts the new chunks into ChromaDB.

**Querying**: client POSTs `{question, k}` to `/query` → `RAGPipeline.query`
calls `VectorStore.search` (embeds the question, does a cosine-similarity
search, returns up to `k` `SearchResult`s) → if none, return the
no-index-yet message → otherwise `ClaudeLLM.answer` is called with the
retrieved chunks and returns a grounded, cited answer → response includes the
answer and the full source chunks (with per-chunk similarity score).

**Evaluation**: client POSTs `{k, use_judge}` to `/evaluate` →
`load_sample_repo` reads `eval/sample_repo` → `eval_pipeline.index_files`
re-indexes it into the `"eval"` collection → `load_dataset` reads
`eval/dataset.json` → each example is retrieved and scored (metrics.py) in
parallel; if `use_judge`, each example also gets a generated answer and a
judge score → results are aggregated into `retrieval` (mean precision,
recall, MRR) and, if judged, `generation` (mean faithfulness, relevance,
correctness, overall) → the full per-example `results` list is returned too.

## Chunk id scheme

A chunk's id is always `{path}::{symbol}`, where `symbol` is one of:

- A function or method name, e.g. `hash_password`.
- `ClassName.method_name` for a method inside a class, e.g. `Inventory.remove_item`.
- The class name alone, for the class header chunk, e.g. `Inventory`.
- `<module>` for top-level code in a Python file that isn't a function or
  class definition (imports, constants, code between class methods that
  the AST walk doesn't attribute to a method).
- `L{start}-{end}` (1-based, inclusive line range) for non-Python files
  chunked by line window, e.g. `utils.ts::L1-40`.
- Any of the above with a `#n` suffix (`#2`, `#3`, ...) if the same symbol
  name occurs more than once in a file (added by `_dedupe`).

## Chroma collections

Two collections live side by side in the same `CHROMA_DIR`:

| Collection | Used by | Purpose |
|---|---|---|
| `code` | `/index/files`, `/query` | User-indexed code for interactive Q&A |
| `eval` | `/evaluate` | The bundled `eval/sample_repo`, re-indexed on every evaluation run |

## Prompts

- **`ANSWER_SYSTEM`** (used for both `/query` answers and evaluation-generation
  answers): tells Claude to answer strictly from the provided snippets, cite
  snippet ids in `[brackets]`, and say plainly if the snippets don't contain
  the answer rather than guessing.
- **`JUDGE_SYSTEM`**: tells Claude to score `faithfulness` (every claim
  supported by the retrieved code), `relevance` (answers the question asked),
  and `correctness` (agrees with the reference answer) each 1-5, with brief
  reasoning, returned as structured JSON matching `JUDGE_SCHEMA`.

## Error handling

| Failure | Where | Result |
|---|---|---|
| `LLMError` from `/query` (no API key, API error, or model refusal) | `app/main.py` | `HTTPException(502)`, `detail` = error message |
| `LLMError` per evaluation row (answer or judge call fails) | `app/evaluation.py` | that row gets an `error` field; `judge` is omitted; the run still completes and returns 200 |
| Upload over 1 MB or not valid UTF-8 | `app/main.py` | file is skipped, listed in the response's `skipped` array with a reason; no error raised |
| Request validation (missing/out-of-range fields) | FastAPI / pydantic | `422 Unprocessable Entity` |
