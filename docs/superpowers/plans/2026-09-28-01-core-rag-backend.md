# Core RAG Backend Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastAPI service that chunks uploaded code files (function/class-aware for Python, line windows otherwise), stores them in a persistent ChromaDB collection, and answers questions grounded in the retrieved code via Claude.

**Architecture:** Five small modules under `app/`: `chunker.py` (pure functions, `Chunk` dataclass), `store.py` (ChromaDB wrapper), `llm.py` (Claude calls), `rag.py` (pipeline gluing the three), `main.py` (FastAPI app factory `create_app`). Embeddings come from ChromaDB's built-in `DefaultEmbeddingFunction` — the sentence-transformers `all-MiniLM-L6-v2` model run through ONNX, so it's free and local with no torch dependency.

**Tech Stack:** Python 3.12, uv, FastAPI, ChromaDB 1.x (PersistentClient), anthropic SDK ≥1.8 (model `claude-opus-5`), pytest.

**Spec:** `requirements.md` (repo root)

## Global Constraints

- Python pinned to 3.12 (`.python-version`), because chromadb/onnxruntime wheels don't support the machine's default 3.14.
- Vector DB: ChromaDB **persistent** (`chromadb.PersistentClient`), directory from env `CHROMA_DIR` (default `./data/chroma`).
- Embeddings: sentence-transformers `all-MiniLM-L6-v2` via ChromaDB's default ONNX embedding function (free, local).
- Framework: FastAPI. Endpoints: `POST /index/files`, `POST /query` (this plan), `POST /evaluate` (Plan 2).
- LLM: Anthropic SDK, model from env `LLM_MODEL`, default `claude-opus-5`; server-side refusal fallback enabled (`betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`).
- App is built with the factory `app.main:create_app` (run as `uvicorn app.main:create_app --factory`), so importing the module has no side effects.
- Uploads: max 1 MB per file; non-UTF-8 files are skipped with a reason, never a 500.
- Tests never call the real Claude API. They use a `FakeLLM` or a fake client object.

## Review Focus

- Re-indexing a file with the same path replaces its old chunks. No stale or duplicate chunks may remain (test: `test_reindex_same_path_replaces_chunks`, Task 4).
- A binary or oversized upload is reported in `skipped` and never causes a 500 (test: `test_index_skips_binary_and_large_files`, Task 5).
- Querying before anything is indexed returns a friendly message without calling the LLM (test: `test_query_empty_index_skips_llm`, Task 4).
- A missing API key or any LLM failure gives HTTP 502 with a readable message, not a stack trace (test: `test_query_llm_failure_returns_502`, Task 5).
- A Python file with a syntax error still gets indexed through line-window chunking (test: `test_python_syntax_error_falls_back_to_windows`, Task 1). Asking for `k` larger than the number of stored chunks works (test: `test_search_k_larger_than_count`, Task 2).

---

### Task 1: Project scaffold + code chunker

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `app/__init__.py` (empty), `app/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Produces: `Chunk(path: str, symbol: str, start_line: int, end_line: int, content: str, language: str)` frozen dataclass with property `id -> "path::symbol"`; `chunk_file(path: str, text: str) -> list[Chunk]`; `detect_language(path: str) -> str`.
- Symbols: Python top-level function `name`; class header `ClassName`; method `ClassName.method`; leftover module-level code `<module>`; non-Python windows `L{start}-{end}`; duplicate symbols in the same file get `#2`, `#3`, and so on.

- [ ] **Step 1: Scaffold project**

`pyproject.toml`:
```toml
[project]
name = "rag-evaluation"
version = "0.1.0"
description = "Codebase Q&A RAG system with retrieval metrics and LLM-as-judge evaluation"
requires-python = ">=3.12,<3.13"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "python-multipart>=0.0.9",
    "chromadb>=1.0",
    "anthropic>=1.8",
]

[dependency-groups]
dev = ["pytest>=8", "httpx>=0.27"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```
`.python-version`: `3.12`

`.gitignore`:
```
.venv/
__pycache__/
.pytest_cache/
data/
.env
```
Run: `uv sync` and confirm the lock file `uv.lock` was created.

- [ ] **Step 2: Write the failing tests** (`tests/test_chunker.py`)

```python
from app.chunker import Chunk, chunk_file, detect_language

PY = '''import os

LIMIT = 3


def top(a):
    return a + 1


class Greeter:
    """Says hello."""

    greeting = "hi"

    @staticmethod
    def hello(name):
        return f"hi {name}"

    def bye(self):
        return "bye"
'''


def ids(chunks):
    return [c.id for c in chunks]


def test_python_chunks_by_function_class_and_method():
    chunks = chunk_file("pkg/mod.py", PY)
    assert ids(chunks) == [
        "pkg/mod.py::<module>",
        "pkg/mod.py::top",
        "pkg/mod.py::Greeter",
        "pkg/mod.py::Greeter.hello",
        "pkg/mod.py::Greeter.bye",
    ]
    by_id = {c.id: c for c in chunks}
    assert "import os" in by_id["pkg/mod.py::<module>"].content
    assert "LIMIT = 3" in by_id["pkg/mod.py::<module>"].content
    assert by_id["pkg/mod.py::top"].start_line == 6
    assert by_id["pkg/mod.py::top"].end_line == 7
    assert 'greeting = "hi"' in by_id["pkg/mod.py::Greeter"].content
    assert by_id["pkg/mod.py::Greeter.hello"].content.lstrip().startswith("@staticmethod")
    assert all(c.language == "python" for c in chunks)


def test_python_syntax_error_falls_back_to_windows():
    chunks = chunk_file("broken.py", "def oops(:\n    pass\n")
    assert ids(chunks) == ["broken.py::L1-2"]


def test_generic_file_uses_overlapping_line_windows():
    text = "\n".join(f"line {i}" for i in range(1, 101))
    chunks = chunk_file("web/app.ts", text)
    assert ids(chunks) == ["web/app.ts::L1-40", "web/app.ts::L31-70", "web/app.ts::L61-100"]
    assert chunks[0].language == "typescript"
    assert chunks[1].content.splitlines()[0] == "line 31"


def test_short_generic_file_is_single_chunk():
    assert ids(chunk_file("a.js", "const a = 1;\n")) == ["a.js::L1-1"]


def test_empty_file_has_no_chunks():
    assert chunk_file("empty.py", "   \n\n") == []


def test_duplicate_symbols_get_suffix():
    chunks = chunk_file("d.py", "def f():\n    return 1\n\n\ndef f():\n    return 2\n")
    assert ids(chunks) == ["d.py::f", "d.py::f#2"]


def test_detect_language():
    assert detect_language("x/y.py") == "python"
    assert detect_language("Y.TSX") == "typescript"
    assert detect_language("README") == "text"


def test_chunk_id():
    assert Chunk("a.py", "f", 1, 2, "x", "python").id == "a.py::f"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.chunker'`

- [ ] **Step 4: Implement `app/chunker.py`**

```python
"""Code-aware chunking: Python by function/class/method via ast, everything else by line windows."""
import ast
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

WINDOW = 40
OVERLAP = 10

LANGUAGES = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript",
    ".jsx": "javascript", ".go": "go", ".java": "java", ".rb": "ruby", ".rs": "rust",
    ".c": "c", ".cpp": "cpp", ".cs": "csharp", ".php": "php", ".md": "markdown",
}


@dataclass(frozen=True)
class Chunk:
    path: str
    symbol: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    content: str
    language: str

    @property
    def id(self) -> str:
        return f"{self.path}::{self.symbol}"


def detect_language(path: str) -> str:
    return LANGUAGES.get(PurePosixPath(path).suffix.lower(), "text")


def chunk_file(path: str, text: str) -> list[Chunk]:
    if not text.strip():
        return []
    language = detect_language(path)
    if language == "python":
        try:
            return _dedupe(_chunk_python(path, text))
        except SyntaxError:
            pass
    return _dedupe(_chunk_lines(path, text, language))


def _chunk_python(path: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    tree = ast.parse(text)
    chunks: list[Chunk] = []
    covered: set[int] = set()

    def span(node) -> tuple[int, int]:
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        return start, node.end_lineno

    def add(symbol: str, start: int, end: int) -> None:
        chunks.append(Chunk(path, symbol, start, end, "\n".join(lines[start - 1:end]), "python"))
        covered.update(range(start, end + 1))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name, *span(node))
        elif isinstance(node, ast.ClassDef):
            start, end = span(node)
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            header_end = span(methods[0])[0] - 1 if methods else end
            while header_end > start and not lines[header_end - 1].strip():
                header_end -= 1
            add(node.name, start, header_end)
            for method in methods:
                add(f"{node.name}.{method.name}", *span(method))

    rest = [i for i in range(1, len(lines) + 1) if i not in covered and lines[i - 1].strip()]
    if rest:
        module = "\n".join(lines[i - 1] for i in rest)
        chunks.insert(0, Chunk(path, "<module>", rest[0], rest[-1], module, "python"))
    return chunks


def _chunk_lines(path: str, text: str, language: str) -> list[Chunk]:
    lines = text.splitlines()
    chunks = []
    for start in range(0, len(lines), WINDOW - OVERLAP):
        window = lines[start:start + WINDOW]
        end = start + len(window)
        if any(line.strip() for line in window):
            chunks.append(Chunk(path, f"L{start + 1}-{end}", start + 1, end, "\n".join(window), language))
        if end >= len(lines):
            break
    return chunks


def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
    seen: dict[str, int] = {}
    out = []
    for chunk in chunks:
        seen[chunk.symbol] = seen.get(chunk.symbol, 0) + 1
        n = seen[chunk.symbol]
        out.append(chunk if n == 1 else replace(chunk, symbol=f"{chunk.symbol}#{n}"))
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: 8 passed. (Class attributes defined *between* methods land in `<module>`. This is a known, accepted limitation; don't fix it.)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore app/__init__.py app/chunker.py tests/test_chunker.py
git commit -m "feat: add project scaffold and code-aware chunker"
```

---

### Task 2: ChromaDB vector store

**Files:**
- Create: `app/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `Chunk` from `app.chunker`.
- Produces: `SearchResult(chunk: Chunk, score: float)` frozen dataclass (score = cosine similarity, `1 - distance`); `VectorStore(persist_dir: str, collection: str = "code")` with `add(chunks: list[Chunk]) -> None`, `delete_path(path: str) -> None`, `count() -> int`, `search(query: str, k: int) -> list[SearchResult]` (best first).

- [ ] **Step 1: Write the failing tests** (`tests/test_store.py`)

```python
from app.chunker import Chunk
from app.store import VectorStore

PW = Chunk("auth.py", "hash_password", 1, 3,
           "def hash_password(pw):\n    return pbkdf2_hmac('sha256', pw, salt, 100000)", "python")
SLUG = Chunk("text.ts", "L1-3", 1, 3,
             "export function slugify(s) { return s.toLowerCase().replace(/ /g, '-'); }", "typescript")


def test_search_returns_most_similar_first(tmp_path):
    store = VectorStore(str(tmp_path))
    store.add([PW, SLUG])
    results = store.search("how are passwords hashed", k=2)
    assert [r.chunk.id for r in results] == ["auth.py::hash_password", "text.ts::L1-3"]
    assert results[0].chunk == PW
    assert results[0].score > results[1].score


def test_upsert_same_id_does_not_duplicate(tmp_path):
    store = VectorStore(str(tmp_path))
    store.add([PW])
    store.add([PW])
    assert store.count() == 1


def test_delete_path_removes_only_that_file(tmp_path):
    store = VectorStore(str(tmp_path))
    store.add([PW, SLUG])
    store.delete_path("auth.py")
    assert [r.chunk.id for r in store.search("anything", k=5)] == ["text.ts::L1-3"]
    store.delete_path("missing.py")  # no error for an unknown path


def test_search_empty_store_returns_nothing(tmp_path):
    assert VectorStore(str(tmp_path)).search("q", k=5) == []


def test_search_k_larger_than_count(tmp_path):
    store = VectorStore(str(tmp_path))
    store.add([PW])
    assert len(store.search("q", k=20)) == 1


def test_persists_across_instances_and_collections_are_isolated(tmp_path):
    VectorStore(str(tmp_path)).add([PW])
    assert VectorStore(str(tmp_path)).count() == 1
    assert VectorStore(str(tmp_path), "other").count() == 0


def test_add_empty_list_is_noop(tmp_path):
    store = VectorStore(str(tmp_path))
    store.add([])
    assert store.count() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.store'`

- [ ] **Step 3: Implement `app/store.py`**

```python
"""Persistent ChromaDB vector store for code chunks (local all-MiniLM-L6-v2 ONNX embeddings)."""
from dataclasses import dataclass

import chromadb

from app.chunker import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity, higher is better


class VectorStore:
    def __init__(self, persist_dir: str, collection: str = "code"):
        client = chromadb.PersistentClient(path=persist_dir)
        self._col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.id for c in chunks],
            # Path and symbol are prepended so the embedding sees file and function names.
            documents=[f"# {c.path} :: {c.symbol}\n{c.content}" for c in chunks],
            metadatas=[
                {"path": c.path, "symbol": c.symbol, "start_line": c.start_line,
                 "end_line": c.end_line, "language": c.language, "content": c.content}
                for c in chunks
            ],
        )

    def delete_path(self, path: str) -> None:
        self._col.delete(where={"path": path})

    def count(self) -> int:
        return self._col.count()

    def search(self, query: str, k: int) -> list[SearchResult]:
        n = min(k, self.count())
        if n <= 0:
            return []
        result = self._col.query(query_texts=[query], n_results=n, include=["metadatas", "distances"])
        return [
            SearchResult(
                Chunk(m["path"], m["symbol"], m["start_line"], m["end_line"], m["content"], m["language"]),
                1 - d,
            )
            for m, d in zip(result["metadatas"][0], result["distances"][0])
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: 7 passed. (The first run may download the ~80 MB ONNX model into `~/.cache/chroma`.)

- [ ] **Step 5: Commit**

```bash
git add app/store.py tests/test_store.py
git commit -m "feat: add persistent ChromaDB vector store"
```

---

### Task 3: Claude LLM client (answers + judge)

**Files:**
- Create: `app/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `Chunk`.
- Produces: `LLMError(RuntimeError)`; `JudgeScore` pydantic model `(faithfulness: int, relevance: int, correctness: int, reasoning: str)` with property `overall -> float` (sum / 15, range 0.2–1.0); `ClaudeLLM(client=None, model=None)` with `answer(question: str, chunks: list[Chunk]) -> str` and `judge(question: str, answer: str, chunks: list[Chunk], reference: str) -> JudgeScore`; `format_context(chunks) -> str`; `DEFAULT_MODEL = "claude-opus-5"`.
- Every SDK or credential failure, refusal, or bad judge JSON raises `LLMError`.

- [ ] **Step 1: Write the failing tests** (`tests/test_llm.py`)

```python
import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from app.chunker import Chunk
from app.llm import DEFAULT_MODEL, ClaudeLLM, JudgeScore, LLMError

CHUNK = Chunk("auth.py", "hash_password", 1, 2, "def hash_password(): ...", "python")


class FakeClient:
    def __init__(self, text="", stop_reason="end_turn", error=None):
        self.calls = []
        self._text, self._stop, self._error = text, stop_reason, error
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return SimpleNamespace(stop_reason=self._stop,
                               content=[SimpleNamespace(type="text", text=self._text)])


def test_answer_sends_context_and_returns_text(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = FakeClient(text="It uses PBKDF2 [auth.py::hash_password].")
    out = ClaudeLLM(client=client).answer("How are passwords hashed?", [CHUNK])
    assert out == "It uses PBKDF2 [auth.py::hash_password]."
    call = client.calls[0]
    assert call["model"] == DEFAULT_MODEL == "claude-opus-5"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["fallbacks"] == "default"
    prompt = call["messages"][0]["content"]
    assert 'id="auth.py::hash_password"' in prompt and "How are passwords hashed?" in prompt
    assert "output_config" not in call


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5")
    client = FakeClient(text="x")
    ClaudeLLM(client=client).answer("q", [CHUNK])
    assert client.calls[0]["model"] == "claude-sonnet-5"


def test_judge_parses_and_clamps_scores():
    payload = {"faithfulness": 7, "relevance": 0, "correctness": 3, "reasoning": "ok"}
    client = FakeClient(text=json.dumps(payload))
    score = ClaudeLLM(client=client).judge("q", "a", [CHUNK], "ref")
    assert score == JudgeScore(faithfulness=5, relevance=1, correctness=3, reasoning="ok")
    assert score.overall == pytest.approx(9 / 15)
    fmt = client.calls[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert set(fmt["schema"]["required"]) == {"faithfulness", "relevance", "correctness", "reasoning"}
    assert "<reference_answer>ref</reference_answer>" in client.calls[0]["messages"][0]["content"]


def test_judge_invalid_json_raises():
    with pytest.raises(LLMError):
        ClaudeLLM(client=FakeClient(text="not json")).judge("q", "a", [CHUNK], "ref")


def test_refusal_raises():
    with pytest.raises(LLMError, match="declined"):
        ClaudeLLM(client=FakeClient(stop_reason="refusal")).answer("q", [CHUNK])


def test_sdk_error_is_wrapped():
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    with pytest.raises(LLMError, match="LLM request failed"):
        ClaudeLLM(client=FakeClient(error=err)).answer("q", [CHUNK])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.llm'`

- [ ] **Step 3: Implement `app/llm.py`**

```python
"""Claude calls: answers grounded in retrieved code, and LLM-as-judge scoring."""
import os

import anthropic
from pydantic import BaseModel

from app.chunker import Chunk

DEFAULT_MODEL = "claude-opus-5"

ANSWER_SYSTEM = """You answer questions about a codebase using ONLY the code snippets provided.
Cite the snippets you rely on by their id in square brackets, e.g. [auth.py::hash_password].
If the snippets do not contain the answer, say so plainly instead of guessing."""

JUDGE_SYSTEM = """You are a strict evaluator of answers produced by a code question-answering system.
Score each criterion with an integer from 1 (very poor) to 5 (excellent):
- faithfulness: every claim in the answer is supported by the retrieved code.
- relevance: the answer addresses the question that was asked.
- correctness: the answer agrees with the reference answer.
Explain the scores briefly in reasoning."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "integer"},
        "relevance": {"type": "integer"},
        "correctness": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["faithfulness", "relevance", "correctness", "reasoning"],
    "additionalProperties": False,
}

CRITERIA = ("faithfulness", "relevance", "correctness")


class LLMError(RuntimeError):
    pass


class JudgeScore(BaseModel):
    faithfulness: int
    relevance: int
    correctness: int
    reasoning: str

    @property
    def overall(self) -> float:
        return (self.faithfulness + self.relevance + self.correctness) / 15


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f'<snippet id="{c.id}" lines="{c.start_line}-{c.end_line}">\n{c.content}\n</snippet>'
        for c in chunks
    )


class ClaudeLLM:
    def __init__(self, client=None, model: str | None = None):
        self._client = client  # created lazily so the app starts without credentials
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    def _complete(self, system: str, user: str, output_config: dict | None = None) -> str:
        extra = {"output_config": output_config} if output_config else {}
        try:
            if self._client is None:
                self._client = anthropic.Anthropic()
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **extra,
            )
        except (anthropic.AnthropicError, TypeError) as e:  # TypeError: SDK found no credentials
            raise LLMError(f"LLM request failed: {e}") from e
        if response.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request.")
        return "".join(block.text for block in response.content if block.type == "text")

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        return self._complete(ANSWER_SYSTEM, f"{format_context(chunks)}\n\nQuestion: {question}")

    def judge(self, question: str, answer: str, chunks: list[Chunk], reference: str) -> JudgeScore:
        user = (
            f"<question>{question}</question>\n"
            f"<retrieved_code>\n{format_context(chunks)}\n</retrieved_code>\n"
            f"<reference_answer>{reference}</reference_answer>\n"
            f"<answer>{answer}</answer>"
        )
        text = self._complete(JUDGE_SYSTEM, user, {"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}})
        try:
            score = JudgeScore.model_validate_json(text)
        except ValueError as e:
            raise LLMError(f"Judge returned invalid JSON: {e}") from e
        return score.model_copy(update={k: min(5, max(1, getattr(score, k))) for k in CRITERIA})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add app/llm.py tests/test_llm.py
git commit -m "feat: add Claude client for grounded answers and LLM-as-judge"
```

---

### Task 4: RAG pipeline

**Files:**
- Create: `app/rag.py`, `tests/conftest.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `chunk_file`, `VectorStore`, `SearchResult`, an object with `answer(question, chunks)` and `judge(question, answer, chunks, reference)` (`ClaudeLLM` or `FakeLLM`).
- Produces: `IndexResult(files_indexed: int, chunks_indexed: int)`; `QueryResult(answer: str, sources: list[SearchResult])`; `RAGPipeline(store, llm)` with public attributes `.store` and `.llm`, and methods `index_files(files: list[tuple[str, str]]) -> IndexResult`, `retrieve(question: str, k: int = 5) -> list[SearchResult]`, `query(question: str, k: int = 5) -> QueryResult`; constant `NO_INDEX_ANSWER`.
- `tests/conftest.py` produces the `FakeLLM` class (importable as `from tests.conftest import FakeLLM`; add an empty `tests/__init__.py`) and a `store` fixture.

- [ ] **Step 1: Write shared test helpers** (`tests/__init__.py` empty, `tests/conftest.py`)

```python
import pytest

from app.llm import JudgeScore
from app.store import VectorStore


class FakeLLM:
    """Stands in for ClaudeLLM in tests; never touches the network."""

    def __init__(self, error: Exception | None = None):
        self.answer_calls = []
        self.judge_calls = []
        self.error = error

    def answer(self, question, chunks):
        self.answer_calls.append((question, chunks))
        if self.error:
            raise self.error
        return "fake answer from " + ", ".join(c.id for c in chunks)

    def judge(self, question, answer, chunks, reference):
        self.judge_calls.append((question, answer, chunks, reference))
        if self.error:
            raise self.error
        return JudgeScore(faithfulness=4, relevance=5, correctness=3, reasoning="fake")


@pytest.fixture
def store(tmp_path):
    return VectorStore(str(tmp_path / "chroma"))
```

- [ ] **Step 2: Write the failing tests** (`tests/test_rag.py`)

```python
from app.rag import NO_INDEX_ANSWER, RAGPipeline
from tests.conftest import FakeLLM

AUTH = "def hash_password(pw):\n    return pbkdf2(pw)\n\n\ndef make_token():\n    return secrets.token_hex()\n"
UTIL = "export function slugify(s: string) {\n  return s.toLowerCase();\n}\n"


def test_index_files_counts_files_and_chunks(store):
    result = RAGPipeline(store, FakeLLM()).index_files([("auth.py", AUTH), ("util.ts", UTIL)])
    assert (result.files_indexed, result.chunks_indexed) == (2, 3)
    assert store.count() == 3


def test_reindex_same_path_replaces_chunks(store):
    rag = RAGPipeline(store, FakeLLM())
    rag.index_files([("auth.py", AUTH)])
    rag.index_files([("auth.py", "def only_one():\n    pass\n")])
    assert [r.chunk.id for r in rag.retrieve("anything", k=10)] == ["auth.py::only_one"]


def test_query_retrieves_then_generates(store):
    llm = FakeLLM()
    rag = RAGPipeline(store, llm)
    rag.index_files([("auth.py", AUTH), ("util.ts", UTIL)])
    result = rag.query("how is a password hashed?", k=2)
    assert len(result.sources) == 2
    assert result.sources[0].chunk.id == "auth.py::hash_password"
    question, chunks = llm.answer_calls[0]
    assert question == "how is a password hashed?"
    assert [c.id for c in chunks] == [s.chunk.id for s in result.sources]
    assert result.answer.startswith("fake answer from auth.py::hash_password")


def test_query_empty_index_skips_llm(store):
    llm = FakeLLM()
    result = RAGPipeline(store, llm).query("anything?")
    assert result.answer == NO_INDEX_ANSWER
    assert result.sources == []
    assert llm.answer_calls == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_rag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.rag'`

- [ ] **Step 4: Implement `app/rag.py`**

```python
"""RAG pipeline: index (chunk, embed, store) and query (retrieve, generate)."""
from dataclasses import dataclass

from app.chunker import chunk_file
from app.store import SearchResult, VectorStore

NO_INDEX_ANSWER = "No code has been indexed yet. Upload files in the Indexing section first."


@dataclass(frozen=True)
class IndexResult:
    files_indexed: int
    chunks_indexed: int


@dataclass(frozen=True)
class QueryResult:
    answer: str
    sources: list[SearchResult]


class RAGPipeline:
    def __init__(self, store: VectorStore, llm):
        self.store = store
        self.llm = llm

    def index_files(self, files: list[tuple[str, str]]) -> IndexResult:
        total = 0
        for path, text in files:
            chunks = chunk_file(path, text)
            self.store.delete_path(path)  # re-indexing a file replaces its old chunks
            self.store.add(chunks)
            total += len(chunks)
        return IndexResult(len(files), total)

    def retrieve(self, question: str, k: int = 5) -> list[SearchResult]:
        return self.store.search(question, k)

    def query(self, question: str, k: int = 5) -> QueryResult:
        sources = self.retrieve(question, k)
        if not sources:
            return QueryResult(NO_INDEX_ANSWER, [])
        return QueryResult(self.llm.answer(question, [s.chunk for s in sources]), sources)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_rag.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add app/rag.py tests/__init__.py tests/conftest.py tests/test_rag.py
git commit -m "feat: add RAG pipeline for indexing and grounded querying"
```

---

### Task 5: FastAPI endpoints `/index/files`, `/query`, `/health`

**Files:**
- Create: `app/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `RAGPipeline`, `VectorStore`, `ClaudeLLM`, `LLMError`.
- Produces: `create_app(chroma_dir: str | None = None, llm=None) -> FastAPI` (defaults: env `CHROMA_DIR` or `./data/chroma`, and `ClaudeLLM()`); `app.state.pipeline` (collection `code`); `MAX_FILE_BYTES = 1_000_000`.
- HTTP contract:
  - `POST /index/files` multipart field `files` (repeatable; the filename may contain a relative path) returns `{"files_indexed": int, "chunks_indexed": int, "total_chunks": int, "skipped": [{"path": str, "reason": str}]}`
  - `POST /query` JSON `{"question": str (1–2000 chars, stripped), "k": int 1–20 = 5}` returns `{"answer": str, "sources": [{"id","path","symbol","start_line","end_line","language","content","score"}]}`. An LLM failure returns 502 `{"detail": str}`.
  - `GET /health` returns `{"status": "ok", "indexed_chunks": int}`

- [ ] **Step 1: Write the failing tests** (`tests/test_api.py`)

```python
import pytest
from fastapi.testclient import TestClient

from app.llm import LLMError
from app.main import MAX_FILE_BYTES, create_app
from app.rag import NO_INDEX_ANSWER
from tests.conftest import FakeLLM

AUTH = b"def hash_password(pw):\n    return pbkdf2(pw)\n\n\ndef make_token():\n    return 'x'\n"


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path), FakeLLM()))


def upload(client, *files):
    return client.post("/index/files", files=[("files", f) for f in files])


def test_index_and_query_roundtrip(client):
    r = upload(client, ("src/auth.py", AUTH, "text/x-python"))
    assert r.status_code == 200
    assert r.json() == {"files_indexed": 1, "chunks_indexed": 2, "total_chunks": 2, "skipped": []}
    r = client.post("/query", json={"question": "how are passwords hashed?", "k": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("fake answer from src/auth.py::hash_password")
    src = body["sources"][0]
    assert src["id"] == "src/auth.py::hash_password"
    assert src["path"] == "src/auth.py" and src["symbol"] == "hash_password"
    assert (src["start_line"], src["end_line"]) == (1, 2)
    assert src["language"] == "python" and "pbkdf2" in src["content"]
    assert isinstance(src["score"], float)


def test_index_skips_binary_and_large_files(client):
    r = upload(client,
               ("img.png", b"\x89PNG\r\n\x1a\n\xff\xfe\x00", "image/png"),
               ("big.py", b"x" * (MAX_FILE_BYTES + 1), "text/x-python"),
               ("ok.py", AUTH, "text/x-python"))
    assert r.status_code == 200
    body = r.json()
    assert body["files_indexed"] == 1
    assert {s["path"] for s in body["skipped"]} == {"img.png", "big.py"}
    assert all(s["reason"] for s in body["skipped"])


def test_query_empty_index(client):
    r = client.post("/query", json={"question": "anything?"})
    assert r.status_code == 200
    assert r.json() == {"answer": NO_INDEX_ANSWER, "sources": []}


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "   "}, {"question": "q", "k": 0}, {}])
def test_query_validation(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_query_llm_failure_returns_502(tmp_path):
    client = TestClient(create_app(str(tmp_path), FakeLLM(error=LLMError("LLM request failed: no key"))))
    upload(client, ("auth.py", AUTH, "text/x-python"))
    r = client.post("/query", json={"question": "how are passwords hashed?"})
    assert r.status_code == 502
    assert "no key" in r.json()["detail"]


def test_health_reports_chunk_count(client):
    assert client.get("/health").json() == {"status": "ok", "indexed_chunks": 0}
    upload(client, ("auth.py", AUTH, "text/x-python"))
    assert client.get("/health").json()["indexed_chunks"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Implement `app/main.py`**

```python
"""FastAPI app factory: index and query endpoints. Run: uvicorn app.main:create_app --factory"""
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from app.llm import ClaudeLLM, LLMError
from app.rag import RAGPipeline
from app.store import SearchResult, VectorStore

MAX_FILE_BYTES = 1_000_000


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=20)


def source_dict(result: SearchResult) -> dict:
    c = result.chunk
    return {"id": c.id, "path": c.path, "symbol": c.symbol, "start_line": c.start_line,
            "end_line": c.end_line, "language": c.language, "content": c.content,
            "score": round(result.score, 4)}


def create_app(chroma_dir: str | None = None, llm=None) -> FastAPI:
    chroma_dir = chroma_dir or os.environ.get("CHROMA_DIR", "./data/chroma")
    llm = llm or ClaudeLLM()
    app = FastAPI(title="Codebase RAG", description="Index code, ask questions, evaluate retrieval and answers.")
    pipeline = RAGPipeline(VectorStore(chroma_dir, "code"), llm)
    app.state.pipeline = pipeline

    @app.post("/index/files")
    async def index_files(files: list[UploadFile] = File(...)):
        accepted, skipped = [], []
        for upload in files:
            name = upload.filename or "unnamed"
            data = await upload.read()
            if len(data) > MAX_FILE_BYTES:
                skipped.append({"path": name, "reason": "file larger than 1 MB"})
                continue
            try:
                accepted.append((name, data.decode("utf-8")))
            except UnicodeDecodeError:
                skipped.append({"path": name, "reason": "not a UTF-8 text file"})
        result = await run_in_threadpool(pipeline.index_files, accepted)
        return {"files_indexed": result.files_indexed, "chunks_indexed": result.chunks_indexed,
                "total_chunks": pipeline.store.count(), "skipped": skipped}

    @app.post("/query")
    def query(req: QueryRequest):
        try:
            result = pipeline.query(req.question, req.k)
        except LLMError as e:
            raise HTTPException(status_code=502, detail=str(e))
        return {"answer": result.answer, "sources": [source_dict(s) for s in result.sources]}

    @app.get("/health")
    def health():
        return {"status": "ok", "indexed_chunks": pipeline.store.count()}

    return app
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: all tests pass (8 + 7 + 6 + 4 + 9 = 34).

- [ ] **Step 5: Smoke-test the server**

Run: `CHROMA_DIR=$(mktemp -d) uv run uvicorn app.main:create_app --factory --port 8765 &` then `curl -s localhost:8765/health` and expect `{"status":"ok","indexed_chunks":0}`. Kill the server afterwards.

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: add FastAPI index and query endpoints"
```
