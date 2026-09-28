# Evaluation Framework Implementation Plan (Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure RAG quality with Precision@K, Recall@K, MRR, and LLM-as-judge generation scores over a 14-example ground-truth dataset, exposed as `POST /evaluate`.

**Architecture:** `app/metrics.py` holds pure metric functions. `eval/sample_repo/` is a small, fixed codebase, and `eval/dataset.json` holds questions whose ground truth is chunk ids (`path::symbol`) or whole file paths. `app/evaluation.py` indexes the sample repo into a separate `eval` Chroma collection, so the user's index is never touched. It then scores every example, calling the judge from `app/llm.py` when requested, and aggregates the results.

**Tech Stack:** Python 3.12, FastAPI, ChromaDB, anthropic SDK, pytest. Builds on Plan 1 (`docs/superpowers/plans/2026-09-28-01-core-rag-backend.md`), which must be complete.

**Spec:** `requirements.md` (repo root)

## Global Constraints

- Metrics required: Precision@K, Recall@K, MRR (Mean Reciprocal Rank), and LLM-as-judge for generation quality.
- The evaluation dataset has at least 10 examples (this plan ships 14).
- Relevance rule: a retrieved chunk id `path::symbol` matches a ground-truth entry `r` when `chunk_id == r` or `path == r`. A path-only entry covers any chunk from that file; this suits non-Python files chunked into line windows.
- Precision@K divides by K even when fewer than K chunks are retrieved (the standard definition).
- The evaluation uses Chroma collection `eval`, never `code`.
- Judge failures (no API key, refusal, bad JSON) are recorded on the affected row as `error`. Retrieval metrics are still returned and the endpoint never 500s because of the LLM.
- All Plan 1 Global Constraints still apply.

## Review Focus

- Running `/evaluate` without an API key still returns retrieval metrics, with `generation: null` and a per-row `error` (test: `test_evaluate_endpoint_judge_failure_keeps_retrieval_metrics`, Task 4).
- A dataset ground-truth id that doesn't match any real chunk would silently zero the metrics. Every entry must resolve to a real chunk or path (test: `test_dataset_ground_truth_exists_in_sample_repo`, Task 2).
- Running `/evaluate` twice must not duplicate chunks or change the scores (test: `test_evaluate_endpoint_is_repeatable`, Task 4).
- `k` larger than the number of chunks gives precision below 1 but no crash (test: `test_precision_divides_by_k_even_if_fewer_retrieved`, Task 1).
- An empty example list raises a clear `ValueError` instead of a `StatisticsError` from `mean` (test: `test_run_evaluation_rejects_empty_dataset`, Task 3).

---

### Task 1: Retrieval metrics

**Files:**
- Create: `app/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `matches(chunk_id: str, relevant_item: str) -> bool`; `precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float`; `recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float`; `reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float`. `k <= 0` raises `ValueError`. Empty `relevant` gives recall 0.0.

- [ ] **Step 1: Write the failing tests** (`tests/test_metrics.py`)

```python
import pytest

from app.metrics import matches, precision_at_k, recall_at_k, reciprocal_rank

RETRIEVED = ["a.py::f", "b.py::g", "c.ts::L1-40", "a.py::h"]


def test_matches_exact_id_or_path():
    assert matches("a.py::f", "a.py::f")
    assert matches("c.ts::L1-40", "c.ts")
    assert not matches("a.py::f", "a.py::h")
    assert not matches("a.py::f", "a.p")


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, ["a.py::f", "a.py::h"], 2) == 0.5
    assert precision_at_k(RETRIEVED, ["a.py::f", "a.py::h"], 4) == 0.5
    assert precision_at_k(RETRIEVED, ["c.ts"], 3) == pytest.approx(1 / 3)


def test_precision_divides_by_k_even_if_fewer_retrieved():
    assert precision_at_k(["a.py::f"], ["a.py::f"], 5) == 0.2


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, ["a.py::f", "a.py::h"], 2) == 0.5
    assert recall_at_k(RETRIEVED, ["a.py::f", "a.py::h"], 4) == 1.0
    assert recall_at_k(RETRIEVED, ["zzz.py::x"], 4) == 0.0
    assert recall_at_k(RETRIEVED, [], 4) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, ["a.py::f"]) == 1.0
    assert reciprocal_rank(RETRIEVED, ["c.ts"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(RETRIEVED, ["nope"]) == 0.0
    assert reciprocal_rank([], ["a.py::f"]) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(RETRIEVED, ["a.py::f"], 0)
    with pytest.raises(ValueError):
        recall_at_k(RETRIEVED, ["a.py::f"], -1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.metrics'`

- [ ] **Step 3: Implement `app/metrics.py`**

```python
"""Retrieval metrics. Ground-truth items are chunk ids ("path::symbol") or bare file paths."""


def matches(chunk_id: str, relevant_item: str) -> bool:
    return chunk_id == relevant_item or chunk_id.split("::", 1)[0] == relevant_item


def _is_relevant(chunk_id: str, relevant: list[str]) -> bool:
    return any(matches(chunk_id, r) for r in relevant)


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of the top-k slots holding a relevant chunk."""
    _check_k(k)
    return sum(_is_relevant(c, relevant) for c in retrieved[:k]) / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of ground-truth items matched by at least one top-k chunk."""
    _check_k(k)
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(any(matches(c, r) for c in top) for r in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1 / rank of the first relevant chunk, 0 if none; averaged over queries this is MRR."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if _is_relevant(chunk_id, relevant):
            return 1 / rank
    return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add app/metrics.py tests/test_metrics.py
git commit -m "feat: add Precision@K, Recall@K and reciprocal rank metrics"
```

---

### Task 2: Sample codebase + evaluation dataset

**Files:**
- Create: `eval/sample_repo/auth.py`, `eval/sample_repo/inventory.py`, `eval/sample_repo/pricing.py`, `eval/sample_repo/orders.py`, `eval/sample_repo/utils.ts`, `eval/dataset.json`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces: `eval/dataset.json` is a JSON array of `{"question": str, "relevant": [str], "reference_answer": str}`. The sample repo files are indexed with paths relative to `eval/sample_repo/`, for example `auth.py` and `utils.ts`.

- [ ] **Step 1: Create the sample repo files exactly as below**

`eval/sample_repo/auth.py`:
```python
"""Password hashing and session tokens."""
import hashlib
import hmac
import secrets

ITERATIONS = 200_000


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Hash a password with PBKDF2-HMAC-SHA256 and a random 16-byte salt."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return salt, digest


def verify_password(password: str, salt: bytes, expected: bytes) -> bool:
    """Recompute the hash with the stored salt and compare in constant time."""
    _, digest = hash_password(password, salt)
    return hmac.compare_digest(digest, expected)


def generate_token(nbytes: int = 32) -> str:
    """Create a URL-safe random session token."""
    return secrets.token_urlsafe(nbytes)
```

`eval/sample_repo/inventory.py`:
```python
"""In-memory stock keeping."""


class OutOfStockError(Exception):
    """Raised when removing more units than are available."""


class Inventory:
    """Tracks stock levels per SKU."""

    def __init__(self):
        self._stock: dict[str, int] = {}

    def add_item(self, sku: str, quantity: int) -> None:
        """Increase stock for a SKU; quantity must be positive."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self._stock[sku] = self._stock.get(sku, 0) + quantity

    def remove_item(self, sku: str, quantity: int) -> None:
        """Decrease stock, raising OutOfStockError if not enough units exist."""
        available = self._stock.get(sku, 0)
        if quantity > available:
            raise OutOfStockError(f"{sku}: requested {quantity}, only {available} left")
        self._stock[sku] = available - quantity

    def get_stock(self, sku: str) -> int:
        return self._stock.get(sku, 0)

    def low_stock_items(self, threshold: int = 5) -> list[str]:
        """Return SKUs whose stock is below the threshold, sorted alphabetically."""
        return sorted(sku for sku, qty in self._stock.items() if qty < threshold)
```

`eval/sample_repo/pricing.py`:
```python
"""Price calculations."""

TAX_RATE = 0.08


def apply_discount(price: float, percent: float) -> float:
    """Reduce price by a percentage between 0 and 100, rounded to cents."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return round(price * (1 - percent / 100), 2)


def calculate_tax(amount: float, rate: float = TAX_RATE) -> float:
    """Sales tax for an amount, rounded to cents."""
    return round(amount * rate, 2)


def bulk_discount_rate(quantity: int) -> float:
    """Percent discount for buying in bulk: 10% from 100 units, 5% from 20 units."""
    if quantity >= 100:
        return 10.0
    if quantity >= 20:
        return 5.0
    return 0.0
```

`eval/sample_repo/orders.py`:
```python
"""Orders built from inventory and pricing."""
from dataclasses import dataclass, field

from inventory import Inventory, OutOfStockError
from pricing import apply_discount, bulk_discount_rate, calculate_tax


@dataclass
class Order:
    """A customer order made of (sku, quantity, unit_price) lines."""

    lines: list[tuple[str, int, float]] = field(default_factory=list)

    def add_line(self, sku: str, quantity: int, unit_price: float) -> None:
        self.lines.append((sku, quantity, unit_price))

    def total(self) -> float:
        """Sum of lines after bulk discounts, plus sales tax."""
        subtotal = sum(
            apply_discount(qty * price, bulk_discount_rate(qty)) for _, qty, price in self.lines
        )
        return round(subtotal + calculate_tax(subtotal), 2)


def place_order(inventory: Inventory, order: Order) -> float:
    """Reserve stock for every line, rolling back if any line is out of stock."""
    reserved = []
    try:
        for sku, qty, _ in order.lines:
            inventory.remove_item(sku, qty)
            reserved.append((sku, qty))
    except OutOfStockError:
        for sku, qty in reserved:
            inventory.add_item(sku, qty)
        raise
    return order.total()
```

`eval/sample_repo/utils.ts`:
```ts
// Formatting helpers used by the storefront UI.

export function formatCurrency(amount: number, currency = "USD"): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency }).format(amount);
}

export function slugify(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export function debounce<T extends (...args: unknown[]) => void>(fn: T, waitMs: number): T {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return ((...args: unknown[]) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), waitMs);
  }) as T;
}
```

- [ ] **Step 2: Create `eval/dataset.json`**

```json
[
  {"question": "How are user passwords hashed?",
   "relevant": ["auth.py::hash_password"],
   "reference_answer": "hash_password uses PBKDF2-HMAC-SHA256 (hashlib.pbkdf2_hmac) with 200,000 iterations and a random 16-byte salt from secrets.token_bytes, returning a (salt, digest) tuple."},
  {"question": "How does the code check whether a password is correct?",
   "relevant": ["auth.py::verify_password"],
   "reference_answer": "verify_password re-hashes the given password with the stored salt via hash_password and compares the digests with hmac.compare_digest, a constant-time comparison."},
  {"question": "How are session tokens created?",
   "relevant": ["auth.py::generate_token"],
   "reference_answer": "generate_token returns secrets.token_urlsafe(nbytes), a URL-safe random token built from 32 random bytes by default."},
  {"question": "What happens if I remove more units of a product than are in stock?",
   "relevant": ["inventory.py::Inventory.remove_item", "inventory.py::OutOfStockError"],
   "reference_answer": "Inventory.remove_item raises OutOfStockError with a message showing the requested and available quantity, and the stock level is left unchanged."},
  {"question": "How can I list products that are running low on stock?",
   "relevant": ["inventory.py::Inventory.low_stock_items"],
   "reference_answer": "Inventory.low_stock_items(threshold=5) returns the SKUs whose stock is below the threshold, sorted alphabetically."},
  {"question": "How do I add stock for a product?",
   "relevant": ["inventory.py::Inventory.add_item"],
   "reference_answer": "Inventory.add_item(sku, quantity) increases the stock for the SKU; it raises ValueError if quantity is zero or negative."},
  {"question": "How is a percentage discount applied to a price?",
   "relevant": ["pricing.py::apply_discount"],
   "reference_answer": "apply_discount multiplies the price by (1 - percent/100) and rounds to cents; percent must be between 0 and 100 or ValueError is raised."},
  {"question": "How is sales tax computed and what is the default rate?",
   "relevant": ["pricing.py::calculate_tax", "pricing.py::<module>"],
   "reference_answer": "calculate_tax multiplies the amount by the rate and rounds to cents; the default rate is TAX_RATE = 0.08 (8%)."},
  {"question": "What discount do customers get for buying in bulk?",
   "relevant": ["pricing.py::bulk_discount_rate"],
   "reference_answer": "bulk_discount_rate gives 10% for 100 or more units, 5% for 20 or more units, and 0% otherwise."},
  {"question": "How is the total price of an order calculated?",
   "relevant": ["orders.py::Order.total"],
   "reference_answer": "Order.total applies the bulk discount to each line (quantity x unit price), sums the lines into a subtotal, adds sales tax from calculate_tax, and rounds to cents."},
  {"question": "What happens when an order is placed but one of the items is out of stock?",
   "relevant": ["orders.py::place_order"],
   "reference_answer": "place_order removes stock line by line; if any line raises OutOfStockError it adds back the already reserved quantities (rollback) and re-raises the error."},
  {"question": "How are amounts formatted as currency strings?",
   "relevant": ["utils.ts"],
   "reference_answer": "formatCurrency uses Intl.NumberFormat with the en-US locale and style 'currency', defaulting to USD."},
  {"question": "How is text turned into a URL slug?",
   "relevant": ["utils.ts"],
   "reference_answer": "slugify lowercases and trims the text, replaces runs of non-alphanumeric characters with '-', and strips leading and trailing dashes."},
  {"question": "How can I limit how often a function is called in the UI?",
   "relevant": ["utils.ts"],
   "reference_answer": "debounce wraps a function so it only runs after waitMs milliseconds have passed without another call, by clearing and resetting a setTimeout timer."}
]
```

- [ ] **Step 3: Write the dataset integrity tests** (`tests/test_dataset.py`)

```python
import json
from pathlib import Path

from app.chunker import chunk_file
from app.metrics import matches

ROOT = Path(__file__).resolve().parent.parent / "eval"


def sample_chunk_ids():
    repo = ROOT / "sample_repo"
    return [c.id for p in sorted(repo.rglob("*")) if p.is_file()
            for c in chunk_file(p.relative_to(repo).as_posix(), p.read_text())]


def test_dataset_has_at_least_ten_examples():
    data = json.loads((ROOT / "dataset.json").read_text())
    assert len(data) >= 10
    for ex in data:
        assert ex["question"].strip() and ex["reference_answer"].strip() and ex["relevant"]


def test_dataset_ground_truth_exists_in_sample_repo():
    ids = sample_chunk_ids()
    data = json.loads((ROOT / "dataset.json").read_text())
    missing = [r for ex in data for r in ex["relevant"] if not any(matches(i, r) for i in ids)]
    assert missing == []
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: 2 passed. These tests check data that already exists, so they pass immediately. If `test_dataset_ground_truth_exists_in_sample_repo` fails, a sample file or dataset id was mistyped. Fix the data, not the test.

- [ ] **Step 5: Commit**

```bash
git add eval/ tests/test_dataset.py
git commit -m "feat: add sample codebase and 14-example evaluation dataset"
```

---

### Task 3: Evaluation runner

**Files:**
- Create: `app/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `RAGPipeline` (`.retrieve`, `.llm`, `.index_files`), metrics from Task 1, `LLMError`, and `JudgeScore` (`model_dump()`, `.overall`).
- Produces: `EvalExample(question: str, relevant: list[str], reference_answer: str)`; `DATASET_PATH`, `SAMPLE_REPO` (`Path`s); `load_dataset(path=DATASET_PATH) -> list[EvalExample]`; `load_sample_repo(root=SAMPLE_REPO) -> list[tuple[str, str]]`; `run_evaluation(pipeline, examples, k=5, use_judge=True) -> dict` with this shape:
```
{"k": int, "num_examples": int,
 "retrieval": {"precision_at_k": float, "recall_at_k": float, "mrr": float},
 "generation": None | {"faithfulness": float, "relevance": float, "correctness": float,
                       "overall": float, "judged": int, "errors": int},
 "results": [{"question", "relevant", "retrieved": [ids], "precision_at_k", "recall_at_k",
              "reciprocal_rank", "answer"?, "judge"? {faithfulness, relevance, correctness,
              reasoning, overall}, "error"?}]}
```
  `generation` is `None` when `use_judge` is false or every judge call failed.

- [ ] **Step 1: Write the failing tests** (`tests/test_evaluation.py`)

```python
import pytest

from app.evaluation import EvalExample, load_dataset, load_sample_repo, run_evaluation
from app.llm import LLMError
from app.rag import RAGPipeline
from tests.conftest import FakeLLM


@pytest.fixture
def indexed(store):
    def make(llm):
        pipeline = RAGPipeline(store, llm)
        pipeline.index_files(load_sample_repo())
        return pipeline
    return make


def test_load_helpers():
    examples = load_dataset()
    assert len(examples) >= 10 and isinstance(examples[0], EvalExample)
    paths = [p for p, _ in load_sample_repo()]
    assert "auth.py" in paths and "utils.ts" in paths


def test_retrieval_only_scores_dataset(indexed):
    llm = FakeLLM()
    report = run_evaluation(indexed(llm), load_dataset(), k=5, use_judge=False)
    assert report["k"] == 5 and report["num_examples"] == len(load_dataset())
    assert report["generation"] is None
    assert llm.answer_calls == [] and llm.judge_calls == []
    r = report["retrieval"]
    assert 0 < r["precision_at_k"] <= 1 and 0 < r["recall_at_k"] <= 1
    assert r["mrr"] >= 0.6  # sanity floor for all-MiniLM-L6-v2 on this dataset
    row = report["results"][0]
    assert len(row["retrieved"]) == 5 and "answer" not in row


def test_judge_scores_are_aggregated(indexed):
    llm = FakeLLM()
    examples = load_dataset()[:3]
    report = run_evaluation(indexed(llm), examples, k=3, use_judge=True)
    assert report["generation"] == {"faithfulness": 4, "relevance": 5, "correctness": 3,
                                    "overall": pytest.approx(12 / 15), "judged": 3, "errors": 0}
    row = report["results"][0]
    assert row["answer"].startswith("fake answer")
    assert row["judge"]["reasoning"] == "fake"
    assert len(llm.judge_calls) == 3
    assert llm.judge_calls[0][3] == examples[0].reference_answer


def test_judge_failures_are_recorded_per_row(indexed):
    report = run_evaluation(indexed(FakeLLM(error=LLMError("no key"))), load_dataset()[:2], k=3)
    assert report["generation"] is None
    assert all(row["error"] == "no key" for row in report["results"])
    assert report["retrieval"]["mrr"] > 0


def test_run_evaluation_rejects_empty_dataset(indexed):
    with pytest.raises(ValueError):
        run_evaluation(indexed(FakeLLM()), [], k=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.evaluation'`

- [ ] **Step 3: Implement `app/evaluation.py`**

```python
"""Evaluation suite: retrieval metrics per example, plus optional LLM-as-judge generation scores."""
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from app.llm import LLMError
from app.metrics import precision_at_k, recall_at_k, reciprocal_rank

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
DATASET_PATH = EVAL_DIR / "dataset.json"
SAMPLE_REPO = EVAL_DIR / "sample_repo"
JUDGE_WORKERS = 4  # parallel LLM calls; each example needs an answer call and a judge call


@dataclass(frozen=True)
class EvalExample:
    question: str
    relevant: list[str]
    reference_answer: str


def load_dataset(path: Path = DATASET_PATH) -> list[EvalExample]:
    data = json.loads(Path(path).read_text())
    return [EvalExample(e["question"], e["relevant"], e["reference_answer"]) for e in data]


def load_sample_repo(root: Path = SAMPLE_REPO) -> list[tuple[str, str]]:
    return [(p.relative_to(root).as_posix(), p.read_text()) for p in sorted(root.rglob("*")) if p.is_file()]


def _evaluate_example(pipeline, ex: EvalExample, k: int, use_judge: bool) -> dict:
    results = pipeline.retrieve(ex.question, k)
    ids = [r.chunk.id for r in results]
    row = {
        "question": ex.question,
        "relevant": ex.relevant,
        "retrieved": ids,
        "precision_at_k": precision_at_k(ids, ex.relevant, k),
        "recall_at_k": recall_at_k(ids, ex.relevant, k),
        "reciprocal_rank": reciprocal_rank(ids, ex.relevant),
    }
    if use_judge:
        chunks = [r.chunk for r in results]
        try:
            answer = pipeline.llm.answer(ex.question, chunks)
            score = pipeline.llm.judge(ex.question, answer, chunks, ex.reference_answer)
            row["answer"] = answer
            row["judge"] = {**score.model_dump(), "overall": score.overall}
        except LLMError as e:
            row["error"] = str(e)
    return row


def run_evaluation(pipeline, examples: list[EvalExample], k: int = 5, use_judge: bool = True) -> dict:
    if not examples:
        raise ValueError("evaluation dataset is empty")
    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        rows = list(pool.map(lambda ex: _evaluate_example(pipeline, ex, k, use_judge), examples))
    judged = [row["judge"] for row in rows if "judge" in row]
    generation = None
    if judged:
        generation = {key: mean(j[key] for j in judged)
                      for key in ("faithfulness", "relevance", "correctness", "overall")}
        generation |= {"judged": len(judged), "errors": sum("error" in row for row in rows)}
    return {
        "k": k,
        "num_examples": len(rows),
        "retrieval": {
            "precision_at_k": mean(row["precision_at_k"] for row in rows),
            "recall_at_k": mean(row["recall_at_k"] for row in rows),
            "mrr": mean(row["reciprocal_rank"] for row in rows),
        },
        "generation": generation,
        "results": rows,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: 5 passed. If `test_retrieval_only_scores_dataset` fails only because of the `mrr >= 0.6` floor, print the per-row `retrieved` ids and report the actual MRR. Do not change the dataset to game the metric, and do not lower the floor below the observed value minus 0.05.

- [ ] **Step 5: Commit**

```bash
git add app/evaluation.py tests/test_evaluation.py
git commit -m "feat: add evaluation runner with retrieval metrics and LLM-as-judge"
```

---

### Task 4: `POST /evaluate` endpoint

**Files:**
- Modify: `app/main.py`: add the eval pipeline and the endpoint inside `create_app`
- Test: `tests/test_api_evaluate.py`

**Interfaces:**
- Consumes: `run_evaluation`, `load_dataset`, `load_sample_repo`, `RAGPipeline`, `VectorStore`.
- Produces: `POST /evaluate` with an optional JSON body `{"k": int 1–20 = 5, "use_judge": bool = true}`. It returns the `run_evaluation` dict. `app.state.eval_pipeline` uses Chroma collection `eval`.

- [ ] **Step 1: Write the failing tests** (`tests/test_api_evaluate.py`)

```python
import pytest
from fastapi.testclient import TestClient

from app.llm import LLMError
from app.main import create_app
from tests.conftest import FakeLLM


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path), FakeLLM()))


def test_evaluate_endpoint_default_body(client):
    r = client.post("/evaluate")
    assert r.status_code == 200
    body = r.json()
    assert body["k"] == 5 and body["num_examples"] >= 10
    assert set(body["retrieval"]) == {"precision_at_k", "recall_at_k", "mrr"}
    assert body["generation"]["judged"] == body["num_examples"]


def test_evaluate_endpoint_retrieval_only(client):
    body = client.post("/evaluate", json={"k": 3, "use_judge": False}).json()
    assert body["k"] == 3 and body["generation"] is None


def test_evaluate_does_not_touch_user_index(client):
    client.post("/evaluate", json={"use_judge": False})
    assert client.get("/health").json()["indexed_chunks"] == 0


def test_evaluate_endpoint_is_repeatable(client):
    first = client.post("/evaluate", json={"use_judge": False}).json()
    count_after_first = client.app.state.eval_pipeline.store.count()
    second = client.post("/evaluate", json={"use_judge": False}).json()
    assert first["retrieval"] == second["retrieval"]
    assert client.app.state.eval_pipeline.store.count() == count_after_first


def test_evaluate_endpoint_judge_failure_keeps_retrieval_metrics(tmp_path):
    client = TestClient(create_app(str(tmp_path), FakeLLM(error=LLMError("no key"))))
    r = client.post("/evaluate")
    assert r.status_code == 200
    body = r.json()
    assert body["generation"] is None
    assert body["retrieval"]["mrr"] > 0
    assert all(row["error"] == "no key" for row in body["results"])


def test_evaluate_validation(client):
    assert client.post("/evaluate", json={"k": 0}).status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api_evaluate.py -v`
Expected: FAIL with 404 / `AttributeError: 'State' object has no attribute 'eval_pipeline'`

- [ ] **Step 3: Modify `app/main.py`**

Add these imports next to the existing ones:
```python
from app.evaluation import load_dataset, load_sample_repo, run_evaluation
```
Add this request model below `QueryRequest`:
```python
class EvaluateRequest(BaseModel):
    k: int = Field(5, ge=1, le=20)
    use_judge: bool = True
```
Inside `create_app`, right after `app.state.pipeline = pipeline`:
```python
    eval_pipeline = RAGPipeline(VectorStore(chroma_dir, "eval"), llm)
    app.state.eval_pipeline = eval_pipeline
```
Add this endpoint inside `create_app`, before `return app`:
```python
    @app.post("/evaluate")
    def evaluate(req: EvaluateRequest | None = None):
        req = req or EvaluateRequest()
        eval_pipeline.index_files(load_sample_repo())  # idempotent: re-indexing replaces chunks
        return run_evaluation(eval_pipeline, load_dataset(), req.k, req.use_judge)
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api_evaluate.py
git commit -m "feat: add POST /evaluate endpoint"
```
