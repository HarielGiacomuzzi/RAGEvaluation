# Evaluation

The evaluation suite measures two things: whether the retriever finds the
right code (retrieval metrics) and whether the generated answer is any good
(LLM-as-judge). Both run against the bundled sample codebase (`eval/sample_repo`)
and a hand-labeled question set (`eval/dataset.json`).

## Dataset format

`eval/dataset.json` is a JSON array of examples:

```json
{
  "question": "How are user passwords hashed?",
  "relevant": ["auth.py::hash_password"],
  "reference_answer": "hash_password uses PBKDF2-HMAC-SHA256 (hashlib.pbkdf2_hmac) with 200,000 iterations and a random 16-byte salt from secrets.token_bytes, returning a (salt, digest) tuple."
}
```

- `question` — the natural-language question to ask the RAG pipeline.
- `relevant` — the ground-truth list of chunk ids (or bare file paths, for
  non-Python files) that should be retrieved.
- `reference_answer` — the answer the judge compares generated answers
  against for the `correctness` criterion.

The dataset currently has 14 examples covering `auth.py`, `inventory.py`,
`orders.py`, and `pricing.py` (symbol-level ground truth, since Python is
chunked by `ast`) and `utils.ts` (path-level ground truth, since TypeScript
is chunked by line window and there's no stable symbol id to target).

## Relevance matching

A retrieved chunk id is considered a match for a ground-truth item if
(`app/metrics.py`, `matches`):

- the chunk id equals the item exactly (`"auth.py::hash_password" == "auth.py::hash_password"`), or
- the chunk id's path segment equals the item (`"utils.ts::L1-40".split("::", 1)[0] == "utils.ts"`).

This lets Python examples target an exact chunk while TypeScript examples
target the whole file, since its line-window chunks don't map to stable
symbols.

## Metrics

Let `retrieved` be the ordered list of chunk ids returned for a question at
cutoff `k`, and `relevant` the ground-truth list for that question.

**Precision@K** — fraction of the top-K retrieved chunks that are relevant:

```
precision@k = |{c in retrieved[:k] : matches(c, any r in relevant)}| / k
```

**Recall@K** — fraction of ground-truth items found somewhere in the top-K:

```
recall@k = |{r in relevant : matches(any c in retrieved[:k], r)}| / |relevant|
```

**MRR (Mean Reciprocal Rank)** — the mean, across questions, of
`reciprocal_rank`, where `reciprocal_rank = 1 / rank` of the first relevant
chunk in `retrieved` (0 if none match). It rewards ranking the right answer
first, not just including it somewhere in the top-K.

### Worked example

For "How are user passwords hashed?" (`relevant = ["auth.py::hash_password"]`),
a real local run at `k=5` retrieved:

```
1. auth.py::hash_password          <- match
2. auth.py::verify_password
3. auth.py::<module>
4. auth.py::generate_token
5. inventory.py::Inventory.low_stock_items
```

- `precision@5 = 1/5 = 0.2` (1 of the top 5 slots is relevant).
- `recall@5 = 1/1 = 1.0` (the one ground-truth item was found).
- `reciprocal_rank = 1/1 = 1.0` (the match was the very first result).

## Measured results

Run locally with `curl -X POST localhost:8000/evaluate -H 'Content-Type: application/json' -d '{"use_judge": false}'`
after indexing nothing manually (`/evaluate` re-indexes `eval/sample_repo`
itself), `k=5`:

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

Precision is low by construction — most examples have exactly one relevant
chunk out of `k=5` results, so the theoretical ceiling for `precision@5` is
`0.2`-ish per example unless `relevant` has multiple items (a couple of
examples do). Recall of 1.0 and MRR of 0.96 mean the single relevant chunk
was retrieved, and almost always ranked first, for nearly every question.

`generation` is `null` here because `use_judge` was `false` (and separately,
because no `ANTHROPIC_API_KEY` was available in this environment — see below).

## LLM-as-judge

When `use_judge: true` (the default) and `ANTHROPIC_API_KEY` is set, each
example also gets: a generated answer (`ClaudeLLM.answer`, the same call path
as `/query`), then a judge call (`ClaudeLLM.judge`) that scores it against
`reference_answer` and the retrieved code.

**Criteria** (`JUDGE_SYSTEM` in `app/llm.py`), each scored 1 (very poor) to 5
(excellent):

- **faithfulness** — every claim in the answer is supported by the retrieved code.
- **relevance** — the answer addresses the question that was asked.
- **correctness** — the answer agrees with the reference answer.

**Structured output**: the judge call passes
`output_config = {"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}}`,
so Claude returns JSON matching:

```json
{
  "faithfulness": 4,
  "relevance": 5,
  "correctness": 4,
  "reasoning": "Brief explanation of the scores."
}
```

This is parsed into a `JudgeScore` pydantic model; each of the three scores
is clamped to `[1, 5]` as a defensive measure.

**Overall score**: `overall = (faithfulness + relevance + correctness) / 15`
— i.e. the mean of the three criteria normalized to a 0-1 scale (each
criterion maxes at 5, three criteria, so the max possible sum is 15).

At the run level, `generation` is the mean of `faithfulness`, `relevance`,
`correctness`, and `overall` across every example whose judge call succeeded,
plus `judged` (how many succeeded) and `errors` (how many failed — e.g. due
to a missing API key). If no example succeeds, `generation` is `null`.

## Adding examples

Append to the JSON array in `eval/dataset.json` with a `question`, one or
more `relevant` ids or paths, and a `reference_answer`. Two rules are
enforced by `tests/test_dataset.py`:

- There must be at least 10 examples, and every example needs a non-empty
  `question`, `reference_answer`, and `relevant` list.
- Every entry in every example's `relevant` list must actually match a chunk
  id produced by chunking `eval/sample_repo` (via `matches`) — so `relevant`
  entries for a new example must correspond to a symbol or path that really
  exists in the sample repo, or the test fails.

If the example targets a new file, add that file to `eval/sample_repo` first.

## Interpreting results

- **Low recall/MRR** → the retriever isn't finding the right code; check
  chunking (is the relevant symbol split oddly?) or the embedding query
  (does the question phrase align with how the code is worded?).
- **High recall/MRR but low judge `faithfulness`** → retrieval is fine but
  the LLM is asserting things the retrieved snippets don't support; check
  the `ANSWER_SYSTEM` prompt or context size (`k` too small to cover the
  answer).
- **Low judge `correctness`** with high `faithfulness`/`relevance` → the
  answer is grounded and on-topic but factually diverges from
  `reference_answer`; check whether the reference itself is still accurate,
  or whether the retrieved chunks are missing key context.
- **`errors` > 0 in `generation`** → some judge or answer calls failed (most
  commonly missing/invalid `ANTHROPIC_API_KEY`); `generation`'s averages are
  computed only over the rows that succeeded, so a partial-failure run can
  still report numbers, but ‘`judged`’ tells you how many contributed.
