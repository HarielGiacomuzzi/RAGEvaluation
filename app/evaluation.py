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
