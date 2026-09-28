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
    assert {call[3] for call in llm.judge_calls} == {ex.reference_answer for ex in examples}


def test_judge_failures_are_recorded_per_row(indexed):
    report = run_evaluation(indexed(FakeLLM(error=LLMError("no key"))), load_dataset()[:2], k=3)
    assert report["generation"] is None
    assert all(row["error"] == "no key" for row in report["results"])
    assert report["retrieval"]["mrr"] > 0


def test_run_evaluation_rejects_empty_dataset(indexed):
    with pytest.raises(ValueError):
        run_evaluation(indexed(FakeLLM()), [], k=5)
