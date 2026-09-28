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
