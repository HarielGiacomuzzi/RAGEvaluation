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
