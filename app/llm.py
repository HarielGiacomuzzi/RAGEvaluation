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
        except anthropic.AnthropicError as e:
            raise LLMError(f"LLM request failed: {e}") from e
        except TypeError as e:  # SDK raises TypeError (not AnthropicError) when it finds no credentials
            if "authentication" not in str(e):
                raise
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
