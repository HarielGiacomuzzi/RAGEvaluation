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


def test_missing_credentials_type_error_is_wrapped():
    err = TypeError(
        "Could not resolve authentication method. Expected one of api_key, "
        "auth_token, or credentials to be set."
    )
    with pytest.raises(LLMError, match="LLM request failed"):
        ClaudeLLM(client=FakeClient(error=err)).answer("q", [CHUNK])


def test_unrelated_type_error_propagates():
    err = TypeError("unexpected keyword argument 'x'")
    with pytest.raises(TypeError):
        ClaudeLLM(client=FakeClient(error=err)).answer("q", [CHUNK])
