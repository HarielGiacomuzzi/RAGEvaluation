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


def test_index_python_file_with_nul_byte_falls_back(client):
    r = upload(client, ("weird.py", b"x = 1\x00\n", "text/x-python"))
    assert r.status_code == 200
    assert r.json()["chunks_indexed"] == 1
