from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakeLLM

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"


def test_index_page_served(tmp_path):
    client = TestClient(create_app(str(tmp_path), FakeLLM()))
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    for text in ("Indexing", "Querying", "Evaluation", 'name="viewport"'):
        assert text in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_frontend_never_uses_innerhtml():
    assert "innerHTML" not in (STATIC / "app.js").read_text()
