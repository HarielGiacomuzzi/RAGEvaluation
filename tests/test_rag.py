from app.rag import NO_INDEX_ANSWER, RAGPipeline
from tests.conftest import FakeLLM

AUTH = "def hash_password(pw):\n    return pbkdf2(pw)\n\n\ndef make_token():\n    return secrets.token_hex()\n"
UTIL = "export function slugify(s: string) {\n  return s.toLowerCase();\n}\n"


def test_index_files_counts_files_and_chunks(store):
    result = RAGPipeline(store, FakeLLM()).index_files([("auth.py", AUTH), ("util.ts", UTIL)])
    assert (result.files_indexed, result.chunks_indexed) == (2, 3)
    assert store.count() == 3


def test_reindex_same_path_replaces_chunks(store):
    rag = RAGPipeline(store, FakeLLM())
    rag.index_files([("auth.py", AUTH)])
    rag.index_files([("auth.py", "def only_one():\n    pass\n")])
    assert [r.chunk.id for r in rag.retrieve("anything", k=10)] == ["auth.py::only_one"]


def test_query_retrieves_then_generates(store):
    llm = FakeLLM()
    rag = RAGPipeline(store, llm)
    rag.index_files([("auth.py", AUTH), ("util.ts", UTIL)])
    result = rag.query("how is a password hashed?", k=2)
    assert len(result.sources) == 2
    assert result.sources[0].chunk.id == "auth.py::hash_password"
    question, chunks = llm.answer_calls[0]
    assert question == "how is a password hashed?"
    assert [c.id for c in chunks] == [s.chunk.id for s in result.sources]
    assert result.answer.startswith("fake answer from auth.py::hash_password")


def test_query_empty_index_skips_llm(store):
    llm = FakeLLM()
    result = RAGPipeline(store, llm).query("anything?")
    assert result.answer == NO_INDEX_ANSWER
    assert result.sources == []
    assert llm.answer_calls == []
