from app.chunker import Chunk
from app.store import VectorStore

PW = Chunk("auth.py", "hash_password", 1, 3,
           "def hash_password(pw):\n    return pbkdf2_hmac('sha256', pw, salt, 100000)", "python")
SLUG = Chunk("text.ts", "L1-3", 1, 3,
             "export function slugify(s) { return s.toLowerCase().replace(/ /g, '-'); }", "typescript")


def test_search_returns_most_similar_first(tmp_path):
    store = VectorStore(str(tmp_path))
    store.add([PW, SLUG])
    results = store.search("how are passwords hashed", k=2)
    assert [r.chunk.id for r in results] == ["auth.py::hash_password", "text.ts::L1-3"]
    assert results[0].chunk == PW
    assert results[0].score > results[1].score


def test_upsert_same_id_does_not_duplicate(tmp_path):
    store = VectorStore(str(tmp_path))
    store.add([PW])
    store.add([PW])
    assert store.count() == 1


def test_delete_path_removes_only_that_file(tmp_path):
    store = VectorStore(str(tmp_path))
    store.add([PW, SLUG])
    store.delete_path("auth.py")
    assert [r.chunk.id for r in store.search("anything", k=5)] == ["text.ts::L1-3"]
    store.delete_path("missing.py")  # no error for an unknown path


def test_search_empty_store_returns_nothing(tmp_path):
    assert VectorStore(str(tmp_path)).search("q", k=5) == []


def test_search_k_larger_than_count(tmp_path):
    store = VectorStore(str(tmp_path))
    store.add([PW])
    assert len(store.search("q", k=20)) == 1


def test_persists_across_instances_and_collections_are_isolated(tmp_path):
    VectorStore(str(tmp_path)).add([PW])
    assert VectorStore(str(tmp_path)).count() == 1
    assert VectorStore(str(tmp_path), "other").count() == 0


def test_add_empty_list_is_noop(tmp_path):
    store = VectorStore(str(tmp_path))
    store.add([])
    assert store.count() == 0
