import json
from pathlib import Path

from app.chunker import chunk_file
from app.metrics import matches

ROOT = Path(__file__).resolve().parent.parent / "eval"


def sample_chunk_ids():
    repo = ROOT / "sample_repo"
    return [c.id for p in sorted(repo.rglob("*")) if p.is_file()
            for c in chunk_file(p.relative_to(repo).as_posix(), p.read_text())]


def test_dataset_has_at_least_ten_examples():
    data = json.loads((ROOT / "dataset.json").read_text())
    assert len(data) >= 10
    for ex in data:
        assert ex["question"].strip() and ex["reference_answer"].strip() and ex["relevant"]


def test_dataset_ground_truth_exists_in_sample_repo():
    ids = sample_chunk_ids()
    data = json.loads((ROOT / "dataset.json").read_text())
    missing = [r for ex in data for r in ex["relevant"] if not any(matches(i, r) for i in ids)]
    assert missing == []
