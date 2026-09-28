import pytest

from app.metrics import matches, precision_at_k, recall_at_k, reciprocal_rank

RETRIEVED = ["a.py::f", "b.py::g", "c.ts::L1-40", "a.py::h"]


def test_matches_exact_id_or_path():
    assert matches("a.py::f", "a.py::f")
    assert matches("c.ts::L1-40", "c.ts")
    assert not matches("a.py::f", "a.py::h")
    assert not matches("a.py::f", "a.p")


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, ["a.py::f", "a.py::h"], 2) == 0.5
    assert precision_at_k(RETRIEVED, ["a.py::f", "a.py::h"], 4) == 0.5
    assert precision_at_k(RETRIEVED, ["c.ts"], 3) == pytest.approx(1 / 3)


def test_precision_divides_by_k_even_if_fewer_retrieved():
    assert precision_at_k(["a.py::f"], ["a.py::f"], 5) == 0.2


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, ["a.py::f", "a.py::h"], 2) == 0.5
    assert recall_at_k(RETRIEVED, ["a.py::f", "a.py::h"], 4) == 1.0
    assert recall_at_k(RETRIEVED, ["zzz.py::x"], 4) == 0.0
    assert recall_at_k(RETRIEVED, [], 4) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, ["a.py::f"]) == 1.0
    assert reciprocal_rank(RETRIEVED, ["c.ts"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(RETRIEVED, ["nope"]) == 0.0
    assert reciprocal_rank([], ["a.py::f"]) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(RETRIEVED, ["a.py::f"], 0)
    with pytest.raises(ValueError):
        recall_at_k(RETRIEVED, ["a.py::f"], -1)
