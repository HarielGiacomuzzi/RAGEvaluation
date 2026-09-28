"""Retrieval metrics. Ground-truth items are chunk ids ("path::symbol") or bare file paths."""


def matches(chunk_id: str, relevant_item: str) -> bool:
    return chunk_id == relevant_item or chunk_id.split("::", 1)[0] == relevant_item


def _is_relevant(chunk_id: str, relevant: list[str]) -> bool:
    return any(matches(chunk_id, r) for r in relevant)


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of the top-k slots holding a relevant chunk."""
    _check_k(k)
    return sum(_is_relevant(c, relevant) for c in retrieved[:k]) / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of ground-truth items matched by at least one top-k chunk."""
    _check_k(k)
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(any(matches(c, r) for c in top) for r in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1 / rank of the first relevant chunk, 0 if none; averaged over queries this is MRR."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if _is_relevant(chunk_id, relevant):
            return 1 / rank
    return 0.0
