"""Persistent ChromaDB vector store for code chunks (local all-MiniLM-L6-v2 ONNX embeddings)."""
from dataclasses import dataclass

import chromadb

from app.chunker import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity, higher is better


class VectorStore:
    def __init__(self, persist_dir: str, collection: str = "code"):
        client = chromadb.PersistentClient(path=persist_dir)
        self._col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.id for c in chunks],
            # Path and symbol are prepended so the embedding sees file and function names.
            documents=[f"# {c.path} :: {c.symbol}\n{c.content}" for c in chunks],
            metadatas=[
                {"path": c.path, "symbol": c.symbol, "start_line": c.start_line,
                 "end_line": c.end_line, "language": c.language, "content": c.content}
                for c in chunks
            ],
        )

    def delete_path(self, path: str) -> None:
        self._col.delete(where={"path": path})

    def count(self) -> int:
        return self._col.count()

    def search(self, query: str, k: int) -> list[SearchResult]:
        n = min(k, self.count())
        if n <= 0:
            return []
        result = self._col.query(query_texts=[query], n_results=n, include=["metadatas", "distances"])
        return [
            SearchResult(
                Chunk(m["path"], m["symbol"], m["start_line"], m["end_line"], m["content"], m["language"]),
                1 - d,
            )
            for m, d in zip(result["metadatas"][0], result["distances"][0])
        ]
