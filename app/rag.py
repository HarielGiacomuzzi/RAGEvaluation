"""RAG pipeline: index (chunk, embed, store) and query (retrieve, generate)."""
from dataclasses import dataclass

from app.chunker import chunk_file
from app.store import SearchResult, VectorStore

NO_INDEX_ANSWER = "No code has been indexed yet. Upload files in the Indexing section first."


@dataclass(frozen=True)
class IndexResult:
    files_indexed: int
    chunks_indexed: int


@dataclass(frozen=True)
class QueryResult:
    answer: str
    sources: list[SearchResult]


class RAGPipeline:
    def __init__(self, store: VectorStore, llm):
        self.store = store
        self.llm = llm

    def index_files(self, files: list[tuple[str, str]]) -> IndexResult:
        total = 0
        for path, text in files:
            chunks = chunk_file(path, text)
            self.store.delete_path(path)  # re-indexing a file replaces its old chunks
            self.store.add(chunks)
            total += len(chunks)
        return IndexResult(len(files), total)

    def retrieve(self, question: str, k: int = 5) -> list[SearchResult]:
        return self.store.search(question, k)

    def query(self, question: str, k: int = 5) -> QueryResult:
        sources = self.retrieve(question, k)
        if not sources:
            return QueryResult(NO_INDEX_ANSWER, [])
        return QueryResult(self.llm.answer(question, [s.chunk for s in sources]), sources)
