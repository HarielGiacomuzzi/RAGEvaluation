"""FastAPI app factory: index and query endpoints. Run: uvicorn app.main:create_app --factory"""
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from app.evaluation import load_dataset, load_sample_repo, run_evaluation
from app.llm import ClaudeLLM, LLMError
from app.rag import RAGPipeline
from app.store import SearchResult, VectorStore

MAX_FILE_BYTES = 1_000_000


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=20)


class EvaluateRequest(BaseModel):
    k: int = Field(5, ge=1, le=20)
    use_judge: bool = True


def source_dict(result: SearchResult) -> dict:
    c = result.chunk
    return {"id": c.id, "path": c.path, "symbol": c.symbol, "start_line": c.start_line,
            "end_line": c.end_line, "language": c.language, "content": c.content,
            "score": round(result.score, 4)}


def create_app(chroma_dir: str | None = None, llm=None) -> FastAPI:
    chroma_dir = chroma_dir or os.environ.get("CHROMA_DIR", "./data/chroma")
    llm = llm or ClaudeLLM()
    app = FastAPI(title="Codebase RAG", description="Index code, ask questions, evaluate retrieval and answers.")
    pipeline = RAGPipeline(VectorStore(chroma_dir, "code"), llm)
    app.state.pipeline = pipeline
    eval_pipeline = RAGPipeline(VectorStore(chroma_dir, "eval"), llm)
    app.state.eval_pipeline = eval_pipeline

    @app.post("/index/files")
    async def index_files(files: list[UploadFile] = File(...)):
        accepted, skipped = [], []
        for upload in files:
            name = upload.filename or "unnamed"
            data = await upload.read()
            if len(data) > MAX_FILE_BYTES:
                skipped.append({"path": name, "reason": "file larger than 1 MB"})
                continue
            try:
                accepted.append((name, data.decode("utf-8")))
            except UnicodeDecodeError:
                skipped.append({"path": name, "reason": "not a UTF-8 text file"})
        result = await run_in_threadpool(pipeline.index_files, accepted)
        return {"files_indexed": result.files_indexed, "chunks_indexed": result.chunks_indexed,
                "total_chunks": pipeline.store.count(), "skipped": skipped}

    @app.post("/query")
    def query(req: QueryRequest):
        try:
            result = pipeline.query(req.question, req.k)
        except LLMError as e:
            raise HTTPException(status_code=502, detail=str(e))
        return {"answer": result.answer, "sources": [source_dict(s) for s in result.sources]}

    @app.get("/health")
    def health():
        return {"status": "ok", "indexed_chunks": pipeline.store.count()}

    @app.post("/evaluate")
    def evaluate(req: EvaluateRequest | None = None):
        req = req or EvaluateRequest()
        eval_pipeline.index_files(load_sample_repo())  # idempotent: re-indexing replaces chunks
        return run_evaluation(eval_pipeline, load_dataset(), req.k, req.use_judge)

    return app
