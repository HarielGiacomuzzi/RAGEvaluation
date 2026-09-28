FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY eval ./eval

# Bake the all-MiniLM-L6-v2 ONNX embedding model into the image so the first request is fast.
RUN uv run --no-sync python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warmup'])"

ENV CHROMA_DIR=/data/chroma PORT=8000 ANONYMIZED_TELEMETRY=False
EXPOSE 8000
CMD ["sh", "-c", "uv run --no-sync uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT}"]
