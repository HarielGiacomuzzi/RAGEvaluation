"""Code-aware chunking: Python by function/class/method via ast, everything else by line windows."""
import ast
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

WINDOW = 40
OVERLAP = 10

LANGUAGES = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript",
    ".jsx": "javascript", ".go": "go", ".java": "java", ".rb": "ruby", ".rs": "rust",
    ".c": "c", ".cpp": "cpp", ".cs": "csharp", ".php": "php", ".md": "markdown",
}


@dataclass(frozen=True)
class Chunk:
    path: str
    symbol: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    content: str
    language: str

    @property
    def id(self) -> str:
        return f"{self.path}::{self.symbol}"


def detect_language(path: str) -> str:
    return LANGUAGES.get(PurePosixPath(path).suffix.lower(), "text")


def chunk_file(path: str, text: str) -> list[Chunk]:
    if not text.strip():
        return []
    language = detect_language(path)
    if language == "python":
        try:
            return _dedupe(_chunk_python(path, text))
        except (SyntaxError, ValueError, RecursionError):
            pass
    return _dedupe(_chunk_lines(path, text, language))


def _chunk_python(path: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    tree = ast.parse(text)
    chunks: list[Chunk] = []
    covered: set[int] = set()

    def span(node) -> tuple[int, int]:
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        return start, node.end_lineno

    def add(symbol: str, start: int, end: int) -> None:
        chunks.append(Chunk(path, symbol, start, end, "\n".join(lines[start - 1:end]), "python"))
        covered.update(range(start, end + 1))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name, *span(node))
        elif isinstance(node, ast.ClassDef):
            start, end = span(node)
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            header_end = span(methods[0])[0] - 1 if methods else end
            while header_end > start and not lines[header_end - 1].strip():
                header_end -= 1
            add(node.name, start, header_end)
            for method in methods:
                add(f"{node.name}.{method.name}", *span(method))

    rest = [i for i in range(1, len(lines) + 1) if i not in covered and lines[i - 1].strip()]
    if rest:
        module = "\n".join(lines[i - 1] for i in rest)
        chunks.insert(0, Chunk(path, "<module>", rest[0], rest[-1], module, "python"))
    return chunks


def _chunk_lines(path: str, text: str, language: str) -> list[Chunk]:
    lines = text.splitlines()
    chunks = []
    for start in range(0, len(lines), WINDOW - OVERLAP):
        window = lines[start:start + WINDOW]
        end = start + len(window)
        if any(line.strip() for line in window):
            chunks.append(Chunk(path, f"L{start + 1}-{end}", start + 1, end, "\n".join(window), language))
        if end >= len(lines):
            break
    return chunks


def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
    seen: dict[str, int] = {}
    out = []
    for chunk in chunks:
        seen[chunk.symbol] = seen.get(chunk.symbol, 0) + 1
        n = seen[chunk.symbol]
        out.append(chunk if n == 1 else replace(chunk, symbol=f"{chunk.symbol}#{n}"))
    return out
