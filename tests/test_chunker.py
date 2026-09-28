from app.chunker import Chunk, chunk_file, detect_language

PY = '''import os

LIMIT = 3


def top(a):
    return a + 1


class Greeter:
    """Says hello."""

    greeting = "hi"

    @staticmethod
    def hello(name):
        return f"hi {name}"

    def bye(self):
        return "bye"
'''


def ids(chunks):
    return [c.id for c in chunks]


def test_python_chunks_by_function_class_and_method():
    chunks = chunk_file("pkg/mod.py", PY)
    assert ids(chunks) == [
        "pkg/mod.py::<module>",
        "pkg/mod.py::top",
        "pkg/mod.py::Greeter",
        "pkg/mod.py::Greeter.hello",
        "pkg/mod.py::Greeter.bye",
    ]
    by_id = {c.id: c for c in chunks}
    assert "import os" in by_id["pkg/mod.py::<module>"].content
    assert "LIMIT = 3" in by_id["pkg/mod.py::<module>"].content
    assert by_id["pkg/mod.py::top"].start_line == 6
    assert by_id["pkg/mod.py::top"].end_line == 7
    assert 'greeting = "hi"' in by_id["pkg/mod.py::Greeter"].content
    assert by_id["pkg/mod.py::Greeter.hello"].content.lstrip().startswith("@staticmethod")
    assert all(c.language == "python" for c in chunks)


def test_python_syntax_error_falls_back_to_windows():
    chunks = chunk_file("broken.py", "def oops(:\n    pass\n")
    assert ids(chunks) == ["broken.py::L1-2"]


def test_generic_file_uses_overlapping_line_windows():
    text = "\n".join(f"line {i}" for i in range(1, 101))
    chunks = chunk_file("web/app.ts", text)
    assert ids(chunks) == ["web/app.ts::L1-40", "web/app.ts::L31-70", "web/app.ts::L61-100"]
    assert chunks[0].language == "typescript"
    assert chunks[1].content.splitlines()[0] == "line 31"


def test_short_generic_file_is_single_chunk():
    assert ids(chunk_file("a.js", "const a = 1;\n")) == ["a.js::L1-1"]


def test_empty_file_has_no_chunks():
    assert chunk_file("empty.py", "   \n\n") == []


def test_duplicate_symbols_get_suffix():
    chunks = chunk_file("d.py", "def f():\n    return 1\n\n\ndef f():\n    return 2\n")
    assert ids(chunks) == ["d.py::f", "d.py::f#2"]


def test_detect_language():
    assert detect_language("x/y.py") == "python"
    assert detect_language("Y.TSX") == "typescript"
    assert detect_language("README") == "text"


def test_chunk_id():
    assert Chunk("a.py", "f", 1, 2, "x", "python").id == "a.py::f"
