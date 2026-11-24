import ast
import time
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_chunking import (
    AstCodeChunker,
    AstCodeConfig,
    CodeSeparatorChunker,
    CodeSeparatorConfig,
    StructureParseError,
    register_builtin_plugins,
    render,
)
from rag_foundry_techniques import ChunkInput, TechniqueRegistry, load_catalogue

ROOT = Path(__file__).resolve().parents[2]
SOURCE = '''import os

# Storage helpers.

class Store:
    """Store docs."""

    def add_item(self, item: dict[str, int]) -> list[int]:
        # keep the order
        values = [item["a"], item["b"]]
        return values

    @property
    def count(self) -> int:
        total = len(os.sep)
        return total
    # trailing comment in the class

@decorator
def first():
    return 1

def second(): return 2
# end of file
'''


def _c13(text, **kwargs):
    return CodeSeparatorChunker(CodeSeparatorConfig(**kwargs)).chunk_structured(ChunkInput(text=text)).chunks


def _c14(text, **kwargs):
    return AstCodeChunker(AstCodeConfig(**kwargs)).chunk_structured(ChunkInput(text=text)).chunks


def _invariants(text, chunks, size=None):
    assert all(0 <= c.start < c.end <= len(text) for c in chunks)
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:], strict=False))
    assert all(not text[c.start].isspace() and not text[c.end - 1].isspace() for c in chunks)
    covered = [i for c in chunks for i in range(c.start, c.end) if not text[i].isspace()]
    assert covered == [i for i, char in enumerate(text) if not char.isspace()]
    if size is not None:
        assert all(c.end - c.start <= size for c in chunks if "split_part" in c.flags)


def _at_line_start(text, start):
    previous_break = max(text.rfind("\n", 0, start), text.rfind("\r", 0, start))
    return not text[previous_break + 1 : start].strip()


def test_c13_python_declarations_decorators_and_paths():
    chunks = _c13(SOURCE, chunk_size=64)
    assert all(_at_line_start(SOURCE, c.start) for c in chunks)
    count = next(c for c in chunks if "def count" in SOURCE[c.start : c.end])
    assert SOURCE[count.start : count.end].startswith("@property")
    add = next(c for c in chunks if "def add_item" in SOURCE[c.start : c.end])
    assert add.path == ("class Store:", "def add_item(self, item: dict[str, int]) -> list[int]:")
    assert add.attr("language") == "python"
    first = next(c for c in chunks if "def first" in SOURCE[c.start : c.end])
    assert SOURCE[first.start : first.end].startswith("@decorator") and first.path[0] == "def first():"
    _invariants(SOURCE, chunks, 64)


def test_c13_javascript_cuts_before_declarations():
    text = "const x = 1;\n// one\nfunction one() { return x; }\nclass Two {}\nexport function three() {}"
    chunks = _c13(text, language="javascript", chunk_size=36)
    bodies = [text[c.start : c.end] for c in chunks]
    assert bodies == [
        "const x = 1;",
        "// one\nfunction one() { return x; }",
        "class Two {}",
        "export function three() {}",
    ]
    _invariants(text, chunks, 40)


def test_c13_oversized_line_keep_and_split():
    long_line = "const_value = '" + "x " * 100 + "'"
    assert [c.flags for c in _c13(long_line, chunk_size=32, oversized="keep")] == [("oversized",)]
    split = _c13(long_line, chunk_size=32)
    assert len(split) > 1 and all(c.flags == ("oversized", "split_part") for c in split)
    _invariants(long_line, split, 32)


def _node_boundaries(source):
    """Start and end offsets of every statement, computed independently with ast (ASCII source)."""
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    bounds = set()

    def visit(statements):
        for node in statements:
            first = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", ()))])
            line = source[starts[first - 1] : starts[first]]
            bounds.add(starts[first - 1] + len(line) - len(line.lstrip()))
            bounds.add(starts[node.end_lineno - 1] + node.end_col_offset)
            visit(getattr(node, "body", []))

    visit(ast.parse(source).body)
    return bounds


def test_c14_paths_context_docstring_and_node_boundaries():
    chunks = _c14(SOURCE, chunk_size=80)
    add = [c for c in chunks if c.path == ("Store", "add_item")]
    assert add and add[0].attr("node_type") == "FunctionDef" and add[0].attr("qualname") == "Store.add_item"
    assert [SOURCE[s.start : s.end] for s in add[0].context] == ["class Store:"]
    assert all(s.role == "declaration" for s in add[0].context)
    assert render(SOURCE, add[0]).startswith("class Store:\n# keep the order") or render(SOURCE, add[0]).startswith(
        "class Store:\ndef add_item"
    )
    header = next(c for c in chunks if c.path == ("Store",))
    # The comment block directly above the class joins it (text between statements joins the next one).
    assert SOURCE[header.start : header.end] == '# Storage helpers.\n\nclass Store:\n    """Store docs."""'
    assert header.attr("node_type") == "ClassDef"
    # Every cut is a statement boundary (comments and blank lines join the next statement).
    bounds = _node_boundaries(SOURCE)
    for chunk in chunks:
        body = SOURCE[chunk.start : chunk.end]
        if not body.startswith("#"):
            assert chunk.start in bounds, body
        header = chunk.attr("node_type") in ("ClassDef", "FunctionDef") and body.endswith((":", '"""'))
        assert chunk.end in bounds or header or chunk is chunks[-1], body
    _invariants(SOURCE, chunks, 80)


def test_c14_keeps_signature_annotations_comments_and_one_liners():
    source = "def f(x: int) -> dict[str, int]:\n    a = 1\n    # inside\n    return {'k': x + a}\n# end\n"
    chunks = _c14(source, chunk_size=32)
    assert source[chunks[0].start : chunks[0].end] == "def f(x: int) -> dict[str, int]:"
    assert all([source[s.start : s.end] for s in c.context] == ["def f(x: int) -> dict[str, int]:"] for c in chunks[1:])
    _invariants(source, chunks, 32)
    one_liner = "def g(): return " + " + ".join(["1"] * 30)
    chunks = _c14(one_liner, chunk_size=32)
    assert one_liner[chunks[0].start : chunks[0].end] == "def g():"
    _invariants(one_liner, chunks, 32)
    assert _c14("# only a comment\n") and _c14("# only a comment\n")[0].path == ()


def test_c14_unicode_offsets_crlf_and_cr():
    source = (
        "def caf\u00e9(\u0646\u0635='\u0627\u0644\u0639\u0631\u0628\u064a\u0629'):\r\n"
        "    value = '\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645'\r\n"
        "    return value + \u0646\u0635\r\n"
        "x = caf\u00e9()\r\n"
    )
    chunks = _c14(source, chunk_size=40)
    assert (
        source[chunks[0].start : chunks[0].end]
        == "def caf\u00e9(\u0646\u0635='\u0627\u0644\u0639\u0631\u0628\u064a\u0629'):"
    )
    assert [source[c.start : c.end] for c in chunks[1:]] == [
        "value = '\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645'",
        "return value + \u0646\u0635",
        "x = caf\u00e9()",
    ]
    _invariants(source, chunks, 40)
    cr_only = source.replace("\r\n", "\r")
    assert [cr_only[c.start : c.end] for c in _c14(cr_only, chunk_size=40)] == [source[c.start : c.end] for c in chunks]


def test_c14_errors_are_explicit():
    with pytest.raises(StructureParseError) as error:
        _c14("x = 1\ndef broken(:\n")
    assert error.value.code == "syntax_error" and "line 2" in str(error.value)
    with pytest.raises(StructureParseError) as error:
        _c14("x = 1\x00")
    assert error.value.code == "syntax_error"
    fallback = _c14("def broken(:\n    " + "y " * 40, on_syntax_error="fallback", chunk_size=32)
    assert fallback and all("unparsed" in c.flags for c in fallback)
    assert any(c.flags == ("oversized", "split_part", "unparsed") for c in fallback)
    with pytest.raises(StructureParseError) as error:
        _c14("func main() {}", language="go")
    assert error.value.code == "unsupported_language"


def test_c14_oversized_statement_keep_and_split():
    source = "import os\nvalue = '" + "x " * 1500 + "'"
    kept = _c14(source, chunk_size=64, oversized="keep")
    assert [c.flags for c in kept] == [(), ("oversized",)] and kept[1].attr("node_type") == "Assign"
    split = _c14(source, chunk_size=64)
    assert len(split) > 10 and all(c.flags == ("oversized", "split_part") for c in split[1:])
    _invariants(source, split, 64)


def test_chunk_matches_structured_determinism_and_registry():
    doc = ChunkInput(text=SOURCE)
    for chunker in (CodeSeparatorChunker(CodeSeparatorConfig(chunk_size=64)), AstCodeChunker(AstCodeConfig())):
        assert chunker.chunk(doc) == tuple(c.to_span() for c in chunker.chunk_structured(doc).chunks)
        assert chunker.chunk_structured(doc) == chunker.chunk_structured(doc)
    registry = TechniqueRegistry(load_catalogue(ROOT / "config" / "techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C13", {}) == CodeSeparatorConfig()
    assert registry.validate_config("C14", {}) == AstCodeConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("C13", {"unknown": True})
    with pytest.raises(ValidationError):
        registry.validate_config("C14", {"unknown": True})
    with pytest.raises(ValidationError):
        CodeSeparatorConfig(language="cobol")


def test_million_code_points_stays_linear():
    unit = "class K:\n    def m(self):\n        # note\n        return 1\n\n"
    text = unit * (1_000_000 // len(unit) + 1)
    for chunker in (
        CodeSeparatorChunker(CodeSeparatorConfig()),
        CodeSeparatorChunker(CodeSeparatorConfig(chunk_size=40)),
        AstCodeChunker(AstCodeConfig()),
    ):
        started = time.perf_counter()
        result = chunker.chunk_structured(ChunkInput(text=text))
        assert time.perf_counter() - started < 10
        assert len(result.chunks) > 500
