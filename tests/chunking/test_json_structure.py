import json
import time
import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_chunking import (
    JsonStructureChunker,
    JsonStructureConfig,
    StructureParseError,
    json_path_at,
    register_builtin_plugins,
    render,
)
from rag_foundry_chunking.json_structure import _walk
from rag_foundry_techniques import ChunkInput, TechniqueRegistry, load_catalogue

ROOT = Path(__file__).resolve().parents[2]
PUNCTUATION = set("{}[],:")


def _run(text, **config):
    return JsonStructureChunker(JsonStructureConfig(**config)).chunk_structured(ChunkInput(text=text))


def _invariants(text, chunks, size):
    assert all(0 <= c.start < c.end <= len(text) for c in chunks)
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:], strict=False))
    assert all(not text[c.start].isspace() and not text[c.end - 1].isspace() for c in chunks)
    assert all(c.end - c.start <= size for c in chunks if "split_part" in c.flags)
    covered = {i for c in chunks for i in range(c.start, c.end)}
    lost = [i for i, char in enumerate(text) if not char.isspace() and char not in PUNCTUATION and i not in covered]
    assert lost == [], text[lost[0] : lost[0] + 20]


def _store(count=30, indent=None):
    books = [{"title": f"Book {i}", "price": i} for i in range(count)]
    return json.dumps({"store": {"name": "Shelf", "books": books}}, indent=indent)


def test_nested_json_paths_context_and_value_lookup():
    text = _store()
    result = _run(text, chunk_size=120)
    assert any("$['store']['books'][3]" in chunk.path for chunk in result.chunks)
    price = text.index('"price": 12') + len('"price": ')
    assert json_path_at(text, result, price) == "$['store']['books'][12]['price']"
    chunk = next(c for c in result.chunks if "$['store']['books'][12]" in c.path)
    assert [text[s.start : s.end] for s in chunk.context] == ['"store":', '"books":']
    assert all(s.role == "container" for s in chunk.context)
    assert render(text, chunk).startswith('"store":\n"books":\n')
    assert chunk.attr("container_path") == "$['store']['books']"
    _invariants(text, result.chunks, 120)


def test_keys_of_split_containers_stay_in_a_chunk_body():
    text = _store(indent=2)
    chunks = _run(text, chunk_size=100).chunks
    first = chunks[0]
    assert text[first.start : first.end].startswith('"store": {\n    "name": "Shelf"')
    books = next(c for c in chunks if '"books"' in text[c.start : c.end])
    assert text[books.start : books.end].startswith('"books": [')
    assert [text[s.start : s.end] for s in books.context] == ['"store":']
    _invariants(text, chunks, 100)


def test_member_and_index_attrs_and_whole_document():
    text = '{"a": [1, 2, 3], "b": true, "c": "' + "x" * 40 + '"}'
    chunks = _run(text, chunk_size=32).chunks
    assert chunks[0].attr("first_key") == "a" and chunks[0].attr("container_path") == "$"
    for scalar in ("42", '"x"', "{}", "[]", '{"a": 1}'):
        (chunk,) = _run(scalar).chunks
        assert chunk.path == ("$",) and chunk.attrs == (("container_path", "$"),)


def test_paths_escape_keys():
    key = "a'b\\c"
    arabic = "السعر"
    assert unicodedata.is_normalized("NFC", arabic)
    text = json.dumps({key: 1, arabic: 2, "pad": "p" * 40}, ensure_ascii=False)
    paths = {p for c in _run(text, chunk_size=32).chunks for p in c.path}
    assert "$['a\\'b\\\\c']" in paths and f"$['{arabic}']" in paths
    assert json_path_at(text, None, text.index("2")) == f"$['{arabic}']"


def test_walker_member_and_element_value_ranges_are_json():
    text = '{"x": [1, {"y": true}], "z": null, "w": "\\u00e9"}'
    root = _walk(text, 64)
    expected = json.loads(text)

    def check(node, value):
        for entry in node.children or []:
            start = entry.value_start if entry.value_start is not None else entry.start
            child_value = value[entry.key] if entry.key is not None else value[entry.index]
            assert json.loads(text[start : entry.end]) == child_value
            check(entry, child_value)

    check(root, expected)


@pytest.mark.parametrize("text", ['{"a":}', '{"a": 1} trailing', "[NaN]", "[Infinity]", "[1,]", "{'a': 1}"])
def test_invalid_json_reports_code(text):
    with pytest.raises(StructureParseError) as error:
        _run(text)
    assert error.value.code == "invalid_json"


def test_too_deep():
    with pytest.raises(StructureParseError) as error:
        _run("[[[[0]]]]", max_depth=2)
    assert error.value.code == "too_deep"
    assert _run("[[[[0]]]]", max_depth=4).chunks


def test_long_scalar_keep_and_split():
    text = '{"id": 7, "value": "' + "word " * 1000 + '"}'
    kept = _run(text, chunk_size=500, oversized="keep").chunks
    assert [c.flags for c in kept] == [(), ("oversized",)]
    assert kept[1].path == ("$['value']",)
    _invariants(text, kept, 500)
    split = _run(text, chunk_size=500, oversized="split").chunks
    parts = [c for c in split if "split_part" in c.flags]
    assert len(parts) > 5 and all(c.flags == ("oversized", "split_part") for c in parts)
    assert all(c.path == ("$['value']",) and c.attr("first_key") == "value" for c in parts)
    _invariants(text, split, 500)


def test_deterministic_chunk_spans_and_config_validation():
    text = _store()
    chunker = JsonStructureChunker(JsonStructureConfig(chunk_size=64))
    doc = ChunkInput(text=text)
    assert chunker.chunk_structured(doc) == chunker.chunk_structured(doc)
    assert chunker.chunk(doc) == tuple(c.to_span() for c in chunker.chunk_structured(doc).chunks)
    registry = TechniqueRegistry(load_catalogue(ROOT / "config" / "techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C10", {}) == JsonStructureConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("C10", {"unknown": True})
    with pytest.raises(ValidationError):
        JsonStructureConfig(max_depth=0)


def test_one_million_code_points():
    text = "[" + ",".join('{"v": 1}' for _ in range(125_000)) + "]"
    started = time.perf_counter()
    result = _run(text)
    assert time.perf_counter() - started < 10
    assert len(result.chunks) > 500
