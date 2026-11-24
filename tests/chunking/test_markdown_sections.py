import time
import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_chunking import MarkdownSectionChunker, MarkdownSectionConfig, register_builtin_plugins, render
from rag_foundry_techniques import ChunkInput, TechniqueRegistry, load_catalogue

ROOT = Path(__file__).resolve().parents[2]


def _result(text, **kwargs):
    return MarkdownSectionChunker(MarkdownSectionConfig(**kwargs)).chunk_structured(ChunkInput(text=text))


def _invariants(text, chunks, size=None):
    assert all(0 <= c.start < c.end <= len(text) for c in chunks)
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:], strict=False))
    assert all(not text[c.start].isspace() and not text[c.end - 1].isspace() for c in chunks)
    covered = [i for c in chunks for i in range(c.start, c.end) if not text[i].isspace()]
    assert covered == [i for i, char in enumerate(text) if not char.isspace()]
    if size is not None:
        assert all(c.end - c.start <= size or "oversized" in c.flags for c in chunks)
        assert all(c.end - c.start <= size for c in chunks if "split_part" in c.flags)


def _body(text, chunk):
    return text[chunk.start : chunk.end]


def test_nested_paths_and_exact_heading_context():
    text = "# A\nintro a\n\n## B\nbody b\n\n### C\nbody c\n\n## D\nend"
    chunks = _result(text).chunks
    assert [c.path for c in chunks] == [("A",), ("A", "B"), ("A", "B", "C"), ("A", "D")]
    assert [_body(text, s) for s in chunks[2].context] == ["# A", "## B"]
    assert [_body(text, s) for s in chunks[3].context] == ["# A"]
    assert render(text, chunks[2]) == "# A\n## B\n### C\nbody c"
    assert chunks[2].attr("heading_level") == "3"
    _invariants(text, chunks)


def test_heading_only_parents_merge_into_first_child_when_it_fits():
    text = "# A\n## B\nbody"
    chunks = _result(text).chunks
    assert len(chunks) == 1 and chunks[0].start == 0 and chunks[0].path == ("A", "B") and chunks[0].context == ()
    long_body = "word " * 20
    text = "# A\n## B\n" + long_body
    chunks = _result(text, chunk_size=32).chunks
    assert _body(text, chunks[0]) == "# A" and chunks[0].path == ("A",)
    assert _body(text, chunks[1]).startswith("## B") and chunks[1].path == ("A", "B")
    _invariants(text, chunks, 32)
    sibling = "# A\n# B\nbody"
    assert [c.path for c in _result(sibling).chunks] == [("A",), ("B",)]


def test_hash_line_inside_fence_is_not_a_heading_and_fitting_fence_is_never_cut():
    fence = "```python\n# not a heading\nx = 1\n```"
    text = "# X\n\nlead paragraph text\n\n" + fence + "\n\ntail paragraph text"
    chunks = _result(text, chunk_size=40).chunks
    assert all(c.path == ("X",) for c in chunks)
    assert any(_body(text, c) == fence and c.kind == "code" for c in chunks)
    _invariants(text, chunks, 40)


def test_unclosed_fence_runs_to_the_end_and_preamble_has_empty_path():
    text = "intro\n\n# H\n\n```\nunfinished\n# still code"
    chunks = _result(text).chunks
    assert chunks[0].path == () and _body(text, chunks[0]) == "intro"
    assert [c.path for c in chunks[1:]] == [("H",)]
    assert _body(text, chunks[-1]).endswith("# still code")


def test_oversized_fence_keep_and_split():
    fence = "```\n" + "\n".join(f"line {i} of code" for i in range(20)) + "\n```"
    text = "# Code\n\n" + fence
    kept = _result(text, chunk_size=64, oversized="keep").chunks
    assert [c.flags for c in kept if _body(text, c) == fence] == [("oversized",)]
    _invariants(text, kept, 64)
    split = _result(text, chunk_size=64, oversized="split").chunks
    parts = [c for c in split if c.start >= text.index("```")]
    assert len(parts) > 1 and all(c.flags == ("oversized", "split_part") for c in parts)
    assert all(c.end - c.start <= 64 for c in parts)
    _invariants(text, split, 64)


def test_fence_adjacent_text_is_kept_in_both_modes():
    text = "## S\nIntro text\n```\ncode\n```\nAfter"
    for mode in ("keep", "split"):
        _invariants(text, _result(text, chunk_size=32, oversized=mode).chunks, 32)


def test_split_pieces_are_flagged():
    text = "# A\n\n" + ("word " * 30)
    chunks = _result(text, chunk_size=32).chunks
    pieces = [c for c in chunks if c.start >= text.index("word")]
    assert pieces and all(c.flags == ("oversized", "split_part") for c in pieces)


def test_unicode_line_separator_does_not_start_a_heading():
    text = "# A\nx\u2028# not\nmore"
    assert [c.path for c in _result(text).chunks] == [("A",)]


def test_arabic_nfc_heading_path():
    title = "\u0627\u0644\u0641\u0635\u0644 \u0627\u0644\u0623\u0648\u0651\u0644"
    text = f"# {title}\n\n\u0646\u0635 \u0639\u0631\u0628\u064a\u060c English.\n\n## Sub\nok"
    assert unicodedata.is_normalized("NFC", text)
    chunks = _result(text).chunks
    assert chunks[0].path == (title,) and chunks[1].path == (title, "Sub")
    assert _body(text, chunks[1].context[0]) == f"# {title}"


def test_large_sections_get_own_heading_as_context_after_first_chunk():
    text = "# Top\n\n## Part\n\n" + "\n\n".join(f"Paragraph number {i} here." for i in range(6))
    chunks = _result(text, chunk_size=64).chunks
    later = [c for c in chunks if c.path == ("Top", "Part")][1:]
    assert later and all([_body(text, s) for s in c.context] == ["# Top", "## Part"] for c in later)
    _invariants(text, chunks, 64)


def test_chunk_matches_structured_and_is_deterministic():
    text = "pre\n\n# A\n\n" + "text " * 100 + "\n\n## B\n\n```\ncode\n```\n"
    chunker = MarkdownSectionChunker(MarkdownSectionConfig(chunk_size=64))
    doc = ChunkInput(text=text)
    assert chunker.chunk(doc) == tuple(c.to_span() for c in chunker.chunk_structured(doc).chunks)
    assert chunker.chunk_structured(doc) == chunker.chunk_structured(doc)


def test_registry_and_config_validation():
    registry = TechniqueRegistry(load_catalogue(ROOT / "config" / "techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C08", {}) == MarkdownSectionConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("C08", {"unknown": True})
    with pytest.raises(ValidationError):
        MarkdownSectionConfig(max_heading_level=7)


def test_one_million_code_points_with_many_headings():
    unit = "## Heading\n\nSome body text for the section.\n\n```\ncode\n```\n\n"
    text = unit * (1_000_000 // len(unit) + 1)
    started = time.perf_counter()
    chunks = _result(text).chunks
    assert time.perf_counter() - started < 10
    assert len(chunks) > 10_000
