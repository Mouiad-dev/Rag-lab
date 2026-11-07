import time
import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError

pytest.importorskip("selectolax")

from rag_foundry_chunking import (
    HtmlStructureChunker,
    HtmlStructureConfig,
    StructureParseError,
    register_builtin_plugins,
)
from rag_foundry_techniques import ChunkInput, TechniqueRegistry, TextBlock, load_catalogue

ROOT = Path(__file__).resolve().parents[2]
HTML = (
    "<header><p>Home</p><p>About</p><p>Help</p><p>Contact</p></header>"
    "<h1>Guide</h1><h2>Intro</h2><p>Opening words.</p>"
    "<h2>Code</h2><pre>print('hello')\n</pre><p>"
    + "long paragraph "
    * 10
    + "</p><h2>\u0627\u0644\u0641\u0635\u0644</h2>"
    "<p>\u0646\u0635 \u0639\u0631\u0628\u064a</p>"
    "<footer><p>Home</p><p>About</p><p>Help</p><p>Contact</p></footer>"
)


def _input(html=HTML):
    from rag_foundry_ingestion.extract import ExtractionLimits, get_extractor

    extracted = get_extractor("text/html").extract(html.encode("utf-8"), text_encoding=None, limits=ExtractionLimits())
    text = extracted.canonical_text
    blocks = tuple(
        TextBlock(
            block_id=b.block_id,
            start=b.start,
            end=b.end,
            kind=b.kind.value,
            section_path=b.section_path,
            heading_level=b.heading_level,
            page_index=b.page_index,
        )
        for b in extracted.blocks
    )
    assert unicodedata.is_normalized("NFC", text)
    return ChunkInput(text=text, blocks=blocks)


def test_html_sections_boilerplate_and_identity():
    doc = _input()
    result = HtmlStructureChunker(HtmlStructureConfig(chunk_size=64)).chunk_structured(doc)
    assert any(chunk.path == ("Guide", "Intro") for chunk in result.chunks)
    assert any(
        chunk.path == ("Guide", "Code")
        and any("print" in doc.text[b.start : b.end] for b in doc.blocks if b.block_id in chunk.block_ids)
        for chunk in result.chunks
    )
    assert any("\u0627\u0644\u0641\u0635\u0644" in chunk.path for chunk in result.chunks)
    assert len(result.skipped) == 4
    assert [item.block_id for item in result.skipped] == [block.block_id for block in doc.blocks[-4:]]
    assert "Home" in doc.text[result.chunks[0].start : result.chunks[0].end]
    body_ranges = [(c.start, c.end) for c in result.chunks] + [(s.start, s.end) for s in result.skipped]
    for index, char in enumerate(doc.text):
        if not char.isspace():
            assert sum(a <= index < b for a, b in body_ranges) == 1
    for chunk in result.chunks:
        expected = tuple(
            b.block_id
            for b in doc.blocks
            if b.start < chunk.end and b.end > chunk.start and b.block_id not in {s.block_id for s in result.skipped}
        )
        assert chunk.block_ids == expected
    intro = next(c for c in result.chunks if c.path == ("Guide", "Intro"))
    assert any(doc.text[c.start : c.end] == "Guide" for c in intro.context)
    assert not HtmlStructureChunker(HtmlStructureConfig(drop_repeated_boilerplate=False)).chunk_structured(doc).skipped


@pytest.mark.parametrize("oversized", ["keep", "split"])
def test_oversized_modes(oversized):
    doc = _input()
    result = HtmlStructureChunker(HtmlStructureConfig(chunk_size=32, oversized=oversized)).chunk_structured(doc)
    assert any("oversized" in chunk.flags for chunk in result.chunks)


def test_missing_structure_and_registry():
    chunker = HtmlStructureChunker(HtmlStructureConfig())
    with pytest.raises(StructureParseError, match="C09 needs") as exc:
        chunker.chunk_structured(ChunkInput(text="x"))
    assert exc.value.code == "missing_structure"
    doc = _input()
    with pytest.raises(StructureParseError) as exc:
        chunker.chunk_structured(
            ChunkInput(
                text=doc.text,
                blocks=tuple(TextBlock(block_id=b.block_id, start=b.start, end=b.end) for b in doc.blocks),
            )
        )
    assert exc.value.code == "missing_structure"
    registry = TechniqueRegistry(load_catalogue(ROOT / "config" / "techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C09", {}) == HtmlStructureConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("C09", {"unknown": True})


def _covered_once(doc, result):
    ranges = [(c.start, c.end) for c in result.chunks] + [(s.start, s.end) for s in result.skipped]
    for index, char in enumerate(doc.text):
        if not char.isspace():
            assert sum(a <= index < b for a, b in ranges) == 1


def test_exact_heading_context_and_locators():
    doc = _input()
    chunks = HtmlStructureChunker(HtmlStructureConfig(chunk_size=64)).chunk_structured(doc).chunks
    arabic = next(c for c in chunks if c.path == ("Guide", "\u0627\u0644\u0641\u0635\u0644"))
    assert [doc.text[s.start : s.end] for s in arabic.context] == ["Guide"]
    assert (
        doc.text[arabic.start : arabic.end] == "\u0627\u0644\u0641\u0635\u0644\n\n\u0646\u0635 \u0639\u0631\u0628\u064a"
    )
    later = [c for c in chunks if c.path == ("Guide", "Code") and "split_part" in c.flags]
    assert later and all([doc.text[s.start : s.end] for s in c.context] == ["Guide", "Code"] for c in later)
    assert all(s.role == "heading" for c in chunks for s in c.context)


@pytest.mark.parametrize("oversized", ["keep", "split"])
def test_oversized_block_is_flagged_never_dropped(oversized):
    doc = _input()
    result = HtmlStructureChunker(HtmlStructureConfig(chunk_size=32, oversized=oversized)).chunk_structured(doc)
    flagged = [c for c in result.chunks if "oversized" in c.flags]
    if oversized == "keep":
        assert [c.flags for c in flagged] == [("oversized",)] and flagged[0].end - flagged[0].start > 32
    else:
        assert len(flagged) > 1 and all(c.flags == ("oversized", "split_part") for c in flagged)
        assert all(c.end - c.start <= 32 for c in flagged)
        assert {c.block_ids for c in flagged} == {(flagged[0].block_ids[0],)}
    _covered_once(doc, result)


def test_one_million_code_points():
    section = "<h2>Part</h2><p>Body text for this part of the page.</p><p>Home</p><pre>x = 1\n</pre>"
    doc = _input("<h1>Big</h1>" + section * (1_000_000 // 60))
    started = time.perf_counter()
    result = HtmlStructureChunker(HtmlStructureConfig()).chunk_structured(doc)
    assert time.perf_counter() - started < 10
    assert len(result.skipped) > 1000 and len(result.chunks) > 1000
