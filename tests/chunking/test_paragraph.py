import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_chunking import ParagraphChunker, ParagraphConfig, register_builtin_plugins
from rag_foundry_chunking.paragraph import paragraph_ranges
from rag_foundry_techniques import ChunkInput, TechniqueRegistry, TextBlock, load_catalogue

ROOT = Path(__file__).resolve().parents[2]


def _chunker(**kwargs):
    return ParagraphChunker(ParagraphConfig(**kwargs))


def _coverage(text, spans, size):
    assert all(0 <= span.start < span.end <= len(text) for span in spans)
    assert all(span.end - span.start <= size for span in spans)
    covered = {i for span in spans for i in range(span.start, span.end) if not text[i].isspace()}
    assert covered == {i for i, char in enumerate(text) if not char.isspace()}
    assert all(b.start > a.start and b.end > a.end for a, b in zip(spans, spans[1:], strict=False))


def test_whole_paragraph_packing():
    paragraphs = ("First paragraph.", "Second paragraph.", "Third paragraph.")
    text = "\n\n".join(paragraphs)
    assert len(_chunker(chunk_size=64).chunk(ChunkInput(text=text))) == 1
    spans = _chunker(chunk_size=32).chunk(ChunkInput(text=text))
    assert [text[span.start : span.end] for span in spans] == list(paragraphs)
    assert all(span.split_level == 0 for span in spans)


def test_oversized_arabic_sentence_overflow_and_hard_cuts():
    text = (
        "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645\u060c \u0647\u0646\u0627\u061f "
        "\u0647\u0630\u0647 \u062c\u0645\u0644\u0629 \u062b\u0627\u0646\u064a\u0629 \u0647\u0646\u0627\u061f "
        "\u062c\u0645\u0644\u0629 \u0623\u062e\u064a\u0631\u0629 \u0647\u0646\u0627."
    )
    doc = ChunkInput(text=text)
    spans = _chunker(chunk_size=32, overflow="sentences").chunk(doc)
    assert spans and all(span.split_level == 1 for span in spans)
    assert all(text[span.end - 1] in ("\u061f", ".") for span in spans)
    assert all(text[span.end - 1] != "\u060c" for span in spans)
    voweled = "\u0628\u064e" * 40
    assert unicodedata.is_normalized("NFC", voweled)
    hard = _chunker(chunk_size=33, overflow="hard_cut").chunk(ChunkInput(text=voweled))
    assert len(hard) > 1 and all(span.split_level == 3 for span in hard)
    assert all(voweled[span.end] != "\u064e" for span in hard[:-1])
    _coverage(voweled, hard, 33)


def test_blocks_define_paragraphs_without_blank_line():
    text = "first block second block"
    split = len("first block ")
    doc = ChunkInput(
        text=text,
        blocks=(TextBlock(block_id="a", start=0, end=split), TextBlock(block_id="b", start=split, end=len(text))),
    )
    assert paragraph_ranges(doc, True) == [(0, 11), (12, len(text))]
    assert len(_chunker(chunk_size=32).chunk(doc)) == 1
    assert paragraph_ranges(doc, False) == [(0, len(text))]


def test_mixed_nfc_exact_offsets():
    text = "\u0645\u064e\u0631\u0652\u062d\u064e\u0628\u064b\u0627 API\n\nCaf\u00e9 and \u0639\u0631\u0628\u064a"
    expected = ("\u0645\u064e\u0631\u0652\u062d\u064e\u0628\u064b\u0627 API", "Caf\u00e9 and \u0639\u0631\u0628\u064a")
    assert unicodedata.is_normalized("NFC", text)
    ranges = paragraph_ranges(ChunkInput(text=text), False)
    assert [text[start:end] for start, end in ranges] == list(expected)
    assert ranges == [(0, len(expected[0])), (len(expected[0]) + 2, len(text))]


def test_overlap_carries_one_whole_paragraph():
    paragraphs = ("paragraph one", "paragraph two", "paragraph three", "paragraph four")
    text = "\n\n".join(paragraphs)
    spans = _chunker(chunk_size=32, overlap_paragraphs=1).chunk(ChunkInput(text=text))
    assert all(b.start > a.start and b.end > a.end for a, b in zip(spans, spans[1:], strict=False))
    for left, right in zip(spans, spans[1:], strict=False):
        shared = [
            (start, end)
            for start, end in paragraph_ranges(ChunkInput(text=text), False)
            if start >= right.start and end <= left.end
        ]
        assert len(shared) == 1
        assert right.start == shared[0][0]


def test_coverage_bounds_and_determinism():
    text = "A short paragraph.\n\n" + "long " * 15 + ".\n\nLast one."
    chunker = _chunker(chunk_size=32)
    doc = ChunkInput(text=text)
    spans = chunker.chunk(doc)
    assert spans == chunker.chunk(doc)
    _coverage(text, spans, 32)


def test_registry_and_config_validation():
    registry = TechniqueRegistry(load_catalogue(ROOT / "config" / "techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C07", {}) == ParagraphConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("C07", {"unknown": True})
    with pytest.raises(ValidationError):
        ParagraphConfig(overflow="other")
