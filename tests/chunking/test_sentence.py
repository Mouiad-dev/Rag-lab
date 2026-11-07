import random
import time
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_chunking import SentenceChunker, SentenceConfig, register_builtin_plugins
from rag_foundry_techniques import ChunkInput, TechniqueRegistry, load_catalogue

ROOT = Path(__file__).resolve().parents[2]


def _chunker(**kwargs):
    return SentenceChunker(SentenceConfig(**kwargs))


def _assert_coverage(text, spans, size):
    assert all(0 <= span.start < span.end <= len(text) for span in spans)
    assert all(not text[span.start].isspace() and not text[span.end - 1].isspace() for span in spans)
    assert all(span.end - span.start <= size for span in spans)
    covered = {i for span in spans for i in range(span.start, span.end) if not text[i].isspace()}
    assert covered == {i for i, char in enumerate(text) if not char.isspace()}
    for left, right in zip(spans, spans[1:], strict=False):
        assert right.start > left.start and right.end > left.end


def test_sentence_chunker_coverage_and_overlap():
    text = "Alpha. " * 40
    sentences = __import__("rag_foundry_chunking.boundaries", fromlist=["sentence_ranges"]).sentence_ranges(
        text, 0, len(text)
    )
    for overlap in (0, 1, 2):
        chunker = _chunker(chunk_size=64, overlap_sentences=overlap)
        spans = chunker.chunk(ChunkInput(text=text))
        assert spans == chunker.chunk(ChunkInput(text=text))
        _assert_coverage(text, spans, 64)
        for left, right in zip(spans, spans[1:], strict=False):
            shared = [start for start, end in sentences if start >= right.start and end <= left.end]
            assert len(shared) == overlap
            if overlap:
                assert right.start == shared[0]
                assert right.start == sentences[[start for start, _ in sentences].index(right.start)][0]


def test_long_sentence_and_long_word_split_levels_and_safe_codepoints():
    sentence = "a " * 30 + "end."
    spans = _chunker(chunk_size=32).chunk(ChunkInput(text=sentence))
    assert any(span.split_level == 1 for span in spans)
    text = "x" * 31 + "\u064e" + "y" * 30 + "\u200d" + "z" * 30 + "\ufe0f" + "q" * 30
    text += "\U0001f3fb" + "r" * 30 + "\r\n" + "s" * 30 + "\U0001f1f8\U0001f1e6" * 20
    spans = _chunker(chunk_size=32).chunk(ChunkInput(text=text))
    assert any(span.split_level == 2 for span in spans)
    assert all(text[s.end] not in ("\u064e", "\u200d", "\ufe0f", "\U0001f3fb", "\n") for s in spans[:-1])
    assert all(not (text[s.end - 1] == "\U0001f1f8" and text[s.end] == "\U0001f1e6") for s in spans[:-1])
    longword = "w" * 3000
    assert any(s.split_level == 2 for s in _chunker().chunk(ChunkInput(text=longword)))


def test_hard_cut_never_ends_after_zwj():
    text = "a" * 31 + "\u200d" + "b" * 90
    spans = _chunker(chunk_size=32).chunk(ChunkInput(text=text))
    assert all(text[span.end - 1] != "\u200d" for span in spans[:-1])


def test_million_codepoint_mixed_text_chunks_quickly():
    text = ("Arabic \u0645\u0631\u062d\u0628\u0627. English sentence. " * 25000)[:1_000_000]
    chunker = _chunker()
    started = time.perf_counter()
    spans = chunker.chunk(ChunkInput(text=text))
    elapsed = time.perf_counter() - started
    assert spans
    assert elapsed < 5


def test_seeded_random_properties():
    rng = random.Random(8317)
    alphabet = "ab .?!\n\u061f\u060c\u064e\u00bb"
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 240)))
        size = rng.choice((32, 40, 64))
        overlap = rng.choice((0, 1, 2))
        spans = _chunker(chunk_size=size, overlap_sentences=overlap).chunk(ChunkInput(text=text))
        _assert_coverage(text, spans, size)


def test_registry_and_config_validation():
    registry = TechniqueRegistry(load_catalogue(ROOT / "config" / "techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C06", {}) == SentenceConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("C06", {"unknown": True})
    for abbreviations in (("Dr",), ("foo.",), ("foo bar",), ("x", "x")):
        with pytest.raises(ValidationError):
            SentenceConfig(abbreviations=abbreviations)
