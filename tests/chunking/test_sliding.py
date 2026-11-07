import random
import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_chunking import (
    FixedCharChunker,
    FixedCharConfig,
    FixedWordChunker,
    FixedWordConfig,
    LabRegexTokenizer,
    SlidingWindowChunker,
    SlidingWindowConfig,
    WindowStats,
    register_builtin_plugins,
    window_stats,
)
from rag_foundry_chunking._units import char_units, word_units
from rag_foundry_techniques import ChunkInput, TechniqueRegistry, load_catalogue

ROOT = Path(__file__).resolve().parents[2]


def _coverage(text, spans):
    covered = {i for span in spans for i in range(span.start, span.end) if not text[i].isspace()}
    assert covered == {i for i, char in enumerate(text) if not char.isspace()}


def _unit_count(unit, text, start, end, tokenizer):
    if unit == "chars":
        return end - start
    if unit == "words":
        return sum(a >= start and b <= end for a, b in word_units(text))
    return len(tokenizer.token_spans(text[start:end]))


def test_sliding_units_overlap_bounds_coverage_and_exact_offsets():
    text = ("alpha beta gamma delta epsilon zeta eta theta " * 6).strip()
    tokenizer = LabRegexTokenizer()
    for unit in ("chars", "words", "tokens"):
        config = SlidingWindowConfig(unit=unit, window_size=16 if unit == "chars" else 8, overlap=3)
        chunker = SlidingWindowChunker(config)
        spans = chunker.chunk(ChunkInput(text=text))
        assert all(text[s.start : s.end] for s in spans)
        if unit == "chars":
            units = char_units(text)
        elif unit == "words":
            units = word_units(text)
        else:
            units = tokenizer.token_spans(text)
        starts = {start for start, _ in units}
        ends = {end for _, end in units}
        assert all(s.start in starts and s.end in ends for s in spans)
        if unit != "chars":
            assert all(text[s.start].isspace() is False and text[s.end - 1].isspace() is False for s in spans)
        assert all(a.start < b.start and a.end < b.end for a, b in zip(spans, spans[1:], strict=False))
        for span in spans:
            if unit == "chars":
                assert span.end - span.start <= config.window_size
            else:
                assert _unit_count(unit, text, span.start, span.end, tokenizer) <= config.window_size
        for left, right in zip(spans, spans[1:], strict=False):
            amount = _unit_count(unit, text, right.start, left.end, tokenizer)
            assert 1 <= amount <= config.overlap
        _coverage(text, spans)


def test_sliding_preserves_boundary_fact_that_fixed_words_splits():
    fact = "the rare mineral is stored here"
    text = "alpha beta gamma delta the rare mineral is stored here omega psi"
    fixed = FixedWordChunker(FixedWordConfig(words_per_chunk=5)).chunk(ChunkInput(text=text))
    sliding = SlidingWindowChunker(SlidingWindowConfig(unit="words", window_size=8, overlap=4)).chunk(
        ChunkInput(text=text)
    )
    assert not any(fact in text[s.start : s.end] for s in fixed)
    assert any(fact in text[s.start : s.end] for s in sliding)


def test_window_stats_and_fixed_baseline():
    from rag_foundry_techniques import ChunkSpan

    stats = window_stats((ChunkSpan(start=0, end=10, split_level=0), ChunkSpan(start=6, end=16, split_level=0)))
    assert stats == WindowStats(2, 20, 16, 4, 1.25)
    assert window_stats(()) == WindowStats(0, 0, 0, 0, 0.0)
    text = "one two three four five six seven eight nine ten " * 8
    sliding = SlidingWindowChunker(SlidingWindowConfig(window_size=12, overlap=4)).chunk(ChunkInput(text=text))
    fixed = FixedCharChunker(FixedCharConfig(chunk_size=80)).chunk(ChunkInput(text=text))
    assert window_stats(sliding).storage_ratio > 1
    assert window_stats(fixed).duplicated_code_points == 0


def test_arabic_word_punctuation_and_character_combining_boundaries():
    text = "\u0645\u0631\u062d\u0628\u0627\u060c \u0647\u0644\u061f \u0628\u064e\u0627 \u0646\u064e\u0639\u0645"
    assert unicodedata.is_normalized("NFC", text)
    word_spans = SlidingWindowChunker(SlidingWindowConfig(unit="words", window_size=2, overlap=1)).chunk(
        ChunkInput(text=text)
    )
    words = [text[start:end] for start, end in word_units(text)]
    assert words[0].endswith("\u060c") and words[1].endswith("\u061f")
    assert all(text[s.start : s.end] not in ("\u060c", "\u061f") for s in word_spans)
    char_text = "\u0628\u064e" * 16
    char_spans = SlidingWindowChunker(SlidingWindowConfig(unit="chars", window_size=8, overlap=2)).chunk(
        ChunkInput(text=char_text)
    )
    assert all(char_text[s.end] != "\u064e" for s in char_spans[:-1])


def test_config_registry_and_seeded_random_properties():
    with pytest.raises(ValidationError):
        SlidingWindowConfig(window_size=8, overlap=5)
    with pytest.raises(ValidationError):
        SlidingWindowConfig(unit="chars", window_size=6, overlap=1)
    with pytest.raises(ValidationError):
        SlidingWindowConfig(unknown=True)
    registry = TechniqueRegistry(load_catalogue(ROOT / "config" / "techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C04", {}) == SlidingWindowConfig()

    rng = random.Random(41302)
    alphabet = "ab .\t\n\r\u0628\u0627\u064e\u200d.19"
    for _ in range(200):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 240)))
        assert unicodedata.is_normalized("NFC", text)
        for unit in ("chars", "words", "tokens"):
            size = rng.choice((8, 16, 32))
            chunker = SlidingWindowChunker(SlidingWindowConfig(unit=unit, window_size=size, overlap=max(1, size // 4)))
            spans = chunker.chunk(ChunkInput(text=text))
            if unit == "chars":
                assert all(s.end - s.start <= size for s in spans)
            elif unit == "words":
                assert all(_unit_count(unit, text, s.start, s.end, chunker.tokenizer) <= size for s in spans)
            else:
                assert all(_unit_count(unit, text, s.start, s.end, chunker.tokenizer) <= size for s in spans)
            for left, right in zip(spans, spans[1:], strict=False):
                assert left.start < right.start and left.end < right.end
                assert not (right.start >= left.start and right.end <= left.end)
            _coverage(text, spans)


def test_trimmed_whitespace_windows_always_advance_both_offsets():
    text = "\n\talpha   beta \n gamma    delta epsilon"
    for unit in ("chars", "words", "tokens"):
        chunker = SlidingWindowChunker(
            SlidingWindowConfig(unit=unit, window_size=8 if unit == "chars" else 3, overlap=2 if unit == "chars" else 1)
        )
        spans = chunker.chunk(ChunkInput(text=text))
        assert all(a.start < b.start and a.end < b.end for a, b in zip(spans, spans[1:], strict=False))
        _coverage(text, spans)


def test_character_windows_never_emit_whitespace_only_chunks():
    # Found in review: untrimmed character windows emitted chunks made only of spaces.
    text = "ab" + " " * 40 + "cd"
    spans = SlidingWindowChunker(SlidingWindowConfig(unit="chars", window_size=8, overlap=4)).chunk(
        ChunkInput(text=text)
    )
    assert spans
    assert all(not text[s.start : s.end].isspace() for s in spans)
    _coverage(text, spans)
