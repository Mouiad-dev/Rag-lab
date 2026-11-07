import random
import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_chunking import (
    FixedCharChunker,
    FixedCharConfig,
    FixedTokenChunker,
    FixedTokenConfig,
    FixedWordChunker,
    FixedWordConfig,
    LabRegexTokenizer,
    register_builtin_plugins,
)
from rag_foundry_chunking._units import char_units
from rag_foundry_techniques import ChunkInput, TechniqueRegistry, load_catalogue

ROOT = Path(__file__).resolve().parents[2]


def _coverage(text, spans):
    covered = {i for span in spans for i in range(span.start, span.end) if not text[i].isspace()}
    assert covered == {i for i, char in enumerate(text) if not char.isspace()}


def test_fixed_chars_cover_sorted_bounded_deterministic_text():
    text = "First sentence. Second sentence! Third sentence? " * 5
    chunker = FixedCharChunker(FixedCharConfig(chunk_size=16))
    spans = chunker.chunk(ChunkInput(text=text))
    assert spans == chunker.chunk(ChunkInput(text=text))
    assert list(spans) == sorted(spans, key=lambda span: (span.start, span.end))
    assert all(text[span.start : span.end] and span.end - span.start <= 16 for span in spans)
    for left, right in zip(spans, spans[1:], strict=False):
        assert left.end <= right.start
    _coverage(text, spans)
    assert any(text[span.end - 1].isalpha() and text[span.end].isalpha() for span in spans[:-1])


def test_fixed_chars_keep_grapheme_boundaries():
    text = "\u0628\u064e" * 20
    assert unicodedata.is_normalized("NFC", text)
    spans = FixedCharChunker(FixedCharConfig(chunk_size=8)).chunk(ChunkInput(text=text))
    assert all(text[span.end] != "\u064e" for span in spans[:-1])
    for text, forbidden in (
        ("\U0001f469\u200d\U0001f4bb" * 8, "\u200d"),
        ("\u2764\ufe0f" * 8, "\ufe0f"),
        (("x" * 7 + "\r\n") * 8, "\n"),
    ):
        spans = FixedCharChunker(FixedCharConfig(chunk_size=8)).chunk(ChunkInput(text=text))
        assert all(text[span.end] != forbidden for span in spans[:-1])
    assert FixedCharChunker(FixedCharConfig()).chunk(ChunkInput(text=" \n\t ")) == ()


def test_fixed_words_use_whitespace_only_boundaries():
    text = "\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645\u060c \u0647\u0644\u061f four Latin\u0628\u0627"
    assert unicodedata.is_normalized("NFC", text)
    spans = FixedWordChunker(FixedWordConfig(words_per_chunk=1)).chunk(ChunkInput(text=text))
    assert all(len(text[s.start : s.end].split()) <= 1 for s in spans)
    assert [word for s in spans for word in text[s.start : s.end].split()] == text.split()
    assert text[spans[1].start : spans[1].end].endswith("\u060c")
    assert text[spans[2].start : spans[2].end].endswith("\u061f")
    assert text[spans[-1].start : spans[-1].end].endswith("Latin\u0628\u0627")
    assert all(not text[s.start].isspace() and not text[s.end - 1].isspace() for s in spans)


def test_fixed_tokens_default_and_injected_tokenizer():
    tokenizer = LabRegexTokenizer()
    text = "\u0628\u064e\u0627 2026 words"
    assert unicodedata.is_normalized("NFC", text)
    tokens = tokenizer.token_spans(text)
    assert text[tokens[0][0] : tokens[0][1]] == "\u0628\u064e\u0627"
    digits = "2026"
    assert [digits[start:end] for start, end in tokenizer.token_spans(digits)] == ["202", "6"]
    sample = "This sentence has several words."
    assert len(sample) != len(tokenizer.token_spans(sample))
    chunker = FixedTokenChunker(FixedTokenConfig(tokens_per_chunk=8))
    long_text = (sample + " ") * 10
    spans = chunker.chunk(ChunkInput(text=long_text))
    assert all(len(tokenizer.token_spans(long_text[s.start : s.end])) <= 8 for s in spans)
    assert sum(len(tokenizer.token_spans(long_text[s.start : s.end])) for s in spans) == len(
        tokenizer.token_spans(long_text)
    )
    _coverage(long_text, spans)

    class FakeTokenizer:
        name = "fake"

        def token_spans(self, text):
            return tuple((index, index + 1) for index, char in enumerate(text) if not char.isspace())

    fake_text = "a b c d e f g h i j"
    fake = FixedTokenChunker(FixedTokenConfig(tokens_per_chunk=8), tokenizer=FakeTokenizer())
    fake_spans = fake.chunk(ChunkInput(text=fake_text))
    assert fake.tokenizer.name == "fake"
    assert all(len(FakeTokenizer().token_spans(fake_text[s.start : s.end])) <= 8 for s in fake_spans)
    _coverage(fake_text, fake_spans)
    with pytest.raises(ValidationError):
        FixedTokenConfig(tokenizer="gpt2")
    with pytest.raises(ValidationError):
        FixedTokenConfig(unknown=True)


def test_fixed_registry_defaults_and_seeded_properties():
    registry = TechniqueRegistry(load_catalogue(ROOT / "config" / "techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C01", {}) == FixedCharConfig()
    assert registry.validate_config("C02", {}) == FixedWordConfig()
    assert registry.validate_config("C03", {}) == FixedTokenConfig()
    assert {"C01", "C02", "C03", "C05"}.issubset(registry.registered_ids())

    rng = random.Random(8317)
    alphabet = "ab .\t\n\r\u0628\u0627\u064e\u200d.19"
    chunkers = (
        lambda size: FixedCharChunker(FixedCharConfig(chunk_size=max(8, size))),
        lambda size: FixedWordChunker(FixedWordConfig(words_per_chunk=max(1, size // 8))),
        lambda size: FixedTokenChunker(FixedTokenConfig(tokens_per_chunk=max(8, size // 4))),
    )
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 240)))
        assert unicodedata.is_normalized("NFC", text)
        for make_chunker in chunkers:
            chunker = make_chunker(rng.choice((8, 16, 32)))
            spans = chunker.chunk(ChunkInput(text=text))
            for previous, current in zip(spans, spans[1:], strict=False):
                assert current.start > previous.start and current.end > previous.end
                assert not (current.start >= previous.start and current.end <= previous.end)
            assert all(not text[s.start].isspace() and not text[s.end - 1].isspace() for s in spans)
            if isinstance(chunker, FixedCharChunker):
                clusters = set(char_units(text))
                assert all(s.end - s.start <= chunker.config.chunk_size or (s.start, s.end) in clusters for s in spans)
            elif isinstance(chunker, FixedWordChunker):
                assert all(len(text[s.start : s.end].split()) <= chunker.config.words_per_chunk for s in spans)
            else:
                assert all(
                    len(chunker.tokenizer.token_spans(text[s.start : s.end])) <= chunker.config.tokens_per_chunk
                    for s in spans
                )
            _coverage(text, spans)
