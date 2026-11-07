import random
import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_chunking import register_builtin_plugins
from rag_foundry_chunking.boundaries import sentence_ranges
from rag_foundry_chunking.multilingual import (
    MultilingualChunker,
    MultilingualConfig,
    normalize_with_map,
    script_of,
)
from rag_foundry_techniques import ChunkInput, TechniqueRegistry, load_catalogue

ROOT = Path(__file__).resolve().parents[2]


def _chunker(size=800, **kwargs):
    return MultilingualChunker(MultilingualConfig(chunk_size=size, **kwargs))


def _text(span, text):
    return text[span.start : span.end]


def test_script_and_mixed_sentence_ranges_and_abbreviations():
    text = "\u0645\u0631\u062d\u0628\u0627 API \u0627\u0644\u0639\u0627\u0644\u0645\u061f The API returns JSON."
    assert unicodedata.is_normalized("NFC", text)
    assert sentence_ranges(text, 0, len(text)) == [(0, 17), (18, len(text))]
    assert script_of("\u0645") == "arabic"
    assert script_of("A") == "latin"
    assert script_of("3") == "other"
    for char in "\u060c\u061b\u061f\u0660\u0669\u066b\u066c\u064e\u0640\u200c\u200d":
        assert script_of(char) == "other"
    abbrev = "\u0646\u0635 Dr. Smith \u0645\u0648\u062c\u0648\u062f."
    assert len(sentence_ranges(abbrev, 0, len(abbrev))) == 1


def test_oversized_clause_marks_are_level_one_boundaries():
    text = "\u0643" * 18 + "\u060c " + "\u062a" * 18 + "\u061b " + "\u0628" * 18 + "."
    assert unicodedata.is_normalized("NFC", text)
    spans = _chunker(32).chunk(ChunkInput(text=text))
    assert [text[s.end - 1] for s in spans[:-1]] == ["\u060c", "\u061b"]
    assert [s.split_level for s in spans] == [1, 1, 1]


def test_long_mixed_token_cuts_only_between_scripts():
    token = ("\u0643\u064e\u0644" * 6) + "API" + ("\u0628\u0650\u062a" * 6)
    text = token * 2
    spans = _chunker(32).chunk(ChunkInput(text=text))
    assert spans
    for span in spans:
        part = _text(span, text)
        assert part[0] not in "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0640\u200c"
        assert span.split_level >= 3
    for span in spans[:-1]:
        if span.split_level == 3:
            assert script_of(text[span.end - 1]) != "other"
            assert script_of(text[span.end]) != "other"
            assert script_of(text[span.end - 1]) != script_of(text[span.end])


def test_bidi_controls_stay_inside_token_chunk():
    text = "\u0643" * 20 + "\u2066API" + "\u0643" * 20 + "\u2069" + "\u0643" * 20
    spans = _chunker(32).chunk(ChunkInput(text=text))
    assert len(spans) > 1
    assert any("\u2066A" in _text(s, text) for s in spans)
    assert any("\u0643\u2069" in _text(s, text) for s in spans)
    assert all(text[s.start] not in "\u2069\u200f" and text[s.end - 1] != "\u2066" for s in spans)


def test_arabic_indic_decimal_and_nfc_codepoint_offsets():
    text = (
        ("\u0633" * 18)
        + " \u0663\u066b\u0661\u0664 "
        + ("\u062c" * 18)
        + ". "
        + ("\u0633" * 18)
        + " \u0663\u066b\u0661\u0664 "
        + ("\u062c" * 18)
        + "."
    )
    assert unicodedata.is_normalized("NFC", text)
    spans = _chunker(32).chunk(ChunkInput(text=text))
    assert len(spans) > 1
    assert any("\u0663\u066b\u0661\u0664" in _text(span, text) for span in spans)


def test_normalization_map_removes_marks_controls_and_folds_alef():
    text = "\u0622\u064e\u0623\u0640\u0625\u200f\u0671"
    normalized, mapping = normalize_with_map(text)
    assert normalized == "\u0627\u0627\u0627\u0627"
    assert len(normalized) == len(mapping)
    assert all(
        normalized[i]
        == {"\u0622": "\u0627", "\u0623": "\u0627", "\u0625": "\u0627", "\u0671": "\u0627"}.get(
            text[mapping[i]], text[mapping[i]]
        )
        for i in range(len(mapping))
    )
    assert all(left < right for left, right in zip(mapping, mapping[1:], strict=False))
    chunk = text[1:]
    value, offsets = normalize_with_map(text, 1, len(text))
    assert value and all(1 <= offset < len(text) for offset in offsets)
    assert chunk


def test_coverage_bounds_determinism_and_seeded_property():
    rng = random.Random(219)
    alphabet = "\u0643\u062a\u064e\u0651Axyz\u060c\u061f. \u200f\u2066\u2069"
    for overlap in (0, 1):
        for _ in range(2500):
            text = unicodedata.normalize("NFC", "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 180))))
            assert unicodedata.is_normalized("NFC", text)
            size = rng.choice((32, 40))
            chunker = _chunker(size, overlap_sentences=overlap)
            spans = chunker.chunk(ChunkInput(text=text))
            assert spans == chunker.chunk(ChunkInput(text=text))
            assert all(0 <= s.start < s.end <= len(text) for s in spans)
            assert all(s.end - s.start <= size for s in spans)
            assert all(not text[s.start].isspace() and not text[s.end - 1].isspace() for s in spans)
            assert all(a.start < b.start and a.end < b.end for a, b in zip(spans, spans[1:], strict=False))
            covered = {i for s in spans for i in range(s.start, s.end) if not text[i].isspace()}
            assert covered == {i for i, char in enumerate(text) if not char.isspace()}


def test_c21_registry_config_defaults_and_rejects_unknown_key():
    registry = TechniqueRegistry(load_catalogue(ROOT / "config" / "techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C21", {}) == MultilingualConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("C21", {"unknown": True})


def test_hard_cut_keeps_harakat_and_zwj_with_base():
    # Review r2: level-4 cuts must keep the ADR-0012 D3 safety rules.
    text = "\u0628\u064e" * 40
    spans = _chunker(33).chunk(ChunkInput(text=text))
    assert len(spans) > 1 and all(span.split_level == 4 for span in spans)
    assert all(text[span.end] != "\u064e" for span in spans[:-1])
    # A plain cut at 33 lands right after a ZWJ.
    zwj = "\u0628\u0628\u200d" * 30
    assert zwj[32] == "\u200d"
    assert all(zwj[span.end - 1] != "\u200d" for span in _chunker(33).chunk(ChunkInput(text=zwj))[:-1])
