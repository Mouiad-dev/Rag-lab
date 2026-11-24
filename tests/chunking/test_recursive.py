import random
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig, register_builtin_plugins
from rag_foundry_techniques import ChunkInput, TechniqueRegistry, load_catalogue

ROOT = Path(__file__).resolve().parents[2]


def _chunker(**kwargs):
    return RecursiveChunker(RecursiveConfig(**kwargs))


def test_spans_are_bounded_trimmed_cover_text_and_deterministic():
    text = ("First sentence. Second sentence! Third sentence? " * 45) + "\n\nend"
    chunker = _chunker(chunk_size=64, chunk_overlap=12)
    doc = ChunkInput(text=text)
    spans = chunker.chunk(doc)
    assert spans == chunker.chunk(doc)
    assert list(spans) == sorted(spans, key=lambda span: (span.start, span.end))
    assert all(not text[s.start].isspace() and not text[s.end - 1].isspace() for s in spans)
    assert all(s.end - s.start <= chunker.config.chunk_size for s in spans)
    covered = {i for span in spans for i in range(span.start, span.end) if not text[i].isspace()}
    assert covered == {i for i, char in enumerate(text) if not char.isspace()}
    for left, right in zip(spans, spans[1:], strict=False):
        assert max(0, left.end - right.start) <= chunker.config.chunk_overlap


def test_short_pieces_exercise_configured_overlap():
    text = "a b. " * 40
    spans = _chunker(chunk_size=32, chunk_overlap=8).chunk(ChunkInput(text=text))
    overlaps = [max(0, left.end - right.start) for left, right in zip(spans, spans[1:], strict=False)]
    assert any(overlap > 0 for overlap in overlaps)
    assert all(overlap <= 8 for overlap in overlaps)


def test_whitespace_overlap_does_not_emit_contained_span():
    text = (
        "ab.\n \tb.\n.bccb\n\t a\t\t\n\n\n. \nba  \n\nc .bbcb \nacb\t . b\tca abc cc\t..a.b\n\n\n"
        "a\tc .a.bb\n\n\tbab\t b ..\ta\nabbb.b\tba\n\n\nc \t \na\nc\n\t. \n cc.c\t    \n"
        "ac \t .b\t b\t  \taa\t\naa\t..\n c c\nb\tcbcb.\tcbb\n\taa\tc \n\t\t "
    )
    spans = _chunker(chunk_size=50, chunk_overlap=12).chunk(ChunkInput(text=text))
    assert [(span.start, span.end) for span in spans] == [(0, 28), (32, 77), (80, 122), (128, 174)]


def test_seeded_random_spans_have_strictly_increasing_ends():
    rng = random.Random(8317)
    for _ in range(300):
        text = "".join(rng.choice("abc .\t\n\rَ؟‍") for _ in range(rng.randint(1, 240)))
        size = rng.choice((32, 40, 50))
        overlap = rng.choice((0, min(8, size // 4), min(12, size // 4)))
        spans = _chunker(chunk_size=size, chunk_overlap=overlap).chunk(ChunkInput(text=text))
        for previous, current in zip(spans, spans[1:], strict=False):
            assert current.start > previous.start
            assert current.end > previous.end
            assert not (current.start >= previous.start and current.end <= previous.end)


def test_registry_registers_c05_defaults_and_rejects_unknown_config():
    registry = TechniqueRegistry(load_catalogue(ROOT / "config" / "techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C05", {}) == RecursiveConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("C05", {"unknown": True})


def test_config_constraints():
    with pytest.raises(ValidationError):
        RecursiveConfig(chunk_size=32, chunk_overlap=9)
    with pytest.raises(ValidationError):
        RecursiveConfig(separators=(" ",))
    with pytest.raises(ValidationError):
        RecursiveConfig(separators=(" ", " ", ""))


def test_oversized_paragraph_uses_sentence_separator_fallback():
    text = "This sentence has ordinary English words. " * 45
    spans = _chunker().chunk(ChunkInput(text=text))
    assert any(span.split_level >= 2 for span in spans)
    assert all(span.end - span.start <= 800 for span in spans)


def test_combining_mark_is_not_split_from_base():
    text = "\u0628\u064e" * 1500
    spans = _chunker().chunk(ChunkInput(text=text))
    assert all(text[span.end - 1] != "\u0628" for span in spans[:-1])
    assert all(span.end - span.start <= 800 for span in spans)


def test_zwj_sequence_hard_cut_is_not_before_joiner():
    text = "\U0001f469\u200d\U0001f4bb" * 1000
    spans = _chunker(chunk_size=32, chunk_overlap=0).chunk(ChunkInput(text=text))
    assert all(span.split_level == 12 for span in spans)
    assert all(text[span.end] != "\u200d" for span in spans[:-1])
    assert all(span.end - span.start <= 32 for span in spans)


def test_hard_cut_keeps_variation_selector_with_base():
    text = "\u2764\ufe0f" * 1000
    spans = _chunker(chunk_size=32, chunk_overlap=0).chunk(ChunkInput(text=text))
    assert all(text[span.end] != "\ufe0f" for span in spans[:-1])


def test_hard_cut_keeps_crlf_together():
    text = ("x" * 31 + "\r\n") * 100
    spans = _chunker(chunk_size=32, chunk_overlap=0).chunk(ChunkInput(text=text))
    assert all(not (text[span.end - 1] == "\r" and text[span.end] == "\n") for span in spans[:-1])


def test_arabic_separators_use_expected_levels():
    text = (
        "\u0645\u0631\u062d\u0628\u0627\u061f " * 12
        + "\u062a\u0627\u0645\u06d4 " * 12
        + "\u0645\u0631\u062d\u0628\u0627\u060c " * 12
    )
    spans = _chunker(chunk_size=32, chunk_overlap=0).chunk(ChunkInput(text=text))
    assert any(span.split_level == 5 for span in spans)
    assert any(span.split_level == 6 for span in spans)
    assert any(span.split_level == 10 for span in spans)
    assert any(text[span.end - 1] in ("\u061f", "\u060c") for span in spans)


def test_whitespace_only_text_produces_no_chunks():
    assert _chunker().chunk(ChunkInput(text=" \n\t ")) == ()


def test_trimmed_previous_start_is_not_reused_by_overlap():
    # Found by review fuzzing: the next chunk's overlap pulled its start back to the previous chunk's trimmed start.
    text = (
        "b.ba\n؟bَa‍b؟.b\tََa\t؟ac\t\t ؟\r\t\t.a\n\na؟bbcb\n\t\r\rb\t a\rc\r"
        "؟aa؟َ؟ َ b.؟َ\r\nb\t cc  ؟َbc‍cb\nac\t\t؟َ َ\tb"
        "\rcb\t ؟\tb\ta‍\n‍"
    )
    spans = _chunker(chunk_size=32, chunk_overlap=8).chunk(ChunkInput(text=text))
    for previous, current in zip(spans, spans[1:], strict=False):
        assert current.start > previous.start
        assert current.end > previous.end
