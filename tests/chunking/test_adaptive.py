from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_chunking import AdaptiveChunker, AdaptiveConfig, compare_with_fixed, register_builtin_plugins
from rag_foundry_chunking.adaptive import classify
from rag_foundry_chunking.multilingual import MultilingualChunker
from rag_foundry_techniques import ChunkInput, TechniqueRegistry, TextBlock, load_catalogue

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    "English prose paragraph. Another sentence.\n\n"
    "- first item\n- second item\n\n"
    "Name | Value | Extra\n--- | --- | ---\na | b | c\n\n"
    "    def run():\n    return 1\n\n"
    "English \u0643\u0644\u0645\u0627\u062a \u0639\u0631\u0628\u064a "
    "\u0643\u0644\u0645\u0627\u062a \u0639\u0631\u0628\u064a."
)
EXPECTED = (("prose", "C06"), ("list", "C07"), ("table", "C07"), ("code", "C05"), ("mixed_script", "C21"))


def test_regions_route_and_spans_preserve_exact_source():
    doc = ChunkInput(text=FIXTURE)
    chunker = AdaptiveChunker(AdaptiveConfig(chunk_size=64))
    spans, decisions = chunker.segment(doc)
    assert [(region.kind, region.policy) for region in decisions] == list(EXPECTED)
    assert all(FIXTURE[span.start : span.end] == FIXTURE[span.start : span.end].strip() for span in spans)
    assert all(span.end - span.start <= 64 for span in spans)
    assert all(
        not (span.start < region.start < span.end or span.start < region.end < span.end)
        for span in spans
        for region in decisions
    )
    assert all(any(region.start <= span.start < span.end <= region.end for region in decisions) for span in spans)


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        ("- a\n* b", "list"),
        ("1. a\n2) b", "list"),
        ("\u0661. a\n\u0662) b", "list"),
        ("a | b | c\nx | y | z", "table"),
        ("\talpha\n    beta", "code"),
        ("a {\nb }", "code"),
        ("Hello \u0645\u0631\u062d\u0628\u0627", "mixed_script"),
        ("ordinary words\nmore prose", "prose"),
    ],
)
def test_classification_positive(value, kind):
    assert classify(value, 0, len(value), 0.2)[0] == kind


@pytest.mark.parametrize(
    "value", ["a | b\nx | y", "- item\nordinary\nordinary", "Hello only Latin", "    one\nordinary\nordinary"]
)
def test_classification_near_misses(value):
    assert (
        classify(value, 0, len(value), 0.2)[0]
        != {
            "a | b\nx | y": "table",
            "- item\nordinary\nordinary": "list",
            "Hello only Latin": "mixed_script",
            "    one\nordinary\nordinary": "code",
        }[value]
    )


def test_blocks_control_regions_when_enabled():
    text = "one block \u0645\u0631\u062d\u0628\u0627 second block"
    split = len("one block ")
    doc = ChunkInput(
        text=text,
        blocks=(TextBlock(block_id="a", start=0, end=split), TextBlock(block_id="b", start=split, end=len(text))),
    )
    assert len(AdaptiveChunker(AdaptiveConfig(use_blocks=True)).segment(doc)[1]) == 2
    assert len(AdaptiveChunker(AdaptiveConfig(use_blocks=False)).segment(doc)[1]) == 1


def test_policy_exception_falls_back_for_only_affected_region(monkeypatch):
    text = "normal prose\n\nEnglish \u0643\u0644\u0645\u0627\u062a mixed"

    def fail(self, doc):
        raise RuntimeError("sentinel")

    monkeypatch.setattr(MultilingualChunker, "chunk", fail)
    doc = ChunkInput(text=text)
    spans, decisions = AdaptiveChunker(AdaptiveConfig()).segment(doc)
    assert decisions[0].failed is False
    assert decisions[1].failed is True and decisions[1].error == "RuntimeError: sentinel"
    assert decisions[1].policy == "C05"
    assert all(any(r.start <= s.start < s.end <= r.end for r in decisions) for s in spans)


def test_out_of_region_policy_span_is_caught(monkeypatch):
    def invalid(self, doc):
        from rag_foundry_techniques import ChunkSpan

        return (ChunkSpan(start=0, end=len(doc.text) + 1, split_level=0),)

    monkeypatch.setattr(MultilingualChunker, "chunk", invalid)
    _, decisions = AdaptiveChunker(AdaptiveConfig()).segment(ChunkInput(text="English \u0645\u0631\u062d\u0628\u0627"))
    assert decisions[0].failed
    assert "ValueError" in decisions[0].error


def test_config_validation_and_custom_routing():
    custom = dict(AdaptiveConfig().policies)
    custom["prose"] = "C05"
    assert (
        AdaptiveChunker(AdaptiveConfig(policies=custom)).segment(ChunkInput(text="Some prose."))[1][0].policy == "C05"
    )
    with pytest.raises(ValidationError):
        AdaptiveConfig(policies={"prose": "C05"})
    with pytest.raises(ValidationError):
        AdaptiveConfig(policies={**AdaptiveConfig().policies, "code": "C99"})
    with pytest.raises(ValidationError):
        AdaptiveConfig(router="model")


def test_compare_fixed_and_registry_and_repeatability():
    text = "English words form the first block and continue - item two is in another block"
    split = len("English words form the first block and continue")
    doc = ChunkInput(
        text=text,
        blocks=(TextBlock(block_id="one", start=0, end=split), TextBlock(block_id="two", start=split, end=len(text))),
    )
    config = AdaptiveConfig(chunk_size=32)
    result = compare_with_fixed(doc, config, "C05")
    assert result.fixed_cross_region > 0
    assert result.adaptive_chunks == sum(region.chunk_count for region in result.regions)
    assert result.failures == sum(region.failed for region in result.regions)
    assert result.fixed_chunks > 0
    spans = AdaptiveChunker(config).chunk(doc)
    assert spans == AdaptiveChunker(config).chunk(doc)
    assert all(
        not (span.start < boundary < span.end)
        for span in spans
        for region in result.regions
        for boundary in (region.start, region.end)
    )
    registry = TechniqueRegistry(load_catalogue(ROOT / "config" / "techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C19", {}) == AdaptiveConfig()
