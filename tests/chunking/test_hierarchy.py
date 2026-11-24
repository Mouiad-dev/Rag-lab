import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_chunking import (
    ParentChildChunker,
    ParentChildConfig,
    register_builtin_plugins,
)
from rag_foundry_techniques import ChunkInput, ChunkSpan, TechniqueRegistry, TextBlock, load_catalogue

ROOT = Path(__file__).resolve().parents[2]


def _fixture():
    english = (
        "English heading\nA fixed fact is that water freezes at zero C. The neighbouring sentence gives context.\n\n"
    )
    arabic = (
        "Arabic heading\n"
        "\u062d\u0642\u064a\u0642\u0629 \u062b\u0627\u0628\u062a\u0629: \u0627\u0644\u0645\u0627\u0621 "
        "\u064a\u062a\u062c\u0645\u062f \u0639\u0646\u062f \u0635\u0641\u0631 \u062f\u0631\u062c\u0629. "
        "\u0647\u0630\u0647 \u062c\u0645\u0644\u0629 \u062c\u0627\u0648\u0631\u0629 "
        "\u0644\u0644\u0633\u064a\u0627\u0642. \u0648\u0647\u0646\u0627\u0643 \u0641\u0627\u0635\u0644\u0629\u060c "
        "\u062f\u0627\u062e\u0644 \u062c\u0645\u0644\u0629\u061f"
    )
    text = english + arabic
    assert unicodedata.is_normalized("NFC", text)
    split = len(english)
    blocks = (
        TextBlock(
            block_id="eh",
            start=0,
            end=len("English heading"),
            kind="heading",
            section_path=("English",),
            heading_level=1,
        ),
        TextBlock(
            block_id="ep", start=len("English heading\n"), end=split - 2, kind="paragraph", section_path=("English",)
        ),
        TextBlock(
            block_id="ah",
            start=split,
            end=split + len("Arabic heading"),
            kind="heading",
            section_path=("Arabic",),
            heading_level=1,
        ),
        TextBlock(
            block_id="ap",
            start=split + len("Arabic heading\n"),
            end=len(text),
            kind="paragraph",
            section_path=("Arabic",),
        ),
    )
    return ChunkInput(text=text, blocks=blocks)


def test_linkage_coverage_sections_and_stability():
    doc = _fixture()
    chunker = ParentChildChunker(ParentChildConfig(parent_size=500, child_size=100))
    hierarchy = chunker.chunk_hierarchy(doc)
    assert hierarchy == chunker.chunk_hierarchy(doc)
    for child in hierarchy.children:
        owners = [p for p in hierarchy.parents if p.start <= child.start and child.end <= p.end]
        assert len(owners) == 1
        assert child.parent_id == owners[0].parent_id
    for parent in hierarchy.parents:
        children = [c for c in hierarchy.children if c.parent_id == parent.parent_id]
        assert all(a.end <= b.start for a, b in zip(children, children[1:], strict=False))
    for spans in (hierarchy.parents, hierarchy.children):
        covered = [i for span in spans for i in range(span.start, span.end) if not doc.text[i].isspace()]
        assert sorted(covered) == [i for i, char in enumerate(doc.text) if not char.isspace()]
    assert {parent.path for parent in hierarchy.parents} == {("English",), ("Arabic",)}


def test_chunk_with_parents_matches_chunk_and_hierarchy():
    doc = _fixture()
    chunker = ParentChildChunker(ParentChildConfig(parent_size=500, child_size=100))
    linked = chunker.chunk_with_parents(doc)
    hierarchy = chunker.chunk_hierarchy(doc)
    assert tuple(
        ChunkSpan(start=item.start, end=item.end, split_level=item.split_level) for item in linked
    ) == chunker.chunk(doc)
    for item, child in zip(linked, hierarchy.children, strict=True):
        parent = hierarchy.parent_of(child)
        assert (item.parent_start, item.parent_end) == (parent.start, parent.end)


def test_no_section_respect_and_fact_context():
    doc = _fixture()
    chunker = ParentChildChunker(ParentChildConfig(parent_size=500, child_size=100))
    hierarchy = chunker.chunk_hierarchy(doc)
    for quote, neighbour in (
        ("water freezes at zero C", "The neighbouring sentence gives context."),
        (
            "\u0627\u0644\u0645\u0627\u0621 \u064a\u062a\u062c\u0645\u062f",
            "\u0647\u0630\u0647 \u062c\u0645\u0644\u0629 \u062c\u0627\u0648\u0631\u0629",
        ),
    ):
        child = next(c for c in hierarchy.children if quote in doc.text[c.start : c.end])
        parent = hierarchy.parent_of(child)
        assert neighbour in doc.text[parent.start : parent.end]
    text = "First section.\n\nSecond section."
    blocks = (
        TextBlock(block_id="a", start=0, end=15, kind="paragraph", section_path=("A",)),
        TextBlock(block_id="b", start=17, end=len(text), kind="paragraph", section_path=("B",)),
    )
    doc = ChunkInput(text=text, blocks=blocks)
    merged = ParentChildChunker(
        ParentChildConfig(parent_size=200, child_size=50, respect_sections=False)
    ).chunk_hierarchy(doc)
    assert len(merged.parents) == 1
    assert merged.parents[0].path == ()
    respected = ParentChildChunker(ParentChildConfig(parent_size=200, child_size=50)).chunk_hierarchy(doc)
    overlaps = [
        {block.section_path for block in doc.blocks if block.start < parent.end and parent.start < block.end}
        for parent in respected.parents
    ]
    assert all(len(paths) == 1 for paths in overlaps)


def test_last_section_edit_preserves_earlier_parent_ids():
    text = "Alpha stays fixed.\n\nBeta stays fixed.\n\nGamma changes here."
    ranges = [(0, 18, "A"), (20, 38, "B"), (40, len(text), "C")]
    blocks = tuple(
        TextBlock(block_id=key, start=start, end=end, kind="paragraph", section_path=(key,))
        for start, end, key in ranges
    )
    config = ParentChildConfig(parent_size=200, child_size=50)
    before = ParentChildChunker(config).chunk_hierarchy(ChunkInput(text=text, blocks=blocks))
    edited_text = text.replace("changes", "was edited")
    edited_blocks = (*blocks[:2], blocks[2].model_copy(update={"end": len(edited_text)}))
    after = ParentChildChunker(config).chunk_hierarchy(ChunkInput(text=edited_text, blocks=edited_blocks))
    assert len(before.parents) == len(after.parents) == 3
    assert [p.parent_id for p in before.parents[:2]] == [p.parent_id for p in after.parents[:2]]
    assert before.parents[2].parent_id != after.parents[2].parent_id


def test_child_split_levels_are_local_to_child_fallback():
    text = "Short sentence.\n\n" + "word " * 20 + "enormous" * 20
    split = text.index("word")
    blocks = (
        TextBlock(block_id="short", start=0, end=len("Short sentence."), kind="paragraph", section_path=("A",)),
        TextBlock(block_id="long", start=split, end=len(text), kind="paragraph", section_path=("B",)),
    )
    hierarchy = ParentChildChunker(ParentChildConfig(parent_size=500, child_size=50)).chunk_hierarchy(
        ChunkInput(text=text, blocks=blocks)
    )
    levels = {child.split_level for child in hierarchy.children}
    assert {0, 1, 2} <= levels


def test_many_blocks_chunk_within_loose_time_bound():
    import time

    paragraph = "A useful factual sentence with some context here.\n\n"
    text = paragraph * 20_000
    blocks = []
    offset = 0
    for index in range(20_000):
        end = offset + len(paragraph) - 2
        blocks.append(TextBlock(block_id=str(index), start=offset, end=end, kind="paragraph", section_path=("S",)))
        offset += len(paragraph)
    doc = ChunkInput(text=text, blocks=tuple(blocks))
    chunker = ParentChildChunker(ParentChildConfig(parent_size=500, child_size=100))
    started = time.perf_counter()
    chunker.chunk_hierarchy(doc)
    assert time.perf_counter() - started < 3  # linear code takes ~0.3 s; the old per-unit block scan took ~10 s


def test_oversized_config_and_registry():
    text = "Sentence one has facts. " + "longword" * 30
    split = ParentChildChunker(ParentChildConfig(parent_size=200, child_size=50)).chunk_hierarchy(ChunkInput(text=text))
    assert any(p.flags == ("oversized", "split_part") for p in split.parents)
    kept = ParentChildChunker(ParentChildConfig(parent_size=200, child_size=50, oversized="keep")).chunk_hierarchy(
        ChunkInput(text=text)
    )
    assert len(kept.parents) == 1 and kept.parents[0].flags == ("oversized",)
    assert all(c.end - c.start <= 50 for c in kept.children)
    with pytest.raises(ValidationError):
        ParentChildConfig(parent_size=200, child_size=200)
    registry = TechniqueRegistry(load_catalogue(ROOT / "config" / "techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C18", {}) == ParentChildConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("C18", {"unknown": True})
    doc = ChunkInput(text="A sentence. Another sentence.")
    chunker = ParentChildChunker(ParentChildConfig())
    hierarchy = chunker.chunk_hierarchy(doc)
    assert chunker.chunk(doc) == tuple(c.to_span() for c in hierarchy.children)
    assert all(isinstance(span, ChunkSpan) for span in chunker.chunk(doc))
