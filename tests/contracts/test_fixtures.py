"""Authored Arabic/English source fixtures: anchors match code-point slices of their blocks."""

import json
from pathlib import Path

import pytest
from rag_foundry_contracts import SourceAnchor

FIXTURES = Path(__file__).resolve().parents[2] / "packages" / "contracts" / "fixtures" / "sources"
TEXTS = {"en": "en_short.txt", "ar": "ar_short.txt", "mixed": "mixed_short.txt"}


def _text(key: str) -> str:
    raw = (FIXTURES / TEXTS[key]).read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "BOM found"
    text = raw.decode("utf-8")
    assert "\r" not in text
    assert text.endswith("\n")
    assert len(text) < 600
    return text


def _blocks(key: str) -> list[str]:
    return _text(key).rstrip("\n").split("\n\n")


def _anchors() -> list[SourceAnchor]:
    data = json.loads((FIXTURES / "anchors.json").read_bytes().decode("utf-8"))
    return [SourceAnchor.model_validate(item) for item in data]


def _key(anchor: SourceAnchor) -> str:
    return anchor.source.document_id.removeprefix("doc-")


@pytest.mark.parametrize("key", TEXTS)
def test_text_files_are_clean_utf8_lf(key):
    _text(key)


def test_every_anchor_matches_block_slice():
    anchors = _anchors()
    assert anchors
    for a in anchors:
        assert a.source.provider_namespace == "rag-foundry-fixtures"
        assert a.page_index is None and a.regions == ()
        assert a.text_range is not None and a.exact is not None
        blocks = _blocks(_key(a))
        block = blocks[int(a.block_id.removeprefix("p")) - 1]
        assert block[a.text_range.start : a.text_range.end] == a.exact, a.anchor_id


def test_each_fixture_has_whole_and_mid_paragraph_anchor():
    for key in TEXTS:
        blocks = _blocks(key)
        mine = [a for a in _anchors() if _key(a) == key]
        assert mine, key
        whole = [a for a in mine if a.exact == blocks[int(a.block_id[1:]) - 1]]
        mid = [a for a in mine if a.text_range.start > 0]
        assert whole and mid, key


def test_emoji_anchor_counts_code_points_not_utf16_units():
    emoji = [a for a in _anchors() if any(ord(c) > 0xFFFF for c in a.exact)]
    assert emoji
    for a in emoji:
        assert len(a.exact) < len(a.exact.encode("utf-16-le")) // 2


def test_arabic_anchor_with_diacritics():
    marks = [a for a in _anchors() if _key(a) == "ar" and any("ً" <= c <= "ْ" for c in a.exact)]
    assert marks


def test_repeated_english_paragraphs():
    blocks = _blocks("en")
    assert blocks[0] == blocks[2] != blocks[1]
    en = [a for a in _anchors() if _key(a) == "en"]
    p1 = [a for a in en if a.block_id == "p1" and a.text_range.start > 0]
    p3 = [a for a in en if a.block_id == "p3" and a.text_range.start > 0]
    assert p1 and p3
    assert p1[0].exact == p3[0].exact
    assert p1[0].block_id != p3[0].block_id
