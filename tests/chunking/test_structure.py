import pytest
from pydantic import ValidationError
from rag_foundry_chunking import ContextSpan, StructuredChunk, render
from rag_foundry_chunking.structure import line_ranges, split_oversized


def test_validation_and_render():
    with pytest.raises(ValidationError):
        StructuredChunk(
            start=5, end=8, split_level=0, kind="section", context=(ContextSpan(start=6, end=7, role="heading"),)
        )
    with pytest.raises(ValidationError):
        StructuredChunk(start=5, end=8, split_level=0, kind="section", flags=("split_part", "oversized"))
    with pytest.raises(ValidationError):
        StructuredChunk(start=5, end=8, split_level=0, kind="section", attrs=(("x", "1"), ("x", "2")))
    text = "head\nbody"
    chunk = StructuredChunk(
        start=5, end=9, split_level=0, kind="section", context=(ContextSpan(start=0, end=4, role="heading"),)
    )
    assert render(text, chunk) == "head\nbody"


def test_split_oversized_coverage():
    text = "one two\n" + "x" * 90 + "\nlast"
    pieces = split_oversized(text, 0, len(text), 32, 1)
    assert all(b - a <= 32 for a, b, _ in pieces)
    covered = {i for a, b, _ in pieces for i in range(a, b) if not text[i].isspace()}
    assert covered == {i for i, c in enumerate(text) if not c.isspace()}


def test_line_ranges_only_split_physical_newlines():
    text = "a\u2028# x\r\ny\rz"
    ranges = line_ranges(text)
    assert [text[a:b] for a, b, _ in ranges] == ["a\u2028# x", "y", "z"]


def test_pack_units_flags_come_from_the_unit():
    from rag_foundry_chunking.structure import pack_units

    text = "short one\n\n" + "word " * 20 + "\n\nshort two"
    units = [(0, 9), (11, 11 + len("word " * 20) - 1), (len(text) - 9, len(text))]
    split = pack_units(text, units, 32, "split")
    assert split[0] == (0, 9, 0, ())
    assert split[-1] == (len(text) - 9, len(text), 0, ())
    middle = split[1:-1]
    assert middle and all(flags == ("oversized", "split_part") and b - a <= 32 for a, b, _, flags in middle)
    keep = pack_units(text, units, 32, "keep")
    assert [flags for *_, flags in keep] == [(), ("oversized",), ()]
