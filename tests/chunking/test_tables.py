import time
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_chunking import TableChunker, TableConfig, register_builtin_plugins, render, table_cells
from rag_foundry_techniques import ChunkInput, TechniqueRegistry, load_catalogue

ROOT = Path(__file__).resolve().parents[2]
HEADER = "| Item | Weight (kg) | Price (USD) |"
DELIM = "| --- | ---: | :---: |"


def _result(text, **kwargs):
    return TableChunker(TableConfig(**kwargs)).chunk_structured(ChunkInput(text=text))


def _invariants(text, chunks, size=None):
    assert all(0 <= c.start < c.end <= len(text) for c in chunks)
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:], strict=False))
    assert all(not text[c.start].isspace() and not text[c.end - 1].isspace() for c in chunks)
    covered = [i for c in chunks for i in range(c.start, c.end) if not text[i].isspace()]
    assert covered == [i for i, char in enumerate(text) if not char.isspace()]
    if size is not None:
        assert all(c.end - c.start <= size for c in chunks if "split_part" in c.flags)


def _pipe_doc(rows=45):
    body = "\n".join(f"| Item {i} | {i} | {i * 10} |" for i in range(1, rows + 1))
    return "Intro paragraph.\n\n" + HEADER + "\n" + DELIM + "\n" + body + "\n\nAfter the table."


def test_pipe_table_row_groups_keep_header_and_lineage():
    text = _pipe_doc()
    result = _result(text, rows_per_chunk=20)
    tables = [c for c in result.chunks if c.kind == "table"]
    assert len(tables) == 3
    assert [(c.attr("row_first"), c.attr("row_last")) for c in tables] == [("1", "20"), ("21", "40"), ("41", "45")]
    assert {c.attr("table_index") for c in tables} == {"0"}
    assert {c.attr("table_start") for c in tables} == {str(text.index(HEADER))}
    assert {c.attr("columns") for c in tables} == {"3"} and {c.attr("format") for c in tables} == {"pipe"}
    assert text[tables[0].start : tables[0].end].startswith(HEADER + "\n" + DELIM)
    assert tables[0].context == ()
    for chunk in tables[1:]:
        assert [text[s.start : s.end] for s in chunk.context] == [HEADER, DELIM]
        assert all(s.role == "table_header" for s in chunk.context)
    assert render(text, tables[2]).startswith(HEADER + "\n" + DELIM + "\n| Item 41 |")
    cell = next(c for c in table_cells(text, tables[1]) if c.row == 25 and c.column == 2)
    assert (cell.column_label, cell.row_label, text[cell.start : cell.end]) == ("Weight (kg)", "Item 25", "25")
    first = table_cells(text, tables[0])
    assert first[0].row == 1 and first[0].column_label == "Item" and text[first[0].start : first[0].end] == "Item 1"
    assert [c.kind for c in result.chunks] == ["text", "table", "table", "table", "text"]
    _invariants(text, result.chunks)


def test_tsv_table_works_the_same():
    rows = "\n".join(f"Item {i}\t{i}\t{i * 10}" for i in range(1, 46))
    text = "Item\tWeight (kg)\tPrice (USD)\n" + rows
    tables = _result(text, rows_per_chunk=20).chunks
    assert [(c.attr("row_first"), c.attr("row_last"), c.attr("format")) for c in tables] == [
        ("1", "20", "tsv"),
        ("21", "40", "tsv"),
        ("41", "45", "tsv"),
    ]
    assert [text[s.start : s.end] for s in tables[2].context] == ["Item\tWeight (kg)\tPrice (USD)"]
    cell = next(c for c in table_cells(text, tables[1]) if c.row == 25 and c.column == 3)
    assert (cell.column_label, cell.row_label, text[cell.start : cell.end]) == ("Price (USD)", "Item 25", "250")
    _invariants(text, tables)


def test_size_budget_counts_rows_only():
    text = _pipe_doc(rows=12)
    tables = [c for c in _result(text, chunk_size=64, rows_per_chunk=100).chunks if c.kind == "table"]
    assert len(tables) > 1 and not any(c.flags for c in tables)
    assert all(int(c.attr("row_last")) >= int(c.attr("row_first")) for c in tables)
    assert all(c.end - c.start <= 64 for c in tables[1:])


def test_empty_cell_keeps_true_column_positions():
    text = "| A | B | C |\n| - | - | - |\n| a |  | c |"
    cells = table_cells(text, _result(text).chunks[0])
    assert [(c.column, c.column_label, text[c.start : c.end]) for c in cells] == [(1, "A", "a"), (3, "C", "c")]


def test_escaped_pipe_stays_inside_a_cell():
    text = "| A | B |\n| --- | --- |\n| left \\| right | x |"
    cells = table_cells(text, _result(text).chunks[0])
    assert [text[c.start : c.end] for c in cells] == ["left \\| right", "x"]


def test_oversized_single_row_keep_and_split():
    long_cell = " ".join(f"w{i:03d}" for i in range(400))
    text = HEADER + "\n" + DELIM + "\n| big | " + long_cell + " | 1 |\n| small | 2 | 3 |"
    kept = _result(text, chunk_size=200, oversized="keep").chunks
    assert kept[0].attr("part") == "header" and text[kept[0].start : kept[0].end] == HEADER + "\n" + DELIM
    assert kept[0].attr("row_first") is None and table_cells(text, kept[0]) == ()
    assert kept[1].flags == ("oversized",) and (kept[1].attr("row_first"), kept[1].attr("row_last")) == ("1", "1")
    assert [text[s.start : s.end] for s in kept[1].context] == [HEADER, DELIM]
    assert table_cells(text, kept[1])[0].row_label == "big"
    assert kept[2].attr("row_first") == "2" and not kept[2].flags
    _invariants(text, kept)
    split = _result(text, chunk_size=200, oversized="split").chunks
    parts = [c for c in split if "split_part" in c.flags]
    assert len(parts) > 1 and all(c.flags == ("oversized", "split_part") for c in parts)
    assert all((c.attr("row_first"), c.attr("row_last")) == ("1", "1") for c in parts)
    assert all(table_cells(text, c) == () for c in parts)
    assert all([text[s.start : s.end] for s in c.context] == [HEADER, DELIM] for c in parts)
    _invariants(text, split, 200)


def test_text_around_tables_is_packed_and_tab_indented_code_is_not_a_table():
    text = "One.\n\nTwo.\n\n\tif x:\n\t\treturn 1\n\nA\tB\n1\t2\n\nEnd."
    chunks = _result(text).chunks
    assert [c.kind for c in chunks] == ["text", "table", "text"]
    assert text[chunks[0].start : chunks[0].end] == "One.\n\nTwo.\n\n\tif x:\n\t\treturn 1"
    _invariants(text, chunks)


def test_oversized_text_paragraph_is_flagged():
    text = ("word " * 100).strip()
    assert _result(text, chunk_size=64, oversized="keep").chunks[0].flags == ("oversized",)
    split = _result(text, chunk_size=64).chunks
    assert all(c.flags == ("oversized", "split_part") for c in split)
    _invariants(text, split, 64)


def test_chunk_matches_structured_and_is_deterministic():
    chunker = TableChunker(TableConfig(rows_per_chunk=7))
    doc = ChunkInput(text=_pipe_doc())
    assert chunker.chunk(doc) == tuple(c.to_span() for c in chunker.chunk_structured(doc).chunks)
    assert chunker.chunk_structured(doc) == chunker.chunk_structured(doc)


def test_registry_and_config_validation():
    registry = TechniqueRegistry(load_catalogue(ROOT / "config" / "techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C12", {}) == TableConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("C12", {"unknown": True})
    with pytest.raises(ValidationError):
        TableConfig(rows_per_chunk=0)


def test_one_million_code_points_of_tables():
    table = _pipe_doc(rows=200) + "\n\n"
    text = table * (1_000_000 // len(table) + 1)
    started = time.perf_counter()
    chunks = _result(text).chunks
    assert time.perf_counter() - started < 10
    assert sum(c.kind == "table" for c in chunks) > 100
