# C12: Table-aware segments

## Intuition
A table row means little without its header: `| desk | 30 |` needs "Item" and "Weight (kg)". Split large tables into groups of whole rows, and give every group the original header (by source offsets) and its row numbers.

## Mechanism
`TableChunker` is in `packages/chunking/src/rag_foundry_chunking/tables.py`. It finds two kinds of tables in the canonical text:
- pipe tables: header line, delimiter line (`| --- | :---: |`) with the same cell count, then body lines with `|` (escaped `\|` stays in the cell);
- tab-separated tables: two or more lines with the same number of tabs after the first non-space character (tab-indented code is not a table).

Body rows are grouped by `rows_per_chunk` (default 20) and `chunk_size` (only the rows count; the header is context). The first group's chunk starts at the header line; later groups carry `table_header` context spans for the header and delimiter lines. Each table chunk has lineage `attrs`: `table_index`, `table_start`, `row_first`, `row_last`, `columns`, `format`.

A single row larger than `chunk_size` is flagged, never dropped: with `oversized="keep"` it is one chunk flagged `oversized`; with `"split"` (default) it is cut into pieces flagged `oversized` + `split_part`, all with the same row numbers and the header as context. If it is the first row, the header becomes its own chunk (`part=header`).

`table_cells(text, chunk)` returns each cell's exact source offsets with its `row`, `column`, `row_label` (first cell) and `column_label` (header cell). Empty cells keep their column number. Text outside tables is packed by paragraph.

## Worked example
Input (`rows_per_chunk=2`):
```text
| Item | Weight (kg) |
| --- | --- |
| pens | 0.2 |
| books | 1.5 |
| desk | 30 |
```

Real output (`row_first`, `row_last`, `render()`), then `table_cells` of the second chunk:
```text
1 2 '| Item | Weight (kg) |\n| --- | --- |\n| pens | 0.2 |\n| books | 1.5 |'
3 3 '| Item | Weight (kg) |\n| --- | --- |\n| desk | 30 |'
[(3, 'desk', 'Item', 'desk', (70, 74)), (3, 'desk', 'Weight (kg)', '30', (77, 79))]
```
The value `30` is found with its unit (`Weight (kg)`), its row label (`desk`) and its source span `(77, 79)`.

## Runnable demo
```bash
uv run python -c "from rag_foundry_chunking import TableChunker as T, TableConfig as C, render, table_cells; from rag_foundry_techniques import ChunkInput; t='| Item | Weight (kg) |\n| --- | --- |\n| pens | 0.2 |\n| books | 1.5 |\n| desk | 30 |'; ch=T(C(rows_per_chunk=2)).chunk_structured(ChunkInput(text=t)).chunks; [print(c.attr('row_first'), c.attr('row_last'), repr(render(t, c))) for c in ch]; print([(x.row, x.row_label, x.column_label, t[x.start:x.end], (x.start, x.end)) for x in table_cells(t, ch[1])])"
```

## Failure case
One row with a 90-character note and `chunk_size=40` cannot fit. Real output (`part`, `row_first`, `flags`, offsets):
```text
header None () (0, 30)
None 1 ('oversized', 'split_part') (31, 69)
None 1 ('oversized', 'split_part') (70, 109)
None 1 ('oversized', 'split_part') (110, 131)
```
Nothing is lost and every piece says it is part of row 1, but no piece is a whole row, so `table_cells` returns nothing for them. Arbitrary tables do not always fit a size limit.

## Limitations
Only pipe and tab tables in the canonical text. HTML tables are flattened into cell blocks by the current HTML extractor, so row structure is not visible; PDF tables need layout data (C11, still planned). Multi-line cells and merged cells are not supported. Lineage and header context are not yet stored in the index or embedded; `chunk()` gives offsets only.

## Baseline comparison
Not measured yet; comes with the P4.5 comparison report.

## Status
Draft.
