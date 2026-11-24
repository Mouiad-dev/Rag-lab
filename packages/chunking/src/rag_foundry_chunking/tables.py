"""Pipe and TSV table chunker (C12)."""

from __future__ import annotations

import re
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_techniques import ChunkInput, Fidelity, PluginKind, PluginManifest

from rag_foundry_chunking.boundaries import trim
from rag_foundry_chunking.paragraph import _blank_line_ranges
from rag_foundry_chunking.structure import (
    ContextSpan,
    StructuredChunk,
    StructureResult,
    line_ranges,
    pack_units,
    split_oversized,
)


class TableConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_size: int = Field(default=1200, ge=32, le=8000)
    oversized: Literal["split", "keep"] = "split"
    rows_per_chunk: int = Field(default=20, ge=1, le=1000)


class TableCell(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    row: int = Field(ge=1)
    column: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    row_label: str
    column_label: str


_DELIMITER = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")


def _escaped(text: str, i: int, start: int) -> bool:
    backslashes = 0
    while i - 1 - backslashes >= start and text[i - 1 - backslashes] == "\\":
        backslashes += 1
    return backslashes % 2 == 1


def _cells(text: str, start: int, end: int, fmt: str) -> list[tuple[int, int]]:
    """Positional cells of one row; an empty cell is `(x, x)`, so column numbers stay true."""
    item = trim(text, start, end)
    if item is None:
        return []
    left, right = item
    if fmt == "pipe":
        if text[left] == "|":
            left += 1
        if right > left and text[right - 1] == "|" and not _escaped(text, right - 1, left):
            right -= 1
        cuts = [i for i in range(left, right) if text[i] == "|" and not _escaped(text, i, left)]
    else:
        cuts = [i for i in range(left, right) if text[i] == "\t"]
    cells = []
    cursor = left
    for cut in [*cuts, right]:
        trimmed = trim(text, cursor, cut)
        cells.append(trimmed if trimmed is not None else (cursor, cursor))
        cursor = cut + 1
    return cells


def _tsv_tabs(line: str) -> int:
    return line.lstrip().count("\t")


class TableChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C12",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=TableConfig,
    )

    def __init__(self, config: TableConfig) -> None:
        self.config = config

    def _find_tables(self, text: str, lines: list[tuple[int, int, int]]):
        tables = []
        i = 0
        while i < len(lines):
            a, e, _ = lines[i]
            line = text[a:e]
            if i + 2 < len(lines) and "|" in line:
                da, de, _ = lines[i + 1]
                if _DELIMITER.match(text[da:de]) and len(_cells(text, a, e, "pipe")) == len(
                    _cells(text, da, de, "pipe")
                ):
                    j = i + 2
                    while (
                        j < len(lines)
                        and "|" in text[lines[j][0] : lines[j][1]]
                        and text[lines[j][0] : lines[j][1]].strip()
                    ):
                        j += 1
                    if j > i + 2:
                        tables.append(("pipe", i, j))
                        i = j
                        continue
            tabs = _tsv_tabs(line)
            if tabs:
                j = i + 1
                while j < len(lines) and _tsv_tabs(text[lines[j][0] : lines[j][1]]) == tabs:
                    j += 1
                if j > i + 1:
                    tables.append(("tsv", i, j))
                    i = j
                    continue
            i += 1
        return tables

    def chunk_structured(self, doc: ChunkInput) -> StructureResult:
        text = doc.text
        size = self.config.chunk_size
        lines = line_ranges(text)
        chunks: list[StructuredChunk] = []
        cursor = 0
        for table_index, (fmt, first, stop) in enumerate(self._find_tables(text, lines)):
            self._text_chunks(text, cursor, lines[first][0], chunks)
            cursor = lines[stop - 1][2]
            header = trim(text, lines[first][0], lines[first][1])
            header_ctx = [ContextSpan(start=header[0], end=header[1], role="table_header")]
            body_first = first + 1
            if fmt == "pipe":
                delimiter = trim(text, lines[first + 1][0], lines[first + 1][1])
                header_ctx.append(ContextSpan(start=delimiter[0], end=delimiter[1], role="table_header"))
                body_first = first + 2
            header_end = header_ctx[-1].end
            base = (
                ("columns", str(len(_cells(text, header[0], header[1], fmt)))),
                ("format", fmt),
                ("table_index", str(table_index)),
                ("table_start", str(header[0])),
            )
            rows = [r for r in (trim(text, a, e) for a, e, _ in lines[body_first:stop]) if r is not None]
            groups: list[list[tuple[int, int]]] = []
            for row in rows:
                if groups and (len(groups[-1]) < self.config.rows_per_chunk and row[1] - groups[-1][0][0] <= size):
                    groups[-1].append(row)
                else:
                    groups.append([row])
            row_number = 0
            for group_index, group in enumerate(groups):
                row_first, row_number = row_number + 1, row_number + len(group)
                attrs = tuple(sorted((*base, ("row_first", str(row_first)), ("row_last", str(row_number)))))
                start, end = group[0][0], group[-1][1]
                if len(group) == 1 and end - start > size:
                    if group_index == 0:
                        chunks.append(
                            StructuredChunk(
                                start=header[0],
                                end=header_end,
                                split_level=0,
                                kind="table",
                                attrs=tuple(sorted((*base, ("part", "header")))),
                            )
                        )
                    if self.config.oversized == "keep":
                        pieces = [(start, end, 0, ("oversized",))]
                    else:
                        pieces = [
                            (x, y, level, ("oversized", "split_part"))
                            for x, y, level in split_oversized(text, start, end, size, 1)
                        ]
                    for x, y, level, flags in pieces:
                        chunks.append(
                            StructuredChunk(
                                start=x,
                                end=y,
                                split_level=level,
                                kind="table",
                                context=tuple(header_ctx),
                                attrs=attrs,
                                flags=flags,
                            )
                        )
                    continue
                context = () if group_index == 0 else tuple(header_ctx)
                chunks.append(
                    StructuredChunk(
                        start=header[0] if group_index == 0 else start,
                        end=end,
                        split_level=0,
                        kind="table",
                        context=context,
                        attrs=attrs,
                    )
                )
        self._text_chunks(text, cursor, len(text), chunks)
        return StructureResult(chunks=tuple(chunks))

    def _text_chunks(self, text: str, start: int, end: int, chunks: list[StructuredChunk]) -> None:
        units = _blank_line_ranges(text, start, end)
        for a, b, level, flags in pack_units(text, units, self.config.chunk_size, self.config.oversized):
            chunks.append(StructuredChunk(start=a, end=b, split_level=level, kind="text", flags=flags))

    def chunk(self, doc: ChunkInput):
        return tuple(c.to_span() for c in self.chunk_structured(doc).chunks)


def table_cells(text: str, chunk: StructuredChunk) -> tuple[TableCell, ...]:
    """Cells of a table chunk's body rows, labelled from the source header; `()` for parts that are not whole rows."""
    if chunk.kind != "table" or "split_part" in chunk.flags or chunk.attr("part") == "header":
        return ()
    fmt = chunk.attr("format") or "pipe"
    lines = [(a, e) for a, e, _ in line_ranges(text, chunk.start, chunk.end)]
    if chunk.context:
        header = (chunk.context[0].start, chunk.context[0].end)
        body = lines
    else:
        header = lines[0]
        body = lines[2:] if fmt == "pipe" else lines[1:]
    labels = [text[a:b] for a, b in _cells(text, header[0], header[1], fmt)]
    row = int(chunk.attr("row_first") or "1")
    result = []
    for a, e in body:
        cells = _cells(text, a, e, fmt)
        if not cells:
            continue
        row_label = text[cells[0][0] : cells[0][1]]
        for column, (x, y) in enumerate(cells, 1):
            if x == y:
                continue
            label = labels[column - 1] if column <= len(labels) else ""
            result.append(TableCell(row=row, column=column, start=x, end=y, row_label=row_label, column_label=label))
        row += 1
    return tuple(result)
