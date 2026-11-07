"""Shared contracts and helpers for structure-aware chunkers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rag_foundry_techniques import ChunkSpan

from rag_foundry_chunking.boundaries import pack, safe_cut_ranges, trim, word_ranges


def line_ranges(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int, int]]:
    """Return physical line (start, content end, next start) offsets."""
    end = len(text) if end is None else end
    result = []
    cursor = start
    while cursor < end:
        i = cursor
        while i < end and text[i] not in "\r\n":
            i += 1
        content_end = i
        if i < end and text[i] == "\r" and i + 1 < end and text[i + 1] == "\n":
            i += 2
        elif i < end:
            i += 1
        result.append((cursor, content_end, i))
        cursor = i
    if start == end:
        return []
    return result


class ContextSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    role: Literal["heading", "table_header", "declaration", "container"]

    @field_validator("end")
    @classmethod
    def _after_start(cls, value, info):
        if value <= info.data.get("start", -1):
            raise ValueError("end must be greater than start")
        return value


class StructuredChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    split_level: int = Field(ge=0, le=63)
    kind: Literal["section", "table", "text", "code", "json"]
    path: tuple[str, ...] = ()
    context: tuple[ContextSpan, ...] = ()
    block_ids: tuple[str, ...] = ()
    attrs: tuple[tuple[str, str], ...] = ()
    flags: tuple[Literal["oversized", "split_part", "unparsed"], ...] = ()

    @field_validator("end")
    @classmethod
    def _after_start(cls, value, info):
        if value <= info.data.get("start", -1):
            raise ValueError("end must be greater than start")
        return value

    @model_validator(mode="after")
    def _metadata_order(self):
        if tuple(sorted(self.attrs)) != self.attrs or len({key for key, _ in self.attrs}) != len(self.attrs):
            raise ValueError("attrs must have unique, sorted keys")
        if tuple(sorted(set(self.flags))) != self.flags:
            raise ValueError("flags must be unique and sorted")
        previous = 0
        for span in self.context:
            if span.end > self.start or span.start < previous:
                raise ValueError("context spans must be ascending, non-overlapping and before start")
            previous = span.end
        return self

    def to_span(self) -> ChunkSpan:
        return ChunkSpan(start=self.start, end=self.end, split_level=self.split_level)

    def attr(self, key: str) -> str | None:
        return next((value for name, value in self.attrs if name == key), None)


class SkippedRange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    reason: str
    block_id: str | None = None

    @field_validator("end")
    @classmethod
    def _after_start(cls, value, info):
        if value <= info.data.get("start", -1):
            raise ValueError("end must be greater than start")
        return value


class StructureResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunks: tuple[StructuredChunk, ...]
    skipped: tuple[SkippedRange, ...] = ()


def render(text: str, chunk: StructuredChunk, separator: str = "\n") -> str:
    return separator.join([*(text[item.start : item.end] for item in chunk.context), text[chunk.start : chunk.end]])


def split_oversized(text: str, start: int, end: int, size: int, level: int) -> list[tuple[int, int, int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    lines: list[tuple[int, int, int]] = []
    for cursor, content_end, _next_start in line_ranges(text, start, end):
        item = trim(text, cursor, content_end)
        if item:
            a, b = item
            if b - a <= size:
                lines.append((a, b, level))
            else:
                for wa, wb in word_ranges(text, a, b):
                    if wb - wa <= size:
                        lines.append((wa, wb, level + 1))
                    else:
                        lines.extend((x, y, level + 2) for x, y in safe_cut_ranges(text, wa, wb, size))
    return pack(text, lines, size, overlap_units=0)


Flags = tuple[Literal["oversized", "split_part", "unparsed"], ...]


def pack_units(
    text: str, units: list[tuple[int, int]], size: int, oversized: Literal["split", "keep"]
) -> list[tuple[int, int, int, Flags]]:
    """Pack whole units within `size`; an oversized unit is kept whole or split, and always flagged."""
    output: list[tuple[int, int, int, Flags]] = []
    buffer: list[tuple[int, int, int]] = []

    def flush() -> None:
        output.extend((a, b, level, ()) for a, b, level in pack(text, buffer, size, overlap_units=0))
        buffer.clear()

    for a, b in units:
        if b - a <= size:
            buffer.append((a, b, 0))
            continue
        flush()
        if oversized == "keep":
            output.append((a, b, 0, ("oversized",)))
        else:
            output.extend(
                (x, y, level, ("oversized", "split_part")) for x, y, level in split_oversized(text, a, b, size, 1)
            )
    flush()
    return output
