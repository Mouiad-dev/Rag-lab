"""JSON member and element chunking (C10)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_techniques import ChunkInput, Fidelity, PluginKind, PluginManifest

from rag_foundry_chunking.structure import (
    ContextSpan,
    StructuredChunk,
    StructureParseError,
    StructureResult,
    pack_units,
)


class JsonStructureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_size: int = Field(default=1200, ge=32, le=8000)
    oversized: Literal["split", "keep"] = "split"
    max_depth: int = Field(default=64, ge=1, le=256)


@dataclass
class _Entry:
    start: int
    end: int
    path: str
    key: str | None = None
    index: int | None = None
    key_start: int | None = None
    key_end: int | None = None
    children: list[_Entry] | None = None
    container: bool = False
    value_start: int | None = None


def _path_key(parent: str, key: str) -> str:
    return parent + "['" + key.replace("\\", "\\\\").replace("'", "\\'") + "']"


def _walk(text: str, max_depth: int):
    decoder = json.JSONDecoder(parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    n = len(text)

    def ws(pos: int) -> int:
        while pos < n and text[pos] in " \t\r\n":
            pos += 1
        return pos

    def value(pos: int, path: str, depth: int):
        pos = ws(pos)
        if depth > max_depth:
            raise StructureParseError("too_deep")
        if pos >= n:
            raise ValueError("missing value")
        if text[pos] == "{":
            children: list[_Entry] = []
            cursor = ws(pos + 1)
            if cursor < n and text[cursor] == "}":
                return _Entry(pos, cursor + 1, path, children=children, container=True), cursor + 1
            while True:
                cursor = ws(cursor)
                if cursor >= n or text[cursor] != '"':
                    raise ValueError("object key expected")
                key_start = cursor
                key_value, key_end = decoder.raw_decode(text, cursor)
                if not isinstance(key_value, str):
                    raise ValueError("object key expected")
                cursor = ws(key_end)
                if cursor >= n or text[cursor] != ":":
                    raise ValueError("colon expected")
                child, cursor = value(cursor + 1, _path_key(path, key_value), depth + 1)
                children.append(
                    _Entry(
                        key_start,
                        child.end,
                        child.path,
                        key_value,
                        key_start=key_start,
                        key_end=key_end,
                        children=child.children,
                        container=child.container,
                        value_start=child.start,
                    )
                )
                cursor = ws(cursor)
                if cursor < n and text[cursor] == ",":
                    cursor += 1
                    continue
                if cursor < n and text[cursor] == "}":
                    return _Entry(pos, cursor + 1, path, children=children, container=True), cursor + 1
                raise ValueError("object end expected")
        if text[pos] == "[":
            children = []
            cursor = ws(pos + 1)
            if cursor < n and text[cursor] == "]":
                return _Entry(pos, cursor + 1, path, children=children, container=True), cursor + 1
            index = 0
            while True:
                child, cursor = value(cursor, f"{path}[{index}]", depth + 1)
                children.append(
                    _Entry(
                        child.start,
                        child.end,
                        child.path,
                        index=index,
                        children=child.children,
                        container=child.container,
                    )
                )
                cursor = ws(cursor)
                if cursor < n and text[cursor] == ",":
                    cursor += 1
                    index += 1
                    continue
                if cursor < n and text[cursor] == "]":
                    return _Entry(pos, cursor + 1, path, children=children, container=True), cursor + 1
                raise ValueError("array end expected")
        parsed, end = decoder.raw_decode(text, pos)
        return _Entry(pos, end, path), end

    try:
        root, end = value(0, "$", 0)
        if ws(end) != n:
            raise ValueError("trailing data")
        # Keep stdlib semantics as the definitive validity check.
        json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        return root
    except StructureParseError:
        raise
    except (ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise StructureParseError("invalid_json", str(exc)) from exc


def _entries(root: _Entry):
    yield root
    if root.children:
        for child in root.children:
            yield from _entries(child)


class JsonStructureChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C10",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=JsonStructureConfig,
    )

    def __init__(self, config: JsonStructureConfig) -> None:
        self.config = config

    def chunk_structured(self, doc: ChunkInput) -> StructureResult:
        text = doc.text
        size = self.config.chunk_size
        root = _walk(text, self.config.max_depth)
        chunks: list[StructuredChunk] = []
        # Start of the key tokens of split members not yet in any chunk body; the next chunk absorbs them.
        lead: list[int] = []

        def key_context(entry: _Entry) -> ContextSpan:
            end = entry.key_end
            while text[end].isspace():
                end += 1
            return ContextSpan(start=entry.key_start, end=end + 1, role="container")  # through the ':'

        def emit(a: int, b: int, level: int, flags, node: _Entry, units: list[_Entry], ancestors: list[_Entry]):
            if lead:
                if b - lead[0] <= size or flags == ("oversized",):
                    a = lead[0]
                else:
                    # The key tokens do not fit with the first unit: they become a chunk of their own.
                    key_end = key_context(ancestors[-1]).end
                    chunks.append(
                        StructuredChunk(
                            start=lead[0],
                            end=key_end,
                            split_level=level,
                            kind="json",
                            path=(node.path,),
                            context=tuple(key_context(e) for e in ancestors if key_context(e).end <= lead[0]),
                            attrs=(("container_path", node.path), ("part", "keys")),
                        )
                    )
                lead.clear()
            first, last = units[0], units[-1]
            attrs = [("container_path", node.path)]
            if first.key is not None:
                attrs += [("first_key", first.key), ("last_key", last.key)]
            elif first.index is not None:
                attrs += [("first_index", str(first.index)), ("last_index", str(last.index))]
            chunks.append(
                StructuredChunk(
                    start=a,
                    end=b,
                    split_level=level,
                    kind="json",
                    path=(first.path,) if first is last else (first.path, last.path),
                    context=tuple(key_context(e) for e in ancestors if key_context(e).end <= a),
                    attrs=tuple(sorted(attrs)),
                    flags=flags,
                )
            )

        def split(entry: _Entry) -> bool:
            return entry.container and entry.end - entry.start > size and bool(entry.children)

        def process(node: _Entry, ancestors: list[_Entry], level: int) -> None:
            children = node.children or []
            i = 0
            while i < len(children):
                if split(children[i]):
                    child = children[i]
                    if child.key_start is not None and not lead:
                        lead.append(child.key_start)
                    process(child, ancestors + ([child] if child.key_start is not None else []), level + 1)
                    i += 1
                    continue
                group = []
                while i < len(children) and not split(children[i]):
                    group.append(children[i])
                    i += 1
                cursor = 0  # pieces come in order, so a moving index finds their units in linear time
                for a, b, piece_level, flags in pack_units(
                    text, [(u.start, u.end) for u in group], size, self.config.oversized
                ):
                    while group[cursor].end <= a:
                        cursor += 1
                    stop = cursor
                    while stop < len(group) and group[stop].end <= b:
                        stop += 1
                    units = group[cursor:stop] or [group[cursor]]
                    emit(a, b, level + piece_level, flags, node, units, ancestors)

        if split(root):
            process(root, [], 0)
        else:
            # The whole document fits, or is a scalar / empty container: one unit.
            for a, b, level, flags in pack_units(text, [(root.start, root.end)], size, self.config.oversized):
                emit(a, b, level, flags, root, [root], [])
        return StructureResult(chunks=tuple(chunks))

    def chunk(self, doc: ChunkInput):
        return tuple(chunk.to_span() for chunk in self.chunk_structured(doc).chunks)


def json_path_at(text: str, result: StructureResult, offset: int) -> str:
    root = _walk(text, 256)
    matches = [entry for entry in _entries(root) if entry.start <= offset < entry.end]
    return min(matches, key=lambda entry: entry.end - entry.start).path if matches else "$"
