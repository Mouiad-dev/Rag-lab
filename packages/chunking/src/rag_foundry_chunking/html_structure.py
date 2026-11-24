"""HTML extraction block structure chunker (C09)."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_techniques import ChunkInput, Fidelity, PluginKind, PluginManifest

from rag_foundry_chunking.boundaries import trim
from rag_foundry_chunking.structure import (
    ContextSpan,
    SkippedRange,
    StructuredChunk,
    StructureParseError,
    StructureResult,
    pack_units,
)


class HtmlStructureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_size: int = Field(default=1200, ge=32, le=8000)
    oversized: Literal["split", "keep"] = "split"
    drop_repeated_boilerplate: bool = True
    boilerplate_max_chars: int = Field(default=120, ge=1, le=2000)
    boilerplate_min_repeats: int = Field(default=2, ge=2, le=100)


class HtmlStructureChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C09",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=HtmlStructureConfig,
    )

    def __init__(self, config: HtmlStructureConfig) -> None:
        self.config = config

    def chunk_structured(self, doc: ChunkInput) -> StructureResult:
        text = doc.text
        blocks = doc.blocks
        if not blocks or any(block.kind is None for block in blocks):
            raise StructureParseError("missing_structure", "C09 needs extraction blocks with kinds")
        normalized = [" ".join(text[b.start : b.end].split()) for b in blocks]
        counts = Counter(
            value
            for value, block in zip(normalized, blocks, strict=True)
            if block.kind != "heading"
            and len(text[block.start : block.end].strip()) <= self.config.boilerplate_max_chars
        )
        skip = set()
        skipped: list[SkippedRange] = []
        seen: set[str] = set()
        for i, (block, value) in enumerate(zip(blocks, normalized, strict=True)):
            if (
                self.config.drop_repeated_boilerplate
                and block.kind != "heading"
                and len(text[block.start : block.end].strip()) <= self.config.boilerplate_max_chars
                and counts[value] >= self.config.boilerplate_min_repeats
            ):
                # Keep the first source occurrence, suppress subsequent copies.
                if value in seen:
                    skip.add(i)
                    skipped.append(
                        SkippedRange(
                            start=block.start, end=block.end, reason="repeated_boilerplate", block_id=block.block_id
                        )
                    )
                else:
                    seen.add(value)
        headings: list[tuple[tuple[str, ...], int, int, str]] = []
        for block in blocks:
            if block.kind == "heading":
                body = trim(text, block.start, block.end)
                if body:
                    headings.append((block.section_path, body[0], body[1], text[body[0] : body[1]]))
        heading_lookup: dict[tuple[tuple[str, ...], str], list[tuple[tuple[str, ...], int, int, str]]] = {}
        for heading in headings:
            heading_lookup.setdefault((heading[0], heading[3]), []).append(heading)
        heading_offsets = {key: [heading[1] for heading in values] for key, values in heading_lookup.items()}
        chunks: list[StructuredChunk] = []
        group: list[int] = []
        group_path: tuple[str, ...] = ()

        def emit(indices: list[int], path: tuple[str, ...]) -> None:
            units = []
            for idx in indices:
                block = blocks[idx]
                span = trim(text, block.start, block.end)
                if span:
                    units.append((idx, span[0], span[1]))
            cursor = 0
            for a, b, level, flags in pack_units(
                text, [(x, y) for _i, x, y in units], self.config.chunk_size, self.config.oversized
            ):
                while units[cursor][2] <= a:
                    cursor += 1
                stop = cursor
                while stop < len(units) and units[stop][2] <= b:
                    stop += 1
                used = units[cursor:stop] or [units[cursor]]
                ids = tuple(blocks[item[0]].block_id for item in used)
                context_heads = []
                for depth in range(1, len(path) + 1):
                    title_path = path[:depth]
                    key = (title_path, path[depth - 1])
                    candidates = heading_lookup.get(key, [])
                    heading_index = bisect_right(heading_offsets.get(key, []), a - 1) - 1
                    candidate = candidates[heading_index] if heading_index >= 0 else None
                    if candidate and (candidate[1], candidate[2]) not in [(v[1], v[2]) for v in context_heads]:
                        context_heads.append(candidate)
                context = tuple(
                    ContextSpan(start=h[1], end=h[2], role="heading")
                    for h in sorted(context_heads, key=lambda h: h[1])
                    if h[2] <= a
                )
                kinds = [blocks[item[0]].kind for item in used]
                chunks.append(
                    StructuredChunk(
                        start=a,
                        end=b,
                        split_level=level,
                        kind="code" if all(k == "code" for k in kinds) else "section",
                        path=path,
                        context=context,
                        block_ids=ids,
                        flags=flags,
                    )
                )

        for i, block in enumerate(blocks):
            if i in skip:
                if group:
                    emit(group, group_path)
                    group = []
                continue
            path = block.section_path
            if block.kind == "heading" or (group and path != group_path):
                if group:
                    emit(group, group_path)
                    group = []
            if not group:
                group_path = path
            group.append(i)
        if group:
            emit(group, group_path)
        chunks.sort(key=lambda chunk: chunk.start)
        return StructureResult(chunks=tuple(chunks), skipped=tuple(skipped))

    def chunk(self, doc: ChunkInput):
        return tuple(chunk.to_span() for chunk in self.chunk_structured(doc).chunks)
