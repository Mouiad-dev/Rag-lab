"""Markdown heading-section chunker (C08)."""

from __future__ import annotations

import re
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_techniques import ChunkInput, Fidelity, PluginKind, PluginManifest

from rag_foundry_chunking.boundaries import trim
from rag_foundry_chunking.paragraph import _blank_line_ranges
from rag_foundry_chunking.structure import ContextSpan, StructuredChunk, StructureResult, line_ranges, pack_units


class MarkdownSectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_size: int = Field(default=1200, ge=32, le=8000)
    oversized: Literal["split", "keep"] = "split"
    max_heading_level: int = Field(default=6, ge=1, le=6)


_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:\s+(.*)|\s*)$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_CLOSING = re.compile(r"(?:^|[ \t]+)#+$")


class MarkdownSectionChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C08",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=MarkdownSectionConfig,
    )

    def __init__(self, config: MarkdownSectionConfig) -> None:
        self.config = config

    def chunk_structured(self, doc: ChunkInput) -> StructureResult:
        text = doc.text
        lines = [(a, b, text[a:b]) for a, b, _ in line_ranges(text)]
        headings = []
        fences = []
        i = 0
        while i < len(lines):
            start, end, line = lines[i]
            match = _FENCE.match(line)
            if match:
                marker = match.group(1)
                j = i + 1
                while j < len(lines):
                    candidate = lines[j][2]
                    close = _FENCE.match(candidate)
                    if (
                        close
                        and close.group(1)[0] == marker[0]
                        and len(close.group(1)) >= len(marker)
                        and not candidate[close.end() :].strip()
                    ):
                        break
                    j += 1
                last = j if j < len(lines) else len(lines) - 1
                fences.append((start, lines[last][1]))
                i = last + 1
                continue
            m = _HEADING.match(line)
            if m and len(m.group(1)) <= self.config.max_heading_level:
                level = len(m.group(1))
                title = _CLOSING.sub("", m.group(2) or "").strip()
                headings.append((i, start, end, level, title))
            i += 1

        boundaries = [(idx, s, e, level, title) for idx, s, e, level, title in headings]
        sections = []
        cursor = 0
        stack: list[tuple[int, str, int, int]] = []  # ancestors: (level, title, line start, line end)
        for hidx, (_lineidx, start, end, level, title) in enumerate(boundaries):
            if cursor < start:
                sections.append((cursor, start, None, (), ()))
            while stack and stack[-1][0] >= level:
                stack.pop()
            path = tuple(item[1] for item in stack) + (title,)
            ctx = tuple(ContextSpan(start=item[2], end=item[3], role="heading") for item in stack)
            next_start = boundaries[hidx + 1][1] if hidx + 1 < len(boundaries) else len(text)
            sections.append((start, next_start, (level, end), path, ctx))
            cursor = next_start
            stack.append((level, title, *trim(text, start, end)))
        if not headings:
            sections = [(0, len(text), None, (), ())]
        chunks = []
        size = self.config.chunk_size
        pending: int | None = None  # start of heading-only parents waiting to merge into a child section
        for index, (start, end, own, path, context) in enumerate(sections):
            item = trim(text, start, end)
            if item is None:
                continue
            start, end = item
            following = sections[index + 1] if index + 1 < len(sections) else None
            if own and end <= own[1] and following and following[2] and following[2][0] > own[0]:
                # Heading-only section followed by a child: try to merge it into the child's first chunk.
                pending = start if pending is None else pending
                continue
            attrs = (("heading_level", str(own[0])),) if own else ()
            if pending is not None:
                merged_start = pending
                pending = None
                if end - merged_start <= size:
                    context = tuple(span for span in context if span.end <= merged_start)
                    chunks.append(
                        StructuredChunk(
                            start=merged_start,
                            end=end,
                            split_level=0,
                            kind="section",
                            path=path,
                            context=context,
                            attrs=attrs,
                        )
                    )
                    continue
                for heading_start, heading_end in _heading_lines(text, merged_start, start):
                    chunks.append(
                        StructuredChunk(
                            start=heading_start,
                            end=heading_end,
                            split_level=0,
                            kind="section",
                            path=_path_at(text, sections, heading_start),
                            context=tuple(span for span in context if span.end <= heading_start),
                        )
                    )
            if end - start <= size:
                chunks.append(
                    StructuredChunk(
                        start=start, end=end, split_level=0, kind="section", path=path, context=context, attrs=attrs
                    )
                )
                continue
            # Paragraphs and complete fences are indivisible units where possible.
            units = []
            cursor = start
            for fs, fe in fences:
                if fs < start or fe > end:
                    continue
                units.extend(_blank_line_ranges(text, cursor, fs))
                units.append((fs, fe))
                cursor = fe
            units.extend(_blank_line_ranges(text, cursor, end))
            for a, b, lvl, flags in pack_units(text, units, size, self.config.oversized):
                ctx = context
                if own and a > own[1]:
                    ctx += (ContextSpan(start=start, end=own[1], role="heading"),)
                code = any(a == fs and b == fe for fs, fe in fences)
                chunks.append(
                    StructuredChunk(
                        start=a,
                        end=b,
                        split_level=lvl,
                        kind="code" if code else "section",
                        path=path,
                        context=ctx,
                        attrs=attrs,
                        flags=flags,
                    )
                )
        chunks.sort(key=lambda c: c.start)
        return StructureResult(chunks=tuple(chunks))

    def chunk(self, doc: ChunkInput):
        return tuple(c.to_span() for c in self.chunk_structured(doc).chunks)


def _heading_lines(text: str, start: int, end: int) -> list[tuple[int, int]]:
    return [r for r in (trim(text, a, e) for a, e, _ in line_ranges(text, start, end)) if r is not None]


def _path_at(text: str, sections, offset: int) -> tuple[str, ...]:
    return next((path for start, end, own, path, _ctx in sections if own and trim(text, start, end)[0] == offset), ())
