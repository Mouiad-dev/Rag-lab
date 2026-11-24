"""Markdown extractor (`markdown`, version 1): stdlib line scanner, markup kept verbatim.

Known limits: setext headings, lists, tables, block quotes and indented code are paragraphs.
"""

import re

from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.extraction import BlockKind
from rag_foundry_domain.offsets import OffsetMap

from rag_foundry_ingestion.extract.base import (
    ExtractionLimits,
    ExtractionResult,
    RawBlock,
    decode_text,
    finish,
    register,
    split_lines,
)

_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t](.*))?$")
_CLOSING = re.compile(r"(?:^|[ \t]+)#+$")
_FENCE = re.compile(r"^ {0,3}(`{3,}(?![^`]*`)|~{3,})")


def _title(rest: str | None) -> str:
    text = (rest or "").strip()
    return _CLOSING.sub("", text).strip()


class MarkdownExtractor:
    name = "markdown"
    version = "1"
    media_types = (MediaType.TEXT_MARKDOWN,)

    def extract(self, data: bytes, *, text_encoding: str | None, limits: ExtractionLimits) -> ExtractionResult:
        raw = decode_text(data, text_encoding, limits)
        lines = split_lines(raw)
        blocks: list[RawBlock] = []
        path: list[tuple[int, str]] = []  # (level, title)
        run: tuple[int, int] | None = None

        def current() -> tuple[str, ...]:
            return tuple(t for _, t in path)

        def flush() -> None:
            nonlocal run
            if run:
                blocks.append(RawBlock(BlockKind.PARAGRAPH, *run, section_path=current()))
                run = None

        def text(idx: int) -> str:
            return raw[lines[idx][0] : lines[idx][1]]

        i = 0
        while i < len(lines):
            start, end = lines[i]
            line = raw[start:end]
            fence = _FENCE.match(line)
            if fence:
                flush()
                marker = fence.group(1)
                j = i + 1
                closed = False
                while j < len(lines):
                    cand = text(j)
                    m = _FENCE.match(cand)
                    if (
                        m
                        and m.group(1)[0] == marker[0]
                        and len(m.group(1)) >= len(marker)
                        and not cand[m.end() :].strip()
                    ):
                        closed = True
                        break
                    j += 1
                if closed:
                    last = j
                else:
                    last = len(lines) - 1
                    while last > i and not text(last).strip():
                        last -= 1
                blocks.append(RawBlock(BlockKind.CODE, start, lines[last][1], section_path=current()))
                i = (j if closed else len(lines) - 1) + 1
                continue
            atx = _ATX.match(line)
            if atx:
                flush()
                level = len(atx.group(1))
                while path and path[-1][0] >= level:
                    path.pop()
                path.append((level, _title(atx.group(2))))
                blocks.append(RawBlock(BlockKind.HEADING, start, end, section_path=current(), heading_level=level))
            elif line.strip():
                run = (run[0] if run else start, end)
            else:
                flush()
            i += 1
        flush()
        return finish(
            raw,
            OffsetMap.identity(len(raw)),
            blocks,
            extractor=self.name,
            extractor_version=self.version,
            source_space="decoded_text",
        )


EXTRACTOR = MarkdownExtractor()
register(EXTRACTOR)
