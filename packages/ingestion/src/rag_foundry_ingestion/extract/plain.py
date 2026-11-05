"""Plain text extractor (`txt`, version 1): paragraphs are runs of non-blank lines."""

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


class PlainTextExtractor:
    name = "txt"
    version = "1"
    media_types = (MediaType.TEXT_PLAIN,)

    def extract(self, data: bytes, *, text_encoding: str | None, limits: ExtractionLimits) -> ExtractionResult:
        raw = decode_text(data, text_encoding, limits)
        blocks: list[RawBlock] = []
        run: tuple[int, int] | None = None
        for start, end in split_lines(raw):
            if raw[start:end].strip():
                run = (run[0] if run else start, end)
            elif run:
                blocks.append(RawBlock(BlockKind.PARAGRAPH, *run))
                run = None
        if run:
            blocks.append(RawBlock(BlockKind.PARAGRAPH, *run))
        return finish(
            raw,
            OffsetMap.identity(len(raw)),
            blocks,
            extractor=self.name,
            extractor_version=self.version,
            source_space="decoded_text",
        )


EXTRACTOR = PlainTextExtractor()
register(EXTRACTOR)
