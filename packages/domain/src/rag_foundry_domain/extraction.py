"""Extraction views and source blocks (ADR-0010). Views are immutable and owned by one DocumentVersion."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_foundry_domain.documents import Sha256Hex, UtcDatetime, Uuid4

_FROZEN = ConfigDict(frozen=True, extra="forbid")

ExtractorName = Annotated[str, Field(pattern=r"^[a-z0-9._-]{1,64}$")]
ExtractorVersion = Annotated[str, Field(min_length=1, max_length=128)]
ProfileId = Annotated[str, Field(min_length=1, max_length=128)]
UnicodeVersion = Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
WarningCode = Annotated[str, Field(pattern=r"^[a-z0-9_:]{1,64}$")]


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CODE = "code"
    PAGE = "page"


class SourceBlock(BaseModel):
    """A half-open range of code points in the canonical text. `block_text = canonical[start:end]`."""

    model_config = _FROZEN
    block_id: Annotated[str, Field(pattern=r"^b[0-9]{5,8}$")]
    ordinal: Annotated[int, Field(ge=0)]
    kind: BlockKind
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(ge=0)]
    section_path: tuple[str, ...] = ()
    heading_level: Annotated[int, Field(ge=1, le=6)] | None = None
    page_index: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "SourceBlock":
        if self.block_id != f"b{self.ordinal:05d}":
            raise ValueError("block_id must be b + zero-padded ordinal")
        if self.end < self.start:
            raise ValueError("end must be >= start")
        if (self.kind is BlockKind.HEADING) != (self.heading_level is not None):
            raise ValueError("heading_level is required for headings and allowed only for headings")
        return self


class ExtractionKey(BaseModel):
    """Identity of a view (ADR-0010): the same key returns the existing view."""

    model_config = _FROZEN
    document_version_id: Uuid4
    extractor: ExtractorName
    extractor_version: ExtractorVersion
    profile_id: ProfileId
    unicode_version: UnicodeVersion


class NewExtraction(ExtractionKey):
    """Everything known about an extraction before the repository assigns an ID."""

    document_id: Uuid4
    canonical_sha256: Sha256Hex
    char_count: Annotated[int, Field(ge=1)]
    block_count: Annotated[int, Field(ge=1)]
    page_count: Annotated[int, Field(ge=1)] | None = None
    payload_sha256: Sha256Hex
    payload_key: Annotated[str, Field(min_length=1)]
    payload_schema_version: Literal[1] = 1
    warnings: tuple[WarningCode, ...] = ()

    @property
    def key(self) -> ExtractionKey:
        return ExtractionKey(
            document_version_id=self.document_version_id,
            extractor=self.extractor,
            extractor_version=self.extractor_version,
            profile_id=self.profile_id,
            unicode_version=self.unicode_version,
        )


class ExtractionView(NewExtraction):
    id: Uuid4
    tenant_id: Uuid4
    project_id: Uuid4
    created_at: UtcDatetime


class ExtractionOutcome(BaseModel):
    model_config = _FROZEN
    view: ExtractionView
    created: bool
