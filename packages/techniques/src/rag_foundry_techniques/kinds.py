"""Typed contracts shared by technique kinds."""

from __future__ import annotations

from typing import ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_foundry_techniques.protocol import PluginManifest


class TextBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    block_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    kind: Literal["heading", "paragraph", "code", "page"] | None = None
    section_path: tuple[str, ...] = ()
    heading_level: int | None = Field(default=None, ge=1, le=6)
    page_index: int | None = Field(default=None, ge=0)

    @field_validator("end")
    @classmethod
    def end_after_start(cls, value: int, info):
        start = info.data.get("start")
        if start is not None and value <= start:
            raise ValueError("end must be greater than start")
        return value

    @model_validator(mode="after")
    def _consistent(self):
        if (self.kind == "heading") != (self.heading_level is not None):
            raise ValueError("heading_level is required for headings and allowed only for headings")
        return self


class ChunkInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    blocks: tuple[TextBlock, ...] = ()

    @field_validator("blocks")
    @classmethod
    def ordered_within_text(cls, blocks: tuple[TextBlock, ...], info):
        text = info.data.get("text")
        previous_end = 0
        for block in blocks:
            if block.start < previous_end:
                raise ValueError("blocks must be ascending and non-overlapping")
            if text is not None and block.end > len(text):
                raise ValueError("block end must not exceed text length")
            previous_end = block.end
        return blocks


class ChunkSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    split_level: int = Field(ge=0, le=63)

    @field_validator("end")
    @classmethod
    def end_after_start(cls, value: int, info):
        start = info.data.get("start")
        if start is not None and value <= start:
            raise ValueError("end must be greater than start")
        return value


class ParentedSpan(ChunkSpan):
    parent_start: int | None = None
    parent_end: int | None = None

    @model_validator(mode="after")
    def _parent_contains_child(self):
        if (self.parent_start is None) != (self.parent_end is None):
            raise ValueError("parent_start and parent_end must both be set or unset")
        if self.parent_start is not None and not 0 <= self.parent_start <= self.start < self.end <= self.parent_end:
            raise ValueError("parent span must contain chunk span")
        return self


class Chunker(Protocol):
    manifest: ClassVar[PluginManifest]
    config: BaseModel

    def __init__(self, config: BaseModel) -> None: ...

    def chunk(self, doc: ChunkInput) -> tuple[ChunkSpan, ...]: ...


@runtime_checkable
class ParentLinkedChunker(Protocol):
    def chunk_with_parents(self, doc: ChunkInput) -> tuple[ParentedSpan, ...]: ...


class ScoreComponent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: float
    term: str | None = None

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        import re

        if re.fullmatch(r"[a-z0-9_]{1,40}", value) is None:
            raise ValueError("invalid score component name")
        return value

    @field_validator("value")
    @classmethod
    def _finite_value(cls, value: float) -> float:
        import math

        if not math.isfinite(value):
            raise ValueError("score component value must be finite")
        return value


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    extraction_view_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    block_ids: tuple[str, ...]
    text: str
    distance: float | None = None
    score: float
    rank: int = Field(ge=1)
    score_components: tuple[ScoreComponent, ...] = ()
    parent_start: int | None = None
    parent_end: int | None = None

    @model_validator(mode="after")
    def _parent_pair(self):
        if (self.parent_start is None) != (self.parent_end is None):
            raise ValueError("parent_start and parent_end must both be set or unset")
        if self.parent_start is not None and not 0 <= self.parent_start <= self.start < self.end <= self.parent_end:
            raise ValueError("parent span must contain chunk span")
        return self


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    technique_id: str
    grounding: Literal["corpus", "none"]
    index_version_id: str | None
    hits: tuple[RetrievedChunk, ...]
    scoring: str | None = None

    @field_validator("index_version_id")
    @classmethod
    def _none_grounding_has_no_index(cls, value, info):
        if info.data.get("grounding") == "none" and value is not None:
            raise ValueError("ungrounded results cannot reference an index")
        return value

    @field_validator("hits")
    @classmethod
    def _none_grounding_has_no_hits(cls, value, info):
        if info.data.get("grounding") == "none" and value:
            raise ValueError("ungrounded results cannot contain hits")
        return value


class Retriever(Protocol):
    async def retrieve(self, query: str) -> RetrievalResult: ...
