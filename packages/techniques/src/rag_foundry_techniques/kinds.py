"""Typed contracts shared by technique kinds."""

from __future__ import annotations

from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_foundry_techniques.protocol import PluginManifest


class TextBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    block_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @field_validator("end")
    @classmethod
    def end_after_start(cls, value: int, info):
        start = info.data.get("start")
        if start is not None and value <= start:
            raise ValueError("end must be greater than start")
        return value


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


class Chunker(Protocol):
    manifest: ClassVar[PluginManifest]
    config: BaseModel

    def __init__(self, config: BaseModel) -> None: ...

    def chunk(self, doc: ChunkInput) -> tuple[ChunkSpan, ...]: ...
