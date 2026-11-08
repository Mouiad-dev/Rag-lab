"""Immutable index and chunk records."""

import hashlib
import json
import math
import struct
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_foundry_domain.documents import Sha256Hex, UtcDatetime, Uuid4
from rag_foundry_domain.errors import IndexingError

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class IndexState(StrEnum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class IndexConfig(BaseModel):
    model_config = _FROZEN
    chunker_id: Annotated[str, Field(pattern=r"^[A-Z][0-9]{2}$")]
    chunker_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")]
    chunker_config_json: Annotated[str, Field(min_length=1, max_length=8192)]
    chunker_config_sha256: Sha256Hex
    embedding_profile_id: Annotated[str, Field(min_length=1, max_length=128)]
    embedding_provider: Annotated[str, Field(min_length=1, max_length=64)]
    embedding_model: Annotated[str, Field(min_length=1, max_length=256)]
    embedding_dim: Annotated[int, Field(ge=1, le=16000)]
    distance: Literal["cosine"]
    embedding_key: Sha256Hex

    @model_validator(mode="after")
    def _config_hash(self) -> "IndexConfig":
        if hashlib.sha256(self.chunker_config_json.encode("utf-8")).hexdigest() != self.chunker_config_sha256:
            raise ValueError("chunker config hash mismatch")
        return self


class NewIndexVersion(IndexConfig):
    extraction_view_ids: Annotated[tuple[Uuid4, ...], Field(min_length=1, max_length=100)]
    corpus_sha256: Sha256Hex
    unanchored_chunk_count: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _view_order(self) -> "NewIndexVersion":
        if tuple(sorted(self.extraction_view_ids)) != self.extraction_view_ids or len(
            set(self.extraction_view_ids)
        ) != len(self.extraction_view_ids):
            raise ValueError("view ids must be strictly ascending")
        return self


class IndexVersion(NewIndexVersion):
    id: Uuid4
    tenant_id: Uuid4
    project_id: Uuid4
    state: IndexState
    chunk_count: Annotated[int, Field(ge=0)]
    created_at: UtcDatetime
    ready_at: UtcDatetime | None


class NewChunk(BaseModel):
    model_config = _FROZEN
    extraction_view_id: Uuid4
    ordinal: Annotated[int, Field(ge=0)]
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]
    split_level: Annotated[int, Field(ge=0, le=63)]
    block_ids: Annotated[tuple[str, ...], Field(min_length=1)]
    text: str
    text_sha256: Sha256Hex
    embedding: tuple[float, ...]
    parent_start_cp: int | None = Field(default=None, ge=0)
    parent_end_cp: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_content(self) -> "NewChunk":
        if self.end <= self.start or len(self.text) != self.end - self.start:
            raise ValueError("chunk range and text length disagree")
        _validate_parent_span(self.parent_start_cp, self.parent_end_cp, self.start, self.end)
        if any(not __import__("re").fullmatch(r"b[0-9]{5,8}", b) for b in self.block_ids):
            raise ValueError("invalid block id")
        ordinals = [int(b[1:]) for b in self.block_ids]
        if ordinals != sorted(set(ordinals)):
            raise ValueError("block ids must be strictly increasing")
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise ValueError("text hash mismatch")
        object.__setattr__(self, "embedding", check_vector(self.embedding))
        return self


class ChunkRecord(BaseModel):
    model_config = _FROZEN
    extraction_view_id: Uuid4
    ordinal: Annotated[int, Field(ge=0)]
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]
    split_level: Annotated[int, Field(ge=0, le=63)]
    block_ids: Annotated[tuple[str, ...], Field(min_length=1)]
    text: str
    text_sha256: Sha256Hex
    id: Uuid4
    index_version_id: Uuid4
    tenant_id: Uuid4
    project_id: Uuid4
    parent_start_cp: int | None = Field(default=None, ge=0)
    parent_end_cp: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _parent_span(self) -> "ChunkRecord":
        _validate_parent_span(self.parent_start_cp, self.parent_end_cp, self.start, self.end)
        return self


def _validate_parent_span(parent_start: int | None, parent_end: int | None, start: int, end: int) -> None:
    if (parent_start is None) != (parent_end is None):
        raise ValueError("parent span offsets must both be set or unset")
    if parent_start is not None and not 0 <= parent_start <= start < end <= parent_end:
        raise ValueError("parent span must contain chunk span")


class ScoredChunk(BaseModel):
    model_config = _FROZEN
    chunk: ChunkRecord
    distance: float
    rank: Annotated[int, Field(ge=1)]

    @property
    def score(self) -> float:
        return 1.0 - self.distance


def check_vector(values: Iterable[float]) -> tuple[float, ...]:
    try:
        source = tuple(float(x) for x in values)
        if not source or any(not math.isfinite(x) for x in source):
            raise ValueError
        out = tuple(struct.unpack("f", struct.pack("f", x))[0] for x in source)
        norm = struct.unpack("f", struct.pack("f", sum(x * x for x in out)))[0]
        if not math.isfinite(norm) or norm == 0:
            raise ValueError
        return out
    except (TypeError, ValueError, OverflowError, struct.error):
        raise IndexingError("invalid_vector") from None


def embedding_key_for(provider: str, model: str, dim: int) -> str:
    payload = {
        "scheme": "embedding-key-v1",
        "provider": provider,
        "model": model,
        "embedding_dim": dim,
        "distance": "cosine",
        "preprocessing": "none",
    }
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def corpus_sha256_for(pairs: Iterable[tuple[str, str]]) -> str:
    body = "\n".join(f"{view}:{payload}" for view, payload in sorted(pairs, key=lambda pair: pair[0]))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()
