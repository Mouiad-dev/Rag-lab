"""Synthetic chunk representations (R23 summaries and questions, R24 context prefixes).

Representations are generated text. They are never evidence; search returns the original chunk.
"""

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_foundry_domain.documents import Sha256Hex, Uuid4
from rag_foundry_domain.indexing import ChunkRecord, check_vector

_FROZEN = ConfigDict(frozen=True, extra="forbid")

RepresentationKind = Literal["summary", "question", "context_prefix"]


class RepresentationSet(BaseModel):
    model_config = _FROZEN
    representation_key: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    technique_id: Annotated[str, Field(pattern=r"^R[0-9]{2}$")]
    generator_json: Annotated[str, Field(max_length=8000)]


class _RepresentationFields(BaseModel):
    model_config = _FROZEN
    chunk_id: Uuid4
    kind: RepresentationKind
    ordinal: Annotated[int, Field(ge=0, le=15)]
    text: Annotated[str, Field(min_length=1, max_length=4000)]
    text_sha256: Sha256Hex
    embedded_input_sha256: Sha256Hex

    @model_validator(mode="after")
    def _validate_text(self) -> "_RepresentationFields":
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise ValueError("text hash mismatch")
        if self.kind in ("summary", "context_prefix") and self.ordinal != 0:
            raise ValueError("summary and context_prefix require ordinal 0")
        return self


class NewRepresentation(_RepresentationFields):
    embedding: tuple[float, ...]

    @model_validator(mode="after")
    def _validate_embedding(self) -> "NewRepresentation":
        object.__setattr__(self, "embedding", check_vector(self.embedding))
        return self


class RepresentationRecord(_RepresentationFields):
    index_version_id: Uuid4
    tenant_id: Uuid4
    project_id: Uuid4
    representation_key: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    synthetic: Literal[True] = True


class ScoredRepresentation(BaseModel):
    model_config = _FROZEN
    chunk: ChunkRecord
    representation: RepresentationRecord
    distance: float
    rank: Annotated[int, Field(ge=1)]


def representation_key_for(payload: dict) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
