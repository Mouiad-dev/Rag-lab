"""Source anchor resolve request/response contract models, version 1.0.

Framework-neutral: no server, ORM or HTTP imports. ``SourceAnchor`` is reused unchanged.
"""

from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, field_validator, model_validator

from rag_foundry_contracts.source import (
    CONTRACT_VERSION,
    SourceAnchor,
    _Model,
    check_contract_version,
)

MAX_RESOLVE_ANCHORS = 100
MAX_ANCHOR_ID_LENGTH = 256

AnchorIdStr = Annotated[str, Field(min_length=1, max_length=MAX_ANCHOR_ID_LENGTH)]


class ResolutionStatus(StrEnum):
    EXACT = "exact"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"


class ResolutionReason(StrEnum):
    FOREIGN_NAMESPACE = "foreign_namespace"
    NOT_FOUND = "not_found"
    SOURCE_MISMATCH = "source_mismatch"
    UNSUPPORTED_PRECISION = "unsupported_precision"
    BLOCK_NOT_FOUND = "block_not_found"
    RANGE_OUT_OF_BOUNDS = "range_out_of_bounds"
    QUOTE_MISMATCH = "quote_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    SUFFIX_MISMATCH = "suffix_mismatch"
    PAGE_MISMATCH = "page_mismatch"
    INTEGRITY_ERROR = "integrity_error"


class AnchorResolution(_Model):
    """Result for one anchor. Text and position are set only when the status is ``exact``."""

    contract_version: str = CONTRACT_VERSION
    anchor_id: AnchorIdStr
    status: ResolutionStatus
    reason: ResolutionReason | None = None
    exact: str | None = None
    block_id: str | None = None
    section_path: tuple[str, ...] = ()
    page_index: int | None = Field(default=None, ge=0)

    @field_validator("contract_version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        return check_contract_version(v)

    @model_validator(mode="after")
    def _status_consistent(self) -> Self:
        is_exact = self.status is ResolutionStatus.EXACT
        if is_exact != (self.reason is None):
            raise ValueError("reason must be None if and only if status is exact")
        if is_exact:
            if self.exact is None or self.block_id is None:
                raise ValueError("status exact requires exact and block_id")
        else:
            if self.exact is not None or self.block_id is not None:
                raise ValueError("exact and block_id are allowed only when status is exact")
            if self.section_path != () or self.page_index is not None:
                raise ValueError("section_path and page_index are allowed only when status is exact")
        return self


class AnchorResolveRequest(_Model):
    contract_version: str = CONTRACT_VERSION
    anchors: tuple[SourceAnchor, ...] = Field(min_length=1, max_length=MAX_RESOLVE_ANCHORS)

    @field_validator("contract_version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        return check_contract_version(v)

    @model_validator(mode="after")
    def _anchor_ids_fit_resolution(self) -> Self:
        # SourceAnchor does not cap anchor_id; reject here so AnchorResolution can always be built.
        if any(len(a.anchor_id) > MAX_ANCHOR_ID_LENGTH for a in self.anchors):
            raise ValueError(f"anchor_id must be at most {MAX_ANCHOR_ID_LENGTH} characters")
        return self


class AnchorResolveResponse(_Model):
    """Results in the same order as the request anchors."""

    contract_version: str = CONTRACT_VERSION
    results: tuple[AnchorResolution, ...]

    @field_validator("contract_version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        return check_contract_version(v)
