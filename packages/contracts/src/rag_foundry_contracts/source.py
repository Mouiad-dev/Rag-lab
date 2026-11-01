"""Source reference and source anchor contract models, version 1.0.

All models are frozen and reject unknown fields (``extra="forbid"``) in v1. Consumer adapters may
relax this later under REUSE_AND_INTEGRATION section 6 (tolerate documented unknown optional fields).
The JSON shape is informed by, but not claimed to conform to, the W3C Web Annotation Data Model.
"""

import re
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

CONTRACT_VERSION = "1.0"

_VERSION_RE = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class UnsupportedContractVersionError(ValueError):
    """Raised for a malformed contract version or an unsupported major version."""


def check_contract_version(v: str) -> str:
    """Accept any ``1.x`` version; raise for other majors or malformed strings."""
    if not isinstance(v, str) or not _VERSION_RE.fullmatch(v):
        raise UnsupportedContractVersionError(f"malformed contract version: {v!r}")
    major = int(v.split(".", 1)[0])
    if major != 1:
        raise UnsupportedContractVersionError(f"unsupported contract major version {major}; supported: 1.x")
    return v


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceRef(_Model):
    """Provider-qualified pointer to an immutable document version and extraction view."""

    provider_namespace: NonEmptyStr
    document_id: NonEmptyStr
    document_version_id: NonEmptyStr
    extraction_view_id: NonEmptyStr


class TextRange(_Model):
    """Zero-based, half-open range of Unicode code points in canonical block text.

    Not UTF-8 bytes and not UTF-16 code units.
    """

    start: int = Field(ge=0)
    end: int

    @model_validator(mode="after")
    def _end_not_before_start(self) -> Self:
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class PageRegion(_Model):
    """Normalized [0,1] rectangle with top-left origin in the stored upright page view."""

    page_index: int = Field(ge=0)
    x0: UnitFloat
    y0: UnitFloat
    x1: UnitFloat
    y1: UnitFloat

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.x0 > self.x1 or self.y0 > self.y1:
            raise ValueError("require x0 <= x1 and y0 <= y1")
        return self


class Precision(StrEnum):
    EXACT_TEXT = "exact_text"
    EXACT_REGION = "exact_region"
    APPROXIMATE_REGION = "approximate_region"
    PAGE_ONLY = "page_only"
    UNAVAILABLE = "unavailable"


class SourceAnchor(_Model):
    """Stable reference to an exact span or region within a pinned extraction view.

    Owned by the document/extraction version; carries no chunk or index ID. Unknown fields are
    rejected in v1; consumer adapters may relax this later under REUSE_AND_INTEGRATION section 6.
    """

    contract_version: str = CONTRACT_VERSION
    anchor_id: NonEmptyStr
    source: SourceRef
    block_id: NonEmptyStr
    section_path: tuple[str, ...] = ()
    text_range: TextRange | None = None
    exact: str | None = None
    prefix: str | None = None
    suffix: str | None = None
    page_index: int | None = Field(default=None, ge=0)
    page_label: str | None = None
    regions: tuple[PageRegion, ...] = ()
    precision: Precision
    warnings: tuple[str, ...] = ()

    @field_validator("contract_version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        return check_contract_version(v)

    @model_validator(mode="after")
    def _exact_matches_range(self) -> Self:
        if self.exact is not None:
            if self.text_range is None:
                raise ValueError("exact requires text_range")
            if len(self.exact) != self.text_range.end - self.text_range.start:
                raise ValueError("len(exact) must equal text_range.end - text_range.start")
        return self

    @model_validator(mode="after")
    def _precision_backed_by_data(self) -> Self:
        # An anchor must not claim more precision than the data it carries.
        if self.precision is Precision.EXACT_TEXT and self.text_range is None:
            raise ValueError("precision exact_text requires text_range")
        if self.precision in (Precision.EXACT_REGION, Precision.APPROXIMATE_REGION) and not self.regions:
            raise ValueError(f"precision {self.precision.value} requires regions")
        if self.precision is Precision.PAGE_ONLY and self.page_index is None:
            raise ValueError("precision page_only requires page_index")
        if self.regions and self.page_index is None:
            raise ValueError("regions require page_index")
        if self.page_index is not None and any(r.page_index != self.page_index for r in self.regions):
            raise ValueError("every region page_index must equal the anchor page_index")
        return self
