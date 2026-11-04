"""Document and DocumentVersion records. Versions are immutable; there is no update path."""

import uuid
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def new_id() -> str:
    """A new UUID4 string (D9)."""
    return str(uuid.uuid4())


def _check_uuid4(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("must be a UUID4 string") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("must be a canonical lowercase UUID4 string")
    return value


def _check_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("must be timezone-aware UTC")
    return value


Uuid4 = Annotated[str, AfterValidator(_check_uuid4)]
UtcDatetime = Annotated[datetime, AfterValidator(_check_utc)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1, max_length=255)]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Scope(BaseModel):
    """Tenant and project scope. Always taken from authenticated context, never from a request body."""

    model_config = _FROZEN
    tenant_id: Uuid4
    project_id: Uuid4


LOCAL_DEV_SCOPE = Scope(
    tenant_id="7f3c2b1e-5d4a-4c8b-9e6f-1a2b3c4d5e6f",
    project_id="2d9e8f7a-6b5c-4a3d-8e2f-0c1b2a3d4e5f",
)


class MediaType(StrEnum):
    TEXT_PLAIN = "text/plain"
    TEXT_MARKDOWN = "text/markdown"
    TEXT_HTML = "text/html"
    APPLICATION_PDF = "application/pdf"


class NewVersion(BaseModel):
    """Everything known about the bytes of a version before the repository assigns IDs."""

    model_config = _FROZEN
    sha256: Sha256Hex
    size_bytes: Annotated[int, Field(gt=0)]
    media_type: MediaType
    text_encoding: Literal["utf-8", "utf-8-sig"] | None = None
    original_filename: Name
    object_key: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def _encoding_only_none_for_pdf(self) -> "NewVersion":
        if self.text_encoding is None and self.media_type is not MediaType.APPLICATION_PDF:
            raise ValueError("text_encoding may be None only for application/pdf")
        return self


class Document(BaseModel):
    model_config = _FROZEN
    id: Uuid4
    tenant_id: Uuid4
    project_id: Uuid4
    display_name: Name
    created_at: UtcDatetime


class DocumentVersion(NewVersion):
    id: Uuid4
    document_id: Uuid4
    tenant_id: Uuid4
    project_id: Uuid4
    version_number: Annotated[int, Field(ge=1)]
    created_at: UtcDatetime


class IntakeOutcome(BaseModel):
    model_config = _FROZEN
    document: Document
    version: DocumentVersion
    created_document: bool
    created_version: bool

    @model_validator(mode="after")
    def _consistent(self) -> "IntakeOutcome":
        if self.created_document and not self.created_version:
            raise ValueError("a created document always has a created version")
        d, v = self.document, self.version
        if v.document_id != d.id or (v.tenant_id, v.project_id) != (d.tenant_id, d.project_id):
            raise ValueError("version does not belong to document")
        return self

    @property
    def deduplicated(self) -> bool:
        return not self.created_version
