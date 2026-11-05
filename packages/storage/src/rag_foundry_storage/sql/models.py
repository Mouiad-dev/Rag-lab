"""Tables for document intake (ADR-0009). Version immutability is enforced by triggers in migration 0001."""

from datetime import datetime

from rag_foundry_domain import MediaType
from sqlalchemy import (
    ARRAY,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING = {
    "pk": "pk_%(table_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
}

_MEDIA_TYPES = ", ".join(f"'{m.value}'" for m in MediaType)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), nullable=False, server_default=func.clock_timestamp())


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    created_at: Mapped[datetime] = _created_at()


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("tenant_id", "id"),)
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("tenants.id"), nullable=False)
    created_at: Mapped[datetime] = _created_at()


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "project_id", "id"),
        ForeignKeyConstraint(["tenant_id", "project_id"], ["projects.tenant_id", "projects.id"]),
        CheckConstraint("char_length(display_name) BETWEEN 1 AND 255", name="display_name_length"),
    )
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    project_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = _created_at()


class DocumentVersionRow(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version_number"),
        UniqueConstraint("document_id", "sha256"),
        UniqueConstraint("tenant_id", "project_id", "document_id", "id"),
        ForeignKeyConstraint(
            ["tenant_id", "project_id", "document_id"],
            ["documents.tenant_id", "documents.project_id", "documents.id"],
            name="fk_document_versions_scope_document",
        ),
        Index("ix_document_versions_project_sha256", "tenant_id", "project_id", "sha256"),
        CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="sha256_format"),
        CheckConstraint("size_bytes > 0", name="size_positive"),
        CheckConstraint("version_number >= 1", name="version_number_positive"),
        CheckConstraint(f"media_type IN ({_MEDIA_TYPES})", name="media_type_known"),
        CheckConstraint("text_encoding IN ('utf-8', 'utf-8-sig')", name="text_encoding_known"),
        CheckConstraint(
            "text_encoding IS NOT NULL OR media_type = 'application/pdf'", name="text_encoding_required_for_text"
        ),
        CheckConstraint("char_length(original_filename) BETWEEN 1 AND 255", name="filename_length"),
    )
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    project_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    document_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    media_type: Mapped[str] = mapped_column(String(64), nullable=False)
    text_encoding: Mapped[str | None] = mapped_column(String(16), nullable=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class ExtractionViewRow(Base):
    """Immutable extraction view metadata (ADR-0010). The payload lives in the object store (D5)."""

    __tablename__ = "extraction_views"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "project_id", "document_id", "document_version_id"],
            [
                "document_versions.tenant_id",
                "document_versions.project_id",
                "document_versions.document_id",
                "document_versions.id",
            ],
            name="fk_extraction_views_scope_version",
        ),
        UniqueConstraint(
            "document_version_id",
            "extractor",
            "extractor_version",
            "profile_id",
            "unicode_version",
            name="uq_extraction_views_identity_key",
        ),
        Index("ix_extraction_views_version_created", "tenant_id", "project_id", "document_version_id", "created_at"),
        CheckConstraint("canonical_sha256 ~ '^[0-9a-f]{64}$'", name="canonical_sha256_format"),
        CheckConstraint("payload_sha256 ~ '^[0-9a-f]{64}$'", name="payload_sha256_format"),
        CheckConstraint("char_count >= 1", name="char_count_positive"),
        CheckConstraint("block_count >= 1", name="block_count_positive"),
        CheckConstraint("page_count IS NULL OR page_count >= 1", name="page_count_positive"),
        CheckConstraint("payload_schema_version = 1", name="payload_schema_version_known"),
        CheckConstraint("extractor ~ '^[a-z0-9._-]{1,64}$'", name="extractor_format"),
    )
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    project_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    document_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    document_version_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    extractor: Mapped[str] = mapped_column(String(64), nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(128), nullable=False)
    profile_id: Mapped[str] = mapped_column(String(128), nullable=False)
    unicode_version: Mapped[str] = mapped_column(String(32), nullable=False)
    canonical_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    block_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_key: Mapped[str] = mapped_column(Text, nullable=False)
    payload_schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    warnings: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    created_at: Mapped[datetime] = _created_at()
