"""PostgreSQL DocumentRepository. One session and one transaction per call; every query is scoped.

Dedup and numbering are serialized with transaction-level advisory locks. The lock key is hashed, so a
hash collision can only serialize two unrelated uploads, never merge them: every decision re-reads the
real rows inside the lock.
"""

import uuid
from datetime import UTC
from typing import Any

from rag_foundry_domain import (
    Document,
    DocumentVersion,
    IntakeOutcome,
    NewVersion,
    NotFoundError,
    Scope,
    new_id,
)
from sqlalchemy import func, insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_foundry_storage.sql._tx import read, write
from rag_foundry_storage.sql.models import DocumentRow, DocumentVersionRow, Project

_SHA_LOCK = text(
    "SELECT pg_advisory_xact_lock("
    "hashtextextended('sha:' || CAST(:project_id AS text) || ':' || CAST(:sha256 AS text), 0))"
)
_DOC_LOCK = text("SELECT pg_advisory_xact_lock(hashtextextended('doc:' || CAST(:document_id AS text), 0))")


def _canonical_uuid4(value: str) -> str:
    """Malformed IDs look like unknown IDs; PostgreSQL would raise a DataError on a non-UUID."""
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise NotFoundError() from None
    if parsed.version != 4 or str(parsed) != value:
        raise NotFoundError()
    return value


def _document(row: Any) -> Document:
    return Document(
        id=str(row.id),
        tenant_id=str(row.tenant_id),
        project_id=str(row.project_id),
        display_name=row.display_name,
        created_at=row.created_at.astimezone(UTC),
    )


def _version(row: Any) -> DocumentVersion:
    return DocumentVersion(
        id=str(row.id),
        document_id=str(row.document_id),
        tenant_id=str(row.tenant_id),
        project_id=str(row.project_id),
        version_number=row.version_number,
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        media_type=row.media_type,
        text_encoding=row.text_encoding,
        original_filename=row.original_filename,
        object_key=row.object_key,
        created_at=row.created_at.astimezone(UTC),
    )


class SqlDocumentRepository:
    # Note: create (project sha lock) and add-version (document lock) use different locks, so they can
    # concurrently put the same bytes into two documents of one project. The domain allows this; later
    # dedup returns the earliest-created version.
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    async def _project_exists(session: AsyncSession, scope: Scope) -> bool:
        found = await session.scalar(
            select(Project.id).where(Project.id == scope.project_id, Project.tenant_id == scope.tenant_id)
        )
        return found is not None

    @staticmethod
    async def _document_row(session: AsyncSession, scope: Scope, document_id: str) -> Any:
        _canonical_uuid4(document_id)
        row = (
            await session.execute(
                select(DocumentRow).where(
                    DocumentRow.id == document_id,
                    DocumentRow.tenant_id == scope.tenant_id,
                    DocumentRow.project_id == scope.project_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise NotFoundError()
        return row

    @staticmethod
    def _new_values(doc_id: str, scope: Scope, new: NewVersion, number: int) -> dict[str, Any]:
        return {
            **new.model_dump(mode="json"),
            "id": new_id(),
            "document_id": doc_id,
            "tenant_id": scope.tenant_id,
            "project_id": scope.project_id,
            "version_number": number,
        }

    async def project_exists(self, scope: Scope) -> bool:
        async with read(self._session) as session:
            return await self._project_exists(session, scope)

    async def get_document(self, scope: Scope, document_id: str) -> Document:
        async with read(self._session) as session:
            return _document(await self._document_row(session, scope, document_id))

    async def create_document_or_get(self, scope: Scope, new: NewVersion, *, display_name: str) -> IntakeOutcome:
        async with write(self._session) as session:
            if not await self._project_exists(session, scope):
                raise NotFoundError()
            await session.execute(_SHA_LOCK, {"project_id": scope.project_id, "sha256": new.sha256})
            match = (
                await session.execute(
                    select(DocumentVersionRow)
                    .where(
                        DocumentVersionRow.tenant_id == scope.tenant_id,
                        DocumentVersionRow.project_id == scope.project_id,
                        DocumentVersionRow.sha256 == new.sha256,
                    )
                    .order_by(DocumentVersionRow.created_at, DocumentVersionRow.id)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if match is not None:
                doc = await self._document_row(session, scope, str(match.document_id))
                return IntakeOutcome(
                    document=_document(doc), version=_version(match), created_document=False, created_version=False
                )
            doc = (
                await session.execute(
                    insert(DocumentRow)
                    .values(
                        id=new_id(), tenant_id=scope.tenant_id, project_id=scope.project_id, display_name=display_name
                    )
                    .returning(DocumentRow)
                )
            ).scalar_one()
            version = (
                await session.execute(
                    insert(DocumentVersionRow)
                    .values(**self._new_values(str(doc.id), scope, new, 1))
                    .returning(DocumentVersionRow)
                )
            ).scalar_one()
            return IntakeOutcome(
                document=_document(doc), version=_version(version), created_document=True, created_version=True
            )

    async def add_version_or_get(self, scope: Scope, document_id: str, new: NewVersion) -> IntakeOutcome:
        async with write(self._session) as session:
            doc = await self._document_row(session, scope, document_id)
            await session.execute(_DOC_LOCK, {"document_id": document_id})
            found = (
                await session.execute(
                    select(DocumentVersionRow).where(
                        DocumentVersionRow.document_id == document_id,
                        DocumentVersionRow.tenant_id == scope.tenant_id,
                        DocumentVersionRow.project_id == scope.project_id,
                        DocumentVersionRow.sha256 == new.sha256,
                    )
                )
            ).scalar_one_or_none()
            if found is not None:
                return IntakeOutcome(
                    document=_document(doc), version=_version(found), created_document=False, created_version=False
                )
            top = await session.scalar(
                select(func.max(DocumentVersionRow.version_number)).where(DocumentVersionRow.document_id == document_id)
            )
            version = (
                await session.execute(
                    insert(DocumentVersionRow)
                    .values(**self._new_values(document_id, scope, new, (top or 0) + 1))
                    .returning(DocumentVersionRow)
                )
            ).scalar_one()
            return IntakeOutcome(
                document=_document(doc), version=_version(version), created_document=False, created_version=True
            )

    async def get_version(self, scope: Scope, document_id: str, version_id: str) -> DocumentVersion:
        _canonical_uuid4(document_id)
        _canonical_uuid4(version_id)
        async with read(self._session) as session:
            await self._document_row(session, scope, document_id)
            row = (
                await session.execute(
                    select(DocumentVersionRow).where(
                        DocumentVersionRow.id == version_id,
                        DocumentVersionRow.document_id == document_id,
                        DocumentVersionRow.tenant_id == scope.tenant_id,
                        DocumentVersionRow.project_id == scope.project_id,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                raise NotFoundError()
            return _version(row)

    async def list_versions(self, scope: Scope, document_id: str) -> list[DocumentVersion]:
        async with read(self._session) as session:
            await self._document_row(session, scope, document_id)
            rows = (
                await session.execute(
                    select(DocumentVersionRow)
                    .where(
                        DocumentVersionRow.document_id == document_id,
                        DocumentVersionRow.tenant_id == scope.tenant_id,
                        DocumentVersionRow.project_id == scope.project_id,
                    )
                    .order_by(DocumentVersionRow.version_number)
                )
            ).scalars()
            return [_version(r) for r in rows]
