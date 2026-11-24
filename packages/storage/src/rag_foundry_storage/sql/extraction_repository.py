"""PostgreSQL ExtractionRepository. One session and one transaction per call; every query is scoped.

Dedup by identity key is serialized with a transaction-level advisory lock on the version ID. The hashed
lock key can only serialize unrelated creates, never merge them: the key is re-read inside the lock.
"""

from datetime import UTC
from typing import Any

from rag_foundry_domain import NotFoundError, Scope, new_id
from rag_foundry_domain.extraction import ExtractionKey, ExtractionOutcome, ExtractionView, NewExtraction
from sqlalchemy import insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_foundry_storage.sql._tx import read, write
from rag_foundry_storage.sql.models import DocumentVersionRow, ExtractionViewRow
from rag_foundry_storage.sql.repository import _canonical_uuid4

_VIEW_LOCK = text("SELECT pg_advisory_xact_lock(hashtextextended('view:' || CAST(:document_version_id AS text), 0))")


def _view(row: Any) -> ExtractionView:
    return ExtractionView(
        id=str(row.id),
        tenant_id=str(row.tenant_id),
        project_id=str(row.project_id),
        document_id=str(row.document_id),
        document_version_id=str(row.document_version_id),
        extractor=row.extractor,
        extractor_version=row.extractor_version,
        profile_id=row.profile_id,
        unicode_version=row.unicode_version,
        canonical_sha256=row.canonical_sha256,
        char_count=row.char_count,
        block_count=row.block_count,
        page_count=row.page_count,
        payload_sha256=row.payload_sha256,
        payload_key=row.payload_key,
        payload_schema_version=row.payload_schema_version,
        warnings=tuple(row.warnings),
        created_at=row.created_at.astimezone(UTC),
    )


class SqlExtractionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _by_key(scope: Scope, key: ExtractionKey) -> Any:
        return select(ExtractionViewRow).where(
            ExtractionViewRow.tenant_id == scope.tenant_id,
            ExtractionViewRow.project_id == scope.project_id,
            ExtractionViewRow.document_version_id == key.document_version_id,
            ExtractionViewRow.extractor == key.extractor,
            ExtractionViewRow.extractor_version == key.extractor_version,
            ExtractionViewRow.profile_id == key.profile_id,
            ExtractionViewRow.unicode_version == key.unicode_version,
        )

    @staticmethod
    async def _require_version(
        session: AsyncSession, scope: Scope, version_id: str, document_id: str | None = None
    ) -> None:
        query = select(DocumentVersionRow.id).where(
            DocumentVersionRow.id == version_id,
            DocumentVersionRow.tenant_id == scope.tenant_id,
            DocumentVersionRow.project_id == scope.project_id,
        )
        if document_id is not None:
            query = query.where(DocumentVersionRow.document_id == document_id)
        if await session.scalar(query) is None:
            raise NotFoundError()

    async def find_view(self, scope: Scope, key: ExtractionKey) -> ExtractionView | None:
        try:
            _canonical_uuid4(key.document_version_id)
        except NotFoundError:
            return None
        async with read(self._session) as session:
            row = (await session.execute(self._by_key(scope, key))).scalar_one_or_none()
            return None if row is None else _view(row)

    async def create_view_or_get(self, scope: Scope, new: NewExtraction) -> ExtractionOutcome:
        _canonical_uuid4(new.document_version_id)
        _canonical_uuid4(new.document_id)
        async with write(self._session) as session:
            # Check the version together with its document, in scope; never rely on the FK for this.
            # Done before locking so a caller from another scope never holds the lock.
            await self._require_version(session, scope, new.document_version_id, new.document_id)
            await session.execute(_VIEW_LOCK, {"document_version_id": new.document_version_id})
            found = (await session.execute(self._by_key(scope, new.key))).scalar_one_or_none()
            if found is not None:
                return ExtractionOutcome(view=_view(found), created=False)
            row = (
                await session.execute(
                    insert(ExtractionViewRow)
                    .values(
                        **new.model_dump(mode="json"),
                        id=new_id(),
                        tenant_id=scope.tenant_id,
                        project_id=scope.project_id,
                    )
                    .returning(ExtractionViewRow)
                )
            ).scalar_one()
            return ExtractionOutcome(view=_view(row), created=True)

    async def get_view(self, scope: Scope, view_id: str) -> ExtractionView:
        _canonical_uuid4(view_id)
        async with read(self._session) as session:
            row = (
                await session.execute(
                    select(ExtractionViewRow).where(
                        ExtractionViewRow.id == view_id,
                        ExtractionViewRow.tenant_id == scope.tenant_id,
                        ExtractionViewRow.project_id == scope.project_id,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                raise NotFoundError()
            return _view(row)

    async def list_views(self, scope: Scope, document_version_id: str) -> list[ExtractionView]:
        _canonical_uuid4(document_version_id)
        async with read(self._session) as session:
            await self._require_version(session, scope, document_version_id)
            rows = (
                await session.execute(
                    select(ExtractionViewRow)
                    .where(
                        ExtractionViewRow.tenant_id == scope.tenant_id,
                        ExtractionViewRow.project_id == scope.project_id,
                        ExtractionViewRow.document_version_id == document_version_id,
                    )
                    .order_by(ExtractionViewRow.created_at, ExtractionViewRow.id)
                )
            ).scalars()
            return [_view(r) for r in rows]
