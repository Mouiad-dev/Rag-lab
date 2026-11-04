"""IntakeService: limits and type rules while streaming, then store and dedup/version (ADR-0009)."""

import asyncio
from collections.abc import AsyncIterable, AsyncIterator

from rag_foundry_domain import (
    DocumentRepository,
    IntakeOutcome,
    NewVersion,
    NotFoundError,
    ObjectStore,
    Scope,
    StagedObject,
)

from rag_foundry_ingestion.media import ContentInspector, parse_media_type, validate_filename


class IntakeService:
    def __init__(self, store: ObjectStore, repo: DocumentRepository, *, max_bytes: int) -> None:
        self._store = store
        self._repo = repo
        self._max_bytes = max_bytes

    async def ingest(
        self,
        scope: Scope,
        chunks: AsyncIterable[bytes],
        *,
        media_type: str | None,
        filename: str,
        document_id: str | None = None,
    ) -> IntakeOutcome:
        if not await self._repo.project_exists(scope):
            raise NotFoundError()
        if document_id is not None:
            await self._repo.get_document(scope, document_id)
        declared = parse_media_type(media_type)
        name = validate_filename(filename)
        inspector = ContentInspector(declared, name)

        async def inspected() -> AsyncIterator[bytes]:
            async for chunk in chunks:
                inspector.feed(chunk)
                yield chunk

        staged = await self._store.stage(inspected(), max_bytes=self._max_bytes)
        committed = False
        try:
            encoding = inspector.finish()
            key = self._store.key_for(scope, staged.sha256)
            await self._store.commit(staged, key)
            committed = True
            new = NewVersion(
                sha256=staged.sha256,
                size_bytes=staged.size_bytes,
                media_type=declared,
                text_encoding=encoding,
                original_filename=name,
                object_key=key,
            )
            if document_id is None:
                return await self._repo.create_document_or_get(scope, new, display_name=name)
            return await self._repo.add_version_or_get(scope, document_id, new)
        except BaseException:
            if not committed:
                await self._discard_quietly(staged)
            raise

    async def _discard_quietly(self, staged: StagedObject) -> None:
        try:
            await asyncio.shield(self._store.discard(staged))
        except Exception:  # keep the original error
            pass
