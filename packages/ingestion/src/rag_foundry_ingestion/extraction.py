"""ExtractionService: bytes of a stored version -> immutable extraction view (ADR-0010 D4, D5, D10, D11).

No HTTP route and no API-process use (D4). Extracted text is never logged or put in errors.
"""

import asyncio
import hashlib
import unicodedata
from collections.abc import AsyncIterator, Mapping

from rag_foundry_domain import DocumentRepository, MediaType, NotFoundError, ObjectStore, Scope, StagedObject
from rag_foundry_domain.errors import ExtractionError
from rag_foundry_domain.extraction import ExtractionKey, ExtractionOutcome, ExtractionView, NewExtraction
from rag_foundry_domain.ports import ExtractionRepository

from rag_foundry_ingestion.extract.base import ExtractionLimits, Extractor
from rag_foundry_ingestion.normalize import CANONICAL_V1
from rag_foundry_ingestion.payload import ExtractionPayload


def _mismatch(message: str) -> ExtractionError:
    return ExtractionError("integrity_mismatch", message)


async def _aclose(stream: AsyncIterator[bytes]) -> None:
    close = getattr(stream, "aclose", None)
    if close is not None:
        await close()


class ExtractionService:
    def __init__(
        self,
        store: ObjectStore,
        documents: DocumentRepository,
        views: ExtractionRepository,
        *,
        extractors: Mapping[MediaType, Extractor],
        limits: ExtractionLimits,
    ) -> None:
        self._store = store
        self._documents = documents
        self._views = views
        self._extractors = extractors
        self._limits = limits

    async def extract(self, scope: Scope, document_id: str, version_id: str) -> ExtractionOutcome:
        version = await self._documents.get_version(scope, document_id, version_id)
        try:
            extractor = self._extractors.get(MediaType(version.media_type))
        except ValueError:
            extractor = None
        if extractor is None:
            raise ExtractionError("unsupported_extractor", "no extractor is available for this media type")
        key = ExtractionKey(
            document_version_id=version.id,
            extractor=extractor.name,
            extractor_version=extractor.version,
            profile_id=CANONICAL_V1.profile_id,
            unicode_version=unicodedata.unidata_version,
        )
        existing = await self._views.find_view(scope, key)
        if existing is not None:
            return ExtractionOutcome(view=existing, created=False)

        data = await self._read_verified(version.object_key, version.sha256, version.size_bytes)
        result = await asyncio.to_thread(
            extractor.extract, data, text_encoding=version.text_encoding, limits=self._limits
        )
        if (result.extractor, result.extractor_version, result.profile_id, result.unicode_version) != (
            key.extractor,
            key.extractor_version,
            key.profile_id,
            key.unicode_version,
        ):
            raise _mismatch("extractor result does not match its identity key")
        payload = ExtractionPayload(
            extractor=result.extractor,
            extractor_version=result.extractor_version,
            profile_id=result.profile_id,
            unicode_version=result.unicode_version,
            source_space=result.source_space,
            page_starts=result.page_starts,
            canonical_text=result.canonical_text,
            blocks=result.blocks,
            source_map=result.source_map,
        )
        body = payload.dumps()
        ExtractionPayload.loads(body)  # never store a payload that cannot be read back
        payload_sha = hashlib.sha256(body).hexdigest()
        payload_key = self._store.key_for(scope, payload_sha)
        await self._store_payload(body, payload_key)
        new = NewExtraction(
            **key.model_dump(),
            document_id=document_id,
            canonical_sha256=hashlib.sha256(result.canonical_text.encode("utf-8")).hexdigest(),
            char_count=len(result.canonical_text),
            block_count=len(result.blocks),
            page_count=len(result.page_starts) if result.page_starts else None,
            payload_sha256=payload_sha,
            payload_key=payload_key,
            warnings=result.warnings,
        )
        return await self._views.create_view_or_get(scope, new)

    async def load_payload(self, scope: Scope, view: ExtractionView) -> ExtractionPayload:
        if (view.tenant_id, view.project_id) != (scope.tenant_id, scope.project_id):
            raise NotFoundError()
        if view.payload_key != self._store.key_for(scope, view.payload_sha256):
            raise _mismatch("payload key does not match its hash")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        stream = self._store.open_read(view.payload_key)
        try:
            async for chunk in stream:
                digest.update(chunk)
                chunks.append(chunk)
        finally:
            await _aclose(stream)
        if digest.hexdigest() != view.payload_sha256:
            raise _mismatch("payload hash does not match")
        return ExtractionPayload.loads(b"".join(chunks))

    async def _read_verified(self, object_key: str, sha256: str, size_bytes: int) -> bytes:
        digest = hashlib.sha256()
        parts: list[bytes] = []
        total = 0
        stream = self._store.open_read(object_key)
        try:
            async for chunk in stream:
                total += len(chunk)
                if total > size_bytes:  # stop at the first excess chunk instead of reading on
                    raise _mismatch("stored object size does not match")
                digest.update(chunk)
                parts.append(chunk)
        finally:
            await _aclose(stream)
        if total != size_bytes or digest.hexdigest() != sha256:
            raise _mismatch("stored object does not match its recorded hash")
        return b"".join(parts)

    async def _store_payload(self, body: bytes, payload_key: str) -> None:
        async def one() -> AsyncIterator[bytes]:
            yield body

        staged = await self._store.stage(one(), max_bytes=len(body))
        committed = False
        try:
            await self._store.commit(staged, payload_key)
            committed = True
        except BaseException:
            if not committed:
                await self._discard_quietly(staged)
            raise

    async def _discard_quietly(self, staged: StagedObject) -> None:
        try:
            await asyncio.shield(self._store.discard(staged))
        except Exception:  # keep the original error
            pass
