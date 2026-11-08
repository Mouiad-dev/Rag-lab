"""Worker-side index construction and anchor inspection."""

import hashlib
import json
from typing import Any

from rag_foundry_contracts.source import SourceAnchor
from rag_foundry_domain import ObjectStore, Scope
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.indexing import (
    ChunkRecord,
    IndexVersion,
    NewChunk,
    NewIndexVersion,
    corpus_sha256_for,
    embedding_key_for,
)
from rag_foundry_domain.runtime_settings import IndexingSettings
from rag_foundry_domain.uow import UnitOfWorkFactory
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.extraction import read_payload
from rag_foundry_techniques import ChunkInput, ParentLinkedChunker, TextBlock


class ChunkInspection(ChunkRecord):
    anchors: tuple[SourceAnchor, ...]
    parent_anchors: tuple[SourceAnchor, ...] = ()


class IndexService:
    def __init__(
        self,
        store: ObjectStore,
        uow: UnitOfWorkFactory,
        anchors: AnchorService,
        *,
        batch_size: int = IndexingSettings().embed_batch_size,
        max_views: int = 100,  # fixed: ADR-0014 D3
    ) -> None:
        self._store, self._uow, self._anchors = store, uow, anchors
        self._batch_size, self._max_views = batch_size, max_views

    async def build(self, scope: Scope, view_ids: tuple[str, ...], chunker, embedder) -> IndexVersion:
        if len(view_ids) > self._max_views:
            raise IndexingError("too_many_views")
        if not view_ids or len(set(view_ids)) != len(view_ids):
            raise ValueError("view_ids must be non-empty and unique")
        profile = embedder.profile
        dim = profile.capabilities.embedding_dim
        if dim is None:
            raise IndexingError("missing_embedding_dim")
        async with self._uow(read_only=True) as uow:
            views = []
            for view_id in sorted(view_ids):
                views.append(await uow.extractions.get_view(scope, view_id))
        loaded = []
        for view in views:
            payload = await read_payload(self._store, scope, view)
            loaded.append((view, payload))

        config_json = json.dumps(
            chunker.config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        key = embedding_key_for(profile.provider, profile.model, dim)
        linked = isinstance(chunker, ParentLinkedChunker)
        prepared: list[tuple[Any, Any, tuple[str, ...], str, int | None, int | None]] = []
        unanchored = 0
        for view, payload in loaded:
            blocks = tuple(
                TextBlock(
                    block_id=b.block_id,
                    start=b.start,
                    end=b.end,
                    kind=b.kind.value,
                    section_path=b.section_path,
                    heading_level=b.heading_level,
                    page_index=b.page_index,
                )
                for b in payload.blocks
            )
            doc = ChunkInput(text=payload.canonical_text, blocks=blocks)
            spans = chunker.chunk_with_parents(doc) if linked else chunker.chunk(doc)
            for item in spans:
                span = item
                parent_start = item.parent_start if linked else None
                parent_end = item.parent_end if linked else None
                if parent_end is not None and parent_end > len(payload.canonical_text):
                    raise IndexingError("parent_span_out_of_bounds")
                overlapped = tuple(b.block_id for b in payload.blocks if b.start < span.end and b.end > span.start)
                if not overlapped:
                    unanchored += 1
                    continue
                prepared.append(
                    (view, span, overlapped, payload.canonical_text[span.start : span.end], parent_start, parent_end)
                )

        new = NewIndexVersion(
            chunker_id=chunker.manifest.technique_id,
            chunker_version=chunker.manifest.version,
            chunker_config_json=config_json,
            chunker_config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
            embedding_profile_id=profile.id,
            embedding_provider=profile.provider,
            embedding_model=profile.model,
            embedding_dim=dim,
            distance="cosine",
            embedding_key=key,
            extraction_view_ids=tuple(sorted(view.id for view, _ in loaded)),
            corpus_sha256=corpus_sha256_for((v.id, v.payload_sha256) for v, _ in loaded),
            unanchored_chunk_count=unanchored,
        )
        async with self._uow() as uow:
            version = await uow.indexes.create_index_version(scope, new)
            await uow.commit()
        try:
            ordinal = 0
            for start in range(0, len(prepared), self._batch_size):
                batch = prepared[start : start + self._batch_size]
                result = await embedder.embed([item[3] for item in batch], input_type="document")
                if len(result.vectors) != len(batch):
                    raise IndexingError("embedding_count_mismatch")
                records = []
                for (view, span, block_ids, text, parent_start, parent_end), vector in zip(
                    batch, result.vectors, strict=True
                ):
                    records.append(
                        NewChunk(
                            extraction_view_id=view.id,
                            ordinal=ordinal,
                            start=span.start,
                            end=span.end,
                            split_level=span.split_level,
                            block_ids=block_ids,
                            text=text,
                            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            embedding=vector,
                            parent_start_cp=parent_start,
                            parent_end_cp=parent_end,
                        )
                    )
                    ordinal += 1
                async with self._uow() as uow:
                    await uow.indexes.add_chunks(scope, version.id, tuple(records))
                    await uow.commit()
            async with self._uow() as uow:
                await uow.indexes.mark_ready(scope, version.id)
                await uow.commit()
        except BaseException:
            # ADR-0015 D6: mark_failed must run after the failing UoW exits;
            # the nested-write guard in tests/domain/test_uow.py rejects a
            # nested UoW instead of waiting on its own FOR UPDATE lock.
            async with self._uow() as uow:
                await uow.indexes.mark_failed(scope, version.id)
                await uow.commit()
            raise
        async with self._uow(read_only=True) as uow:
            return await uow.indexes.get_index_version(scope, version.id)

    async def inspect_chunks(
        self, scope: Scope, index_version_id: str, *, offset: int = 0, limit: int = 100
    ) -> tuple[ChunkInspection, ...]:
        """Inspect bounded chunks; payloads are re-read once per chunk until per-call caching is added."""
        if offset < 0 or limit < 1 or limit > 100:
            raise IndexingError("invalid_page")
        async with self._uow(read_only=True) as uow:
            rows = await uow.indexes.list_chunks(scope, index_version_id, offset=offset, limit=limit)
        inspections = []
        for row in rows:
            anchors = await self._anchors.anchors_for_span(scope, row.extraction_view_id, row.start, row.end)
            parent_anchors = ()
            if row.parent_start_cp is not None:
                parent_anchors = await self._anchors.anchors_for_span(
                    scope, row.extraction_view_id, row.parent_start_cp, row.parent_end_cp
                )
            inspections.append(ChunkInspection(**row.model_dump(), anchors=anchors, parent_anchors=parent_anchors))
        return tuple(inspections)
