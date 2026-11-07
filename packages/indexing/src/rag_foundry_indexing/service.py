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
from rag_foundry_domain.ports import ExtractionRepository, IndexRepository
from rag_foundry_domain.runtime_settings import IndexingSettings
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.extraction import read_payload
from rag_foundry_techniques import ChunkInput, TextBlock


class ChunkInspection(ChunkRecord):
    anchors: tuple[SourceAnchor, ...]


class IndexService:
    def __init__(
        self,
        store: ObjectStore,
        views: ExtractionRepository,
        index: IndexRepository,
        anchors: AnchorService,
        *,
        batch_size: int = IndexingSettings().embed_batch_size,
        max_views: int = 100,  # fixed: ADR-0014 D3
    ) -> None:
        self._store, self._views, self._index, self._anchors = store, views, index, anchors
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
        loaded = []
        for view_id in sorted(view_ids):
            view = await self._views.get_view(scope, view_id)
            payload = await read_payload(self._store, scope, view)
            loaded.append((view, payload))

        config_json = json.dumps(
            chunker.config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        key = embedding_key_for(profile.provider, profile.model, dim)
        prepared: list[tuple[Any, Any, tuple[str, ...], str]] = []
        unanchored = 0
        for view, payload in loaded:
            blocks = tuple(TextBlock(block_id=b.block_id, start=b.start, end=b.end) for b in payload.blocks)
            spans = chunker.chunk(ChunkInput(text=payload.canonical_text, blocks=blocks))
            for span in spans:
                overlapped = tuple(b.block_id for b in payload.blocks if b.start < span.end and b.end > span.start)
                if not overlapped:
                    unanchored += 1
                    continue
                prepared.append((view, span, overlapped, payload.canonical_text[span.start : span.end]))

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
        version = await self._index.create_index_version(scope, new)
        try:
            ordinal = 0
            for start in range(0, len(prepared), self._batch_size):
                batch = prepared[start : start + self._batch_size]
                result = await embedder.embed([item[3] for item in batch], input_type="document")
                if len(result.vectors) != len(batch):
                    raise IndexingError("embedding_count_mismatch")
                records = []
                for (view, span, block_ids, text), vector in zip(batch, result.vectors, strict=True):
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
                        )
                    )
                    ordinal += 1
                await self._index.add_chunks(scope, version.id, tuple(records))
            await self._index.mark_ready(scope, version.id)
        except BaseException:
            await self._index.mark_failed(scope, version.id)
            raise
        return await self._index.get_index_version(scope, version.id)

    async def inspect_chunks(
        self, scope: Scope, index_version_id: str, *, offset: int = 0, limit: int = 100
    ) -> tuple[ChunkInspection, ...]:
        """Inspect bounded chunks; payloads are re-read once per chunk until per-call caching is added."""
        if offset < 0 or limit < 1 or limit > 100:
            raise IndexingError("invalid_page")
        rows = await self._index.list_chunks(scope, index_version_id, offset=offset, limit=limit)
        inspections = []
        for row in rows:
            anchors = await self._anchors.anchors_for_span(scope, row.extraction_view_id, row.start, row.end)
            inspections.append(ChunkInspection(**row.model_dump(), anchors=anchors))
        return tuple(inspections)
