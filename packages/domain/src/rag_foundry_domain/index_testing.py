"""Reusable conformance checks for IndexRepository implementations."""

import hashlib
import math
from collections.abc import Awaitable, Sequence
from datetime import UTC, datetime
from typing import Protocol

from pydantic import ValidationError

from rag_foundry_domain.documents import Scope, new_id
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_domain.indexing import (
    ChunkRecord,
    IndexState,
    IndexVersion,
    NewChunk,
    NewIndexVersion,
    ScoredChunk,
    check_vector,
    corpus_sha256_for,
    embedding_key_for,
)
from rag_foundry_domain.ports import DocumentRepository, ExtractionRepository, IndexRepository
from rag_foundry_domain.testing import make_new_extraction, make_new_version, make_scope


def make_new_index_version(view_ids: tuple[str, ...], *, dim: int = 4) -> NewIndexVersion:
    config = "{}"
    return NewIndexVersion(
        chunker_id="C05",
        chunker_version="1.0.0",
        chunker_config_json=config,
        chunker_config_sha256=hashlib.sha256(config.encode()).hexdigest(),
        embedding_profile_id="test-profile",
        embedding_provider="test",
        embedding_model="test-model",
        embedding_dim=dim,
        distance="cosine",
        embedding_key=embedding_key_for("test", "test-model", dim),
        extraction_view_ids=tuple(sorted(view_ids)),
        corpus_sha256=corpus_sha256_for((v, hashlib.sha256(v.encode()).hexdigest()) for v in view_ids),
        unanchored_chunk_count=0,
    )


def make_chunk(view_id: str, ordinal: int, vector: Sequence[float], text: str = "t") -> NewChunk:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return NewChunk(
        extraction_view_id=view_id,
        ordinal=ordinal,
        start=0,
        end=len(text),
        split_level=0,
        block_ids=("b00000",),
        text=text,
        text_sha256=digest,
        embedding=tuple(vector),
    )


class IndexRepoFactory(Protocol):
    def __call__(self, scopes: Sequence[Scope]) -> Awaitable[tuple[DocumentRepository, object, IndexRepository]]: ...


class InMemoryIndexRepository:
    def __init__(self, views: ExtractionRepository) -> None:
        self._views = views
        self._versions: dict[str, IndexVersion] = {}
        self._chunks: dict[str, list[ChunkRecord]] = {}
        self._vectors: dict[str, tuple[float, ...]] = {}

    def _get(self, scope: Scope, iid: str) -> IndexVersion:
        v = self._versions.get(iid)
        if v is None or (v.tenant_id, v.project_id) != (scope.tenant_id, scope.project_id):
            raise NotFoundError()
        return v

    async def create_index_version(self, scope: Scope, new: NewIndexVersion) -> IndexVersion:
        for view_id in new.extraction_view_ids:
            await self._views.get_view(scope, view_id)
        row = IndexVersion(
            **new.model_dump(),
            id=new_id(),
            tenant_id=scope.tenant_id,
            project_id=scope.project_id,
            state=IndexState.BUILDING,
            chunk_count=0,
            created_at=datetime.now(UTC),
            ready_at=None,
        )
        self._versions[row.id] = row
        self._chunks[row.id] = []
        return row

    async def add_chunks(self, scope: Scope, index_version_id: str, chunks: tuple[NewChunk, ...]) -> None:
        row = self._get(scope, index_version_id)
        if row.state is not IndexState.BUILDING:
            raise IndexingError("not_building")
        old = self._chunks[index_version_id]
        ordinals = [c.ordinal for c in old]
        if len(set(ordinals + [c.ordinal for c in chunks])) != len(ordinals) + len(chunks):
            raise IndexingError("duplicate_ordinal")
        if any(len(c.embedding) != row.embedding_dim for c in chunks):
            raise IndexingError("dimension_mismatch")
        for c in chunks:
            if c.extraction_view_id not in row.extraction_view_ids:
                raise NotFoundError()
        added = [
            ChunkRecord(
                **c.model_dump(exclude={"embedding"}),
                id=new_id(),
                index_version_id=row.id,
                tenant_id=scope.tenant_id,
                project_id=scope.project_id,
            )
            for c in chunks
        ]
        self._chunks[index_version_id] = old + added
        self._vectors.update({record.id: chunk.embedding for record, chunk in zip(added, chunks, strict=True)})

    async def mark_ready(self, scope: Scope, index_version_id: str) -> IndexVersion:
        row = self._get(scope, index_version_id)
        if row.state is not IndexState.BUILDING:
            raise IndexingError("not_building")
        if not self._chunks[index_version_id]:
            raise IndexingError("empty_index")
        row = row.model_copy(
            update={
                "state": IndexState.READY,
                "chunk_count": len(self._chunks[index_version_id]),
                "ready_at": datetime.now(UTC),
            }
        )
        self._versions[row.id] = row
        return row

    async def mark_failed(self, scope: Scope, index_version_id: str) -> IndexVersion:
        row = self._get(scope, index_version_id)
        if row.state is not IndexState.BUILDING:
            raise IndexingError("not_building")
        row = row.model_copy(update={"state": IndexState.FAILED})
        self._versions[row.id] = row
        return row

    async def get_index_version(self, scope: Scope, index_version_id: str) -> IndexVersion:
        return self._get(scope, index_version_id)

    async def list_chunks(
        self, scope: Scope, index_version_id: str, *, offset: int = 0, limit: int = 100
    ) -> list[ChunkRecord]:
        self._get(scope, index_version_id)
        if offset < 0 or not 1 <= limit <= 500:
            raise IndexingError("invalid_page")
        return sorted(self._chunks[index_version_id], key=lambda c: c.ordinal)[offset : offset + limit]

    async def search(self, scope: Scope, index_version_id: str, query: tuple[float, ...], k: int):
        row = self._get(scope, index_version_id)
        if row.state is not IndexState.READY:
            raise IndexingError("not_ready")
        if not 1 <= k <= 100:
            raise IndexingError("invalid_k")
        q = check_vector(query)
        if len(q) != row.embedding_dim:
            raise IndexingError("dimension_mismatch")
        candidates = []
        for c in self._chunks[index_version_id]:
            v = self._vectors.get(c.id)
            if v is None:
                continue
            cosine = sum(a * b for a, b in zip(q, v, strict=True)) / (
                math.sqrt(sum(a * a for a in q)) * math.sqrt(sum(b * b for b in v))
            )
            candidates.append((1.0 - cosine, c.ordinal, c))
        candidates.sort(key=lambda x: (x[0], x[1]))
        return [ScoredChunk(chunk=c, distance=d, rank=i + 1) for i, (d, _, c) in enumerate(candidates[:k])]


async def _expect_error(awaitable, code: str) -> None:
    try:
        await awaitable
    except IndexingError as exc:
        if exc.code == code:
            return
    except NotFoundError:
        if code == "not_found":
            return
    raise AssertionError(f"expected {code}")


async def _expect_not_found(awaitable) -> str:
    try:
        await awaitable
    except NotFoundError as exc:
        return str(exc)
    raise AssertionError("expected not_found")


async def check_index_repository(factory: IndexRepoFactory) -> None:
    scope, other_tenant = make_scope(), make_scope()
    docs, view_repo, repo = await factory([scope, other_tenant])
    views = []
    for i, sc in enumerate((scope, other_tenant)):
        out = await docs.create_document_or_get(sc, make_new_version(f"idx-{i}"), display_name="fixture")
        v = await view_repo.create_view_or_get(sc, make_new_extraction(out.version, tag=f"idx-{i}"))
        views.append(v.view)
    version = await repo.create_index_version(scope, make_new_index_version((views[0].id,)))
    other_project = Scope(tenant_id=scope.tenant_id, project_id=new_id())
    not_found = await _expect_not_found(repo.get_index_version(scope, "not-a-uuid"))
    if await _expect_not_found(repo.get_index_version(scope, new_id())) != not_found:
        raise AssertionError("unknown and malformed IDs must have identical NotFoundError messages")
    await _expect_not_found(repo.list_chunks(scope, "not-a-uuid"))
    for lookup_scope in (other_tenant, other_project):
        if await _expect_not_found(repo.get_index_version(lookup_scope, version.id)) != not_found:
            raise AssertionError("unknown and out-of-scope IDs must have identical NotFoundError messages")
        for operation in (
            repo.search(lookup_scope, version.id, (1, 0, 0, 0), 1),
            repo.add_chunks(lookup_scope, version.id, (make_chunk(views[0].id, 20, (1, 0, 0, 0)),)),
            repo.mark_ready(lookup_scope, version.id),
            repo.mark_failed(lookup_scope, version.id),
            repo.list_chunks(lookup_scope, version.id),
        ):
            if await _expect_not_found(operation) != not_found:
                raise AssertionError("out-of-scope operations must raise the same NotFoundError")
    before_versions = await repo.list_chunks(scope, version.id)
    await _expect_not_found(repo.create_index_version(scope, make_new_index_version((views[1].id,))))
    if await repo.list_chunks(scope, version.id) != before_versions:
        raise AssertionError("out-of-scope version creation changed existing index data")
    await _expect_error(repo.mark_ready(scope, version.id), "empty_index")
    await _expect_error(repo.list_chunks(scope, version.id, offset=-1), "invalid_page")
    await _expect_error(repo.list_chunks(scope, version.id, limit=0), "invalid_page")
    await _expect_error(repo.list_chunks(scope, version.id, limit=501), "invalid_page")
    chunk_a = make_chunk(views[0].id, 2, (1, 0, 0, 0), "\u0627\u064e\U0001f642")
    chunk_b = make_chunk(views[0].id, 1, (1, 0, 0, 0), "text")
    hidden = make_chunk(views[1].id, 0, (1, 0, 0, 0))
    await repo.add_chunks(scope, version.id, (chunk_a, chunk_b))
    existing = await repo.list_chunks(scope, version.id)
    for batch, code in (
        ((make_chunk(views[0].id, 3, (0, 1, 0, 0)), make_chunk(views[0].id, 4, (1, 0, 0))), "dimension_mismatch"),
        ((make_chunk(views[0].id, 3, (0, 1, 0, 0)), chunk_b), "duplicate_ordinal"),
        ((make_chunk(views[0].id, 3, (0, 1, 0, 0)), make_chunk(views[0].id, 3, (0, 0, 1, 0))), "duplicate_ordinal"),
        ((make_chunk(views[0].id, 3, (0, 1, 0, 0)), hidden), "not_found"),
    ):
        await _expect_error(repo.add_chunks(scope, version.id, batch), code)
        if await repo.list_chunks(scope, version.id) != existing:
            raise AssertionError(f"{code} batch failure must be atomic")
    if [c.ordinal for c in await repo.list_chunks(scope, version.id)] != [1, 2]:
        raise AssertionError("chunks must list by ordinal")
    ready = await repo.mark_ready(scope, version.id)
    if ready.chunk_count != 2 or ready.ready_at is None:
        raise AssertionError("mark_ready must set count and ready timestamp")
    hits = await repo.search(scope, version.id, (1, 0, 0, 0), 10)
    if [h.chunk.ordinal for h in hits] != [1, 2] or [h.rank for h in hits] != [1, 2]:
        raise AssertionError("cosine ties must break by ordinal and rank sequentially")
    if any(abs(hit.distance) > 1e-5 for hit in hits):
        raise AssertionError("cosine distance must match the exact-vector result within 1e-5")
    angled = make_chunk(views[0].id, 3, (1, 1, 0, 0), "angle")
    orthogonal = make_chunk(views[0].id, 4, (0, 1, 0, 0), "orthogonal")
    ranked = await repo.create_index_version(scope, make_new_index_version((views[0].id,)))
    await repo.add_chunks(scope, ranked.id, (chunk_b, chunk_a, angled, orthogonal))
    await repo.mark_ready(scope, ranked.id)
    ranked_hits = await repo.search(scope, ranked.id, (1, 0, 0, 0), 10)
    if [h.chunk.ordinal for h in ranked_hits] != [1, 2, 3, 4]:
        raise AssertionError("cosine results must order exact ties, angled, then orthogonal")
    expected = [0.0, 0.0, 1.0 - 1.0 / math.sqrt(2.0), 1.0]
    if any(
        abs(hit.distance - dist) > 1e-5 or abs(hit.score - (1.0 - hit.distance)) > 1e-5
        for hit, dist in zip(ranked_hits, expected, strict=True)
    ):
        raise AssertionError("cosine distances and scores must match expected values")
    top = await repo.search(scope, ranked.id, (1, 0, 0, 0), 1)
    if len(top) != 1 or top[0].rank != 1:
        raise AssertionError("k=1 must return one hit ranked 1")
    foreign_version = await repo.create_index_version(other_tenant, make_new_index_version((views[1].id,)))
    await repo.add_chunks(other_tenant, foreign_version.id, (make_chunk(views[1].id, 0, (1, 0, 0, 0)),))
    await repo.mark_ready(other_tenant, foreign_version.id)
    foreign_chunk_ids = {c.id for c in await repo.list_chunks(other_tenant, foreign_version.id)}
    scoped_ids = {h.chunk.id for h in await repo.search(scope, version.id, (1, 0, 0, 0), 10)}
    if scoped_ids & foreign_chunk_ids:
        raise AssertionError("foreign chunk IDs must not appear in the scoped index search")
    await _expect_error(repo.add_chunks(scope, version.id, (make_chunk(views[0].id, 4, (1, 0, 0, 0)),)), "not_building")
    await _expect_error(repo.mark_ready(scope, version.id), "not_building")
    await _expect_error(repo.mark_failed(scope, version.id), "not_building")
    await _expect_error(repo.search(scope, version.id, (1, 0, 0, 0), 0), "invalid_k")
    await _expect_error(repo.search(scope, version.id, (1, 0, 0, 0), 101), "invalid_k")
    await _expect_error(repo.search(scope, version.id, (1, 0, 0), 1), "dimension_mismatch")
    await _expect_error(repo.search(scope, version.id, (0, 0, 0, 0), 1), "invalid_vector")
    empty_version = await repo.create_index_version(scope, make_new_index_version((views[0].id,)))
    await _expect_error(repo.search(scope, empty_version.id, (1, 0, 0, 0), 1), "not_ready")
    failed = await repo.mark_failed(scope, empty_version.id)
    if failed.state is not IndexState.FAILED:
        raise AssertionError("mark_failed must set failed state")
    await _expect_error(repo.mark_failed(scope, empty_version.id), "not_building")
    if "\u0627\u064e\U0001f642" not in [c.text for c in await repo.list_chunks(scope, version.id)]:
        raise AssertionError("Unicode text did not round-trip")
    for record in (ready, hits[0].chunk):
        try:
            record.id = new_id()
        except (ValidationError, TypeError, AttributeError):
            pass
        else:
            raise AssertionError("records must be frozen")


INDEX_REPOSITORY_CHECKS = (check_index_repository,)
