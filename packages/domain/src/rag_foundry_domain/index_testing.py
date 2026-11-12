"""Reusable conformance checks for IndexRepository implementations."""

import hashlib
import math
from collections.abc import Awaitable, Sequence
from datetime import UTC, datetime
from typing import Protocol

from pydantic import ValidationError

from rag_foundry_domain.documents import Scope, new_id
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_domain.graphs import (
    Entity,
    EntityGraph,
    GraphSpan,
    Relation,
    entity_id_for,
    graph_key_for,
    relation_id_for,
)
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
from rag_foundry_domain.representations import (
    NewRepresentation,
    RepresentationRecord,
    RepresentationSet,
    ScoredRepresentation,
    representation_key_for,
)
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


def make_chunk(
    view_id: str,
    ordinal: int,
    vector: Sequence[float],
    text: str = "t",
    *,
    parent_start_cp: int | None = None,
    parent_end_cp: int | None = None,
) -> NewChunk:
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
        parent_start_cp=parent_start_cp,
        parent_end_cp=parent_end_cp,
    )


class IndexRepoFactory(Protocol):
    def __call__(self, scopes: Sequence[Scope]) -> Awaitable[tuple[DocumentRepository, object, IndexRepository]]: ...


class InMemoryIndexRepository:
    def __init__(self, views: ExtractionRepository) -> None:
        self._views = views
        self._versions: dict[str, IndexVersion] = {}
        self._chunks: dict[str, list[ChunkRecord]] = {}
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._sparse_vectors: dict[tuple[str, str], tuple[tuple[str, ...], tuple[float, ...]]] = {}
        self._rep_sets: dict[tuple[str, str], RepresentationSet] = {}
        self._reps: dict[tuple[str, str], list[tuple[RepresentationRecord, tuple[float, ...]]]] = {}
        self._graphs: dict[tuple[str, str], EntityGraph] = {}

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

    async def chunk_embeddings(
        self, scope: Scope, index_version_id: str, chunk_ids: tuple[str, ...]
    ) -> dict[str, tuple[float, ...]]:
        self._get(scope, index_version_id)
        if len(chunk_ids) > 100:
            raise IndexingError("too_many_ids")
        allowed = {chunk.id for chunk in self._chunks[index_version_id]}
        return {chunk_id: self._vectors[chunk_id] for chunk_id in chunk_ids if chunk_id in allowed}

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

    async def add_sparse(self, scope: Scope, index_version_id: str, encoder_key: str, rows: tuple) -> None:
        self._get(scope, index_version_id)
        for chunk_id, vector in rows:
            self._sparse_vectors[(chunk_id, encoder_key)] = vector

    async def sparse_search(self, scope: Scope, index_version_id: str, encoder_key: str, query, k: int):
        self._get(scope, index_version_id)
        raise IndexingError("sparse_unavailable")

    async def fts_search(self, scope, index_version_id, query, k, *, operator):
        raise IndexingError("fts_unavailable")

    async def add_representations(
        self,
        scope: Scope,
        index_version_id: str,
        rep_set: RepresentationSet,
        rows: tuple[NewRepresentation, ...],
    ) -> None:
        row = self._get(scope, index_version_id)
        if row.state is not IndexState.READY:
            raise IndexingError("not_ready")
        if len(rows) > 500:
            raise IndexingError("too_many_rows")
        if not rows:
            return
        chunk_ids = {c.id for c in self._chunks[index_version_id]}
        if any(r.chunk_id not in chunk_ids for r in rows):
            raise NotFoundError()
        if any(len(r.embedding) != row.embedding_dim for r in rows):
            raise IndexingError("dimension_mismatch")
        slot = (index_version_id, rep_set.representation_key)
        known = self._rep_sets.get(slot)
        if known is not None and known != rep_set:
            raise IndexingError("representation_set_conflict")
        existing = {(r.chunk_id, r.kind, r.ordinal) for r, _ in self._reps.get(slot, [])}
        incoming = [(r.chunk_id, r.kind, r.ordinal) for r in rows]
        if len(set(incoming)) != len(incoming) or existing & set(incoming):
            raise IndexingError("duplicate_representation")
        added = [
            (
                RepresentationRecord(
                    **r.model_dump(exclude={"embedding"}),
                    index_version_id=row.id,
                    tenant_id=scope.tenant_id,
                    project_id=scope.project_id,
                    representation_key=rep_set.representation_key,
                ),
                r.embedding,
            )
            for r in rows
        ]
        self._rep_sets[slot] = rep_set
        self._reps.setdefault(slot, []).extend(added)

    async def get_representation_set(self, scope: Scope, index_version_id: str, key: str) -> RepresentationSet:
        self._get(scope, index_version_id)
        found = self._rep_sets.get((index_version_id, key))
        if found is None:
            raise NotFoundError()
        return found

    async def representation_search(
        self, scope: Scope, index_version_id: str, key: str, query: tuple[float, ...], k: int
    ) -> list[ScoredRepresentation]:
        row = self._get(scope, index_version_id)
        if row.state is not IndexState.READY:
            raise IndexingError("not_ready")
        if not 1 <= k <= 100:
            raise IndexingError("invalid_k")
        q = check_vector(query)
        if len(q) != row.embedding_dim:
            raise IndexingError("dimension_mismatch")
        slot = (index_version_id, key)
        if slot not in self._rep_sets:
            raise NotFoundError()
        stored = self._reps.get(slot, [])
        if not stored:
            raise IndexingError("representations_unavailable")
        chunks = {c.id: c for c in self._chunks[index_version_id]}
        candidates = []
        for rec, v in stored:
            cosine = sum(a * b for a, b in zip(q, v, strict=True)) / (
                math.sqrt(sum(a * a for a in q)) * math.sqrt(sum(b * b for b in v))
            )
            chunk = chunks[rec.chunk_id]
            candidates.append(((1.0 - cosine, chunk.ordinal, rec.kind, rec.ordinal), chunk, rec))
        candidates.sort(key=lambda x: x[0])
        return [
            ScoredRepresentation(chunk=c, representation=r, distance=key_[0], rank=i + 1)
            for i, (key_, c, r) in enumerate(candidates[:k])
        ]

    async def add_entity_graph(self, scope: Scope, index_version_id: str, graph: EntityGraph) -> None:
        row = self._get(scope, index_version_id)
        if row.state is not IndexState.READY:
            raise IndexingError("not_ready")
        graph.check_scope(scope, index_version_id)
        slot = (index_version_id, graph.graph_key)
        if slot in self._graphs:
            raise IndexingError("graph_exists")
        chunks = {c.id: c for c in self._chunks[index_version_id]}
        spans = [m for e in graph.entities for m in e.mentions] + [s for r in graph.relations for s in r.evidence]
        for span in spans:
            chunk = chunks.get(span.chunk_id)
            if (
                chunk is None
                or span.extraction_view_id != chunk.extraction_view_id
                or not chunk.start <= span.start < span.end <= chunk.end
                or chunk.text[span.start - chunk.start : span.end - chunk.start] != span.quote
            ):
                raise IndexingError("graph_source_mismatch")
        self._graphs[slot] = graph.model_copy(deep=True)

    async def get_entity_graph(self, scope: Scope, index_version_id: str, graph_key: str) -> EntityGraph:
        self._get(scope, index_version_id)
        found = self._graphs.get((index_version_id, graph_key))
        if found is None:
            raise NotFoundError()
        return found.model_copy(deep=True)


async def check_parent_spans(factory: IndexRepoFactory) -> None:
    scope = make_scope()
    docs, views_repo, repo = await factory([scope])
    out = await docs.create_document_or_get(scope, make_new_version("parent-spans"), display_name="fixture")
    view = (await views_repo.create_view_or_get(scope, make_new_extraction(out.version, tag="parent-spans"))).view
    version = await repo.create_index_version(scope, make_new_index_version((view.id,)))
    plain = make_chunk(view.id, 0, (1, 0, 0, 0), "plain")
    linked = make_chunk(view.id, 1, (1, 0, 0, 0), "child", parent_start_cp=0, parent_end_cp=7)
    await repo.add_chunks(scope, version.id, (plain, linked))
    rows = await repo.list_chunks(scope, version.id)
    if [(row.parent_start_cp, row.parent_end_cp) for row in rows] != [(None, None), (0, 7)]:
        raise AssertionError("parent offsets did not round-trip through list_chunks")
    await repo.mark_ready(scope, version.id)
    hits = await repo.search(scope, version.id, (1, 0, 0, 0), 2)
    if [(hit.chunk.parent_start_cp, hit.chunk.parent_end_cp) for hit in hits] != [(None, None), (0, 7)]:
        raise AssertionError("parent offsets did not round-trip through search")


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
    await check_parent_spans(factory)


async def check_chunk_embeddings(factory: IndexRepoFactory) -> None:
    scope, other = make_scope(), make_scope()
    docs, view_repo, repo = await factory([scope, other])
    views = []
    for i, current in enumerate((scope, other)):
        doc = await docs.create_document_or_get(current, make_new_version(f"vectors-{i}"), display_name="fixture")
        views.append(
            (await view_repo.create_view_or_get(current, make_new_extraction(doc.version, tag=f"vectors-{i}"))).view
        )
    version = await repo.create_index_version(scope, make_new_index_version((views[0].id,)))
    await repo.add_chunks(scope, version.id, (make_chunk(views[0].id, 0, (0.1, 0.2, 0.3, 0.4)),))
    own = (await repo.list_chunks(scope, version.id))[0]
    foreign_version = await repo.create_index_version(other, make_new_index_version((views[1].id,)))
    await repo.add_chunks(other, foreign_version.id, (make_chunk(views[1].id, 0, (0.4, 0.3, 0.2, 0.1)),))
    foreign = (await repo.list_chunks(other, foreign_version.id))[0]
    vectors = await repo.chunk_embeddings(scope, version.id, (own.id, foreign.id, "not-a-uuid"))
    if set(vectors) != {own.id} or any(
        abs(a - b) > 1e-6 for a, b in zip(vectors[own.id], (0.1, 0.2, 0.3, 0.4), strict=True)
    ):
        raise AssertionError("chunk embeddings must round-trip and exclude unrelated IDs")
    await _expect_not_found(repo.chunk_embeddings(other, version.id, (own.id,)))
    await _expect_error(repo.chunk_embeddings(scope, version.id, tuple(str(i) for i in range(101))), "too_many_ids")


INDEX_REPOSITORY_CHECKS = (check_index_repository, check_chunk_embeddings)


def make_representation(
    chunk_id: str, kind: str, ordinal: int, text: str, vector: Sequence[float]
) -> NewRepresentation:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return NewRepresentation(
        chunk_id=chunk_id,
        kind=kind,
        ordinal=ordinal,
        text=text,
        text_sha256=digest,
        embedded_input_sha256=hashlib.sha256(("embed:" + text).encode("utf-8")).hexdigest(),
        embedding=tuple(vector),
    )


async def check_representations(factory: IndexRepoFactory) -> None:
    scope, other_tenant = make_scope(), make_scope()
    docs, view_repo, repo = await factory([scope, other_tenant])
    # hasattr, not isinstance: per-call adapters resolve methods dynamically, which runtime protocols cannot see.
    if not all(hasattr(repo, n) for n in ("add_representations", "get_representation_set", "representation_search")):
        raise AssertionError("repository must implement RepresentationStore")
    views = []
    for i, sc in enumerate((scope, other_tenant)):
        out = await docs.create_document_or_get(sc, make_new_version(f"rep-{i}"), display_name="fixture")
        views.append((await view_repo.create_view_or_get(sc, make_new_extraction(out.version, tag=f"rep-{i}"))).view)
    version = await repo.create_index_version(scope, make_new_index_version((views[0].id,)))
    await repo.add_chunks(
        scope,
        version.id,
        (make_chunk(views[0].id, 0, (1, 0, 0, 0), "alpha"), make_chunk(views[0].id, 1, (0, 1, 0, 0), "beta")),
    )
    chunk_a, chunk_b = await repo.list_chunks(scope, version.id)
    foreign_version = await repo.create_index_version(other_tenant, make_new_index_version((views[1].id,)))
    await repo.add_chunks(other_tenant, foreign_version.id, (make_chunk(views[1].id, 0, (1, 0, 0, 0), "foreign"),))
    foreign_chunk = (await repo.list_chunks(other_tenant, foreign_version.id))[0]

    key = representation_key_for({"technique": "R23", "model": "m", "prompt": 1})
    rep_set = RepresentationSet(representation_key=key, technique_id="R23", generator_json='{"model":"m"}')
    summary_a = make_representation(chunk_a.id, "summary", 0, "summary of alpha", (1, 0, 0, 0))
    question_b0 = make_representation(chunk_b.id, "question", 0, "what is beta?", (0, 1, 0, 0))
    question_b1 = make_representation(chunk_b.id, "question", 1, "why beta?", (0.6, 0.8, 0, 0))
    rows = (summary_a, question_b0, question_b1)

    await _expect_error(repo.add_representations(scope, version.id, rep_set, rows), "not_ready")
    await _expect_error(repo.representation_search(scope, version.id, key, (1, 0, 0, 0), 1), "not_ready")
    await repo.mark_ready(scope, version.id)

    await repo.add_representations(scope, version.id, rep_set, ())
    await _expect_not_found(repo.get_representation_set(scope, version.id, key))
    await _expect_error(repo.add_representations(scope, version.id, rep_set, (summary_a,) * 501), "too_many_rows")
    # 500 rows is within the limit: a repeated row is a duplicate, and nothing is written.
    await _expect_error(
        repo.add_representations(scope, version.id, rep_set, (summary_a,) * 500), "duplicate_representation"
    )
    await _expect_not_found(repo.get_representation_set(scope, version.id, key))
    unknown_chunk = make_representation(new_id(), "summary", 0, "orphan", (1, 0, 0, 0))
    foreign_row = make_representation(foreign_chunk.id, "summary", 0, "foreign summary", (1, 0, 0, 0))
    for bad in (unknown_chunk, foreign_row):
        await _expect_not_found(repo.add_representations(scope, version.id, rep_set, (summary_a, bad)))
    short = make_representation(chunk_a.id, "question", 2, "short vector", (1, 0, 0))
    await _expect_error(repo.add_representations(scope, version.id, rep_set, (summary_a, short)), "dimension_mismatch")
    await _expect_not_found(repo.get_representation_set(scope, version.id, key))

    for stranger in (other_tenant, Scope(tenant_id=scope.tenant_id, project_id=new_id())):
        await _expect_not_found(repo.add_representations(stranger, version.id, rep_set, rows))
        await _expect_not_found(repo.get_representation_set(stranger, version.id, key))
        await _expect_not_found(repo.representation_search(stranger, version.id, key, (1, 0, 0, 0), 1))
    await _expect_not_found(repo.get_representation_set(scope, version.id, key))

    await repo.add_representations(scope, version.id, rep_set, rows)
    if await repo.get_representation_set(scope, version.id, key) != rep_set:
        raise AssertionError("representation set must round-trip")
    for stranger in (other_tenant, Scope(tenant_id=scope.tenant_id, project_id=new_id())):
        late = make_representation(chunk_a.id, "question", 5, "late question", (0, 0, 1, 0))
        await _expect_not_found(repo.get_representation_set(stranger, version.id, key))
        await _expect_not_found(repo.representation_search(stranger, version.id, key, (1, 0, 0, 0), 1))
        await _expect_not_found(repo.add_representations(stranger, version.id, rep_set, (late,)))
    other_technique = RepresentationSet(
        representation_key=key, technique_id="R24", generator_json=rep_set.generator_json
    )
    other_generator = RepresentationSet(representation_key=key, technique_id="R23", generator_json='{"model":"x"}')
    for conflicting in (other_technique, other_generator):
        extra = make_representation(chunk_a.id, "question", 3, "extra question", (0, 0, 1, 0))
        await _expect_error(
            repo.add_representations(scope, version.id, conflicting, (extra,)), "representation_set_conflict"
        )
    await _expect_error(
        repo.add_representations(scope, version.id, rep_set, (question_b1,)), "duplicate_representation"
    )
    twin = make_representation(chunk_a.id, "question", 4, "twin", (0, 0, 1, 0))
    await _expect_error(repo.add_representations(scope, version.id, rep_set, (twin, twin)), "duplicate_representation")

    hits = await repo.representation_search(scope, version.id, key, (1, 0, 0, 0), 3)
    got = [(h.chunk.id, h.representation.kind, h.representation.ordinal, h.rank) for h in hits]
    if got != [(chunk_a.id, "summary", 0, 1), (chunk_b.id, "question", 1, 2), (chunk_b.id, "question", 0, 3)]:
        raise AssertionError("representation search order must follow distance then chunk, kind, ordinal")
    if any(abs(h.distance - d) > 1e-5 for h, d in zip(hits, (0.0, 0.4, 1.0), strict=True)):
        raise AssertionError("representation distances must be approximately 0.0, 0.4, 1.0")
    if any(h.representation.synthetic is not True for h in hits):
        raise AssertionError("representations must be labelled synthetic")
    if [h.chunk.text for h in hits] != ["alpha", "beta", "beta"] or hits[0].representation.text == hits[0].chunk.text:
        raise AssertionError("search must return original chunk text separate from synthetic text")
    if [h.chunk.id for h in await repo.representation_search(scope, version.id, key, (1, 0, 0, 0), 2)] != [
        chunk_a.id,
        chunk_b.id,
    ]:
        raise AssertionError("k=2 must return the first two hits")
    await _expect_error(repo.representation_search(scope, version.id, key, (1, 0, 0, 0), 0), "invalid_k")
    await _expect_error(repo.representation_search(scope, version.id, key, (1, 0, 0, 0), 101), "invalid_k")
    await _expect_error(repo.representation_search(scope, version.id, key, (1, 0, 0), 1), "dimension_mismatch")
    await _expect_not_found(repo.representation_search(scope, version.id, "0" * 64, (1, 0, 0, 0), 1))
    await _expect_not_found(repo.get_representation_set(scope, version.id, "0" * 64))
    if len(await repo.representation_search(scope, version.id, key, (1, 0, 0, 0), 100)) != 3:
        raise AssertionError("failed writes must not add rows")


REPRESENTATION_STORE_CHECKS = (check_representations,)


def make_entity_graph(scope: Scope, version: IndexVersion, chunks: Sequence[ChunkRecord]) -> EntityGraph:
    """Two entities and one relation, each cited by spans over the first chunk's text."""
    chunk = chunks[0]
    if len(chunk.text) < 4:
        raise ValueError("chunk text too short for graph fixture")
    half = len(chunk.text) // 2

    def span(start: int, end: int) -> GraphSpan:
        return GraphSpan(
            chunk_id=chunk.id,
            extraction_view_id=chunk.extraction_view_id,
            start=chunk.start + start,
            end=chunk.start + end,
            quote=chunk.text[start:end],
        )

    first = Entity(entity_id=entity_id_for("first", "thing"), name="first", type="thing", mentions=(span(0, half),))
    second = Entity(
        entity_id=entity_id_for("second", "thing"), name="second", type="thing", mentions=(span(half, len(chunk.text)),)
    )
    relation = Relation(
        relation_id=relation_id_for(first.entity_id, "relates to", second.entity_id),
        source_id=first.entity_id,
        target_id=second.entity_id,
        label="relates to",
        evidence=(span(0, len(chunk.text)),),
    )
    key = graph_key_for({"index_version_id": version.id, "chunk": chunk.id, "text": chunk.text})
    return EntityGraph(
        schema="rag_foundry.entity_graph/1",
        graph_key=key,
        tenant_id=scope.tenant_id,
        project_id=scope.project_id,
        index_version_id=version.id,
        generator_json="{}",
        entities=tuple(sorted((first, second), key=lambda e: e.entity_id)),
        relations=(relation,),
    )


async def check_entity_graphs(factory: IndexRepoFactory) -> None:
    scope, other_tenant = make_scope(), make_scope()
    docs, view_repo, repo = await factory([scope, other_tenant])
    # hasattr, not isinstance: per-call adapters resolve methods dynamically, which runtime protocols cannot see.
    if not all(hasattr(repo, n) for n in ("add_entity_graph", "get_entity_graph")):
        raise AssertionError("repository must implement EntityGraphStore")
    views = []
    for i, sc in enumerate((scope, other_tenant)):
        out = await docs.create_document_or_get(sc, make_new_version(f"graph-{i}"), display_name="fixture")
        views.append((await view_repo.create_view_or_get(sc, make_new_extraction(out.version, tag=f"graph-{i}"))).view)
    arabic = "محمد يعمل في الشركة"
    version = await repo.create_index_version(scope, make_new_index_version((views[0].id,)))
    await repo.add_chunks(
        scope,
        version.id,
        (make_chunk(views[0].id, 0, (1, 0, 0, 0), "Ada wrote notes"), make_chunk(views[0].id, 1, (0, 1, 0, 0), arabic)),
    )
    chunk_a, chunk_ar = await repo.list_chunks(scope, version.id)
    foreign_version = await repo.create_index_version(other_tenant, make_new_index_version((views[1].id,)))
    await repo.add_chunks(other_tenant, foreign_version.id, (make_chunk(views[1].id, 0, (1, 0, 0, 0), "foreign text"),))
    await repo.mark_ready(other_tenant, foreign_version.id)
    foreign_chunk = (await repo.list_chunks(other_tenant, foreign_version.id))[0]
    sibling = await repo.create_index_version(scope, make_new_index_version((views[0].id,)))
    await repo.add_chunks(scope, sibling.id, (make_chunk(views[0].id, 0, (1, 0, 0, 0), "Ada wrote notes"),))
    await repo.mark_ready(scope, sibling.id)
    sibling_chunk = (await repo.list_chunks(scope, sibling.id))[0]

    graph = make_entity_graph(scope, version, (chunk_a,))
    await _expect_error(repo.add_entity_graph(scope, version.id, graph), "not_ready")
    await repo.mark_ready(scope, version.id)

    # A graph built for version B must not be stored under version A, even in the same scope.
    sibling_graph = make_entity_graph(scope, sibling, (sibling_chunk,))
    await _expect_error(repo.add_entity_graph(scope, version.id, sibling_graph), "graph_scope_mismatch")
    await _expect_not_found(repo.get_entity_graph(scope, version.id, sibling_graph.graph_key))

    for stranger in (other_tenant, Scope(tenant_id=scope.tenant_id, project_id=new_id())):
        await _expect_not_found(repo.add_entity_graph(stranger, version.id, graph))
        await _expect_not_found(repo.get_entity_graph(stranger, version.id, graph.graph_key))
    await _expect_not_found(repo.get_entity_graph(scope, version.id, graph.graph_key))

    def bad_graph(span: GraphSpan, tag: str) -> EntityGraph:
        entity = graph.entities[0].model_copy(update={"mentions": (span,)})
        return graph.model_copy(
            update={"graph_key": graph_key_for({"bad": tag}), "entities": (entity, *graph.entities[1:])}
        )

    good = graph.entities[0].mentions[0]
    bad_spans = {
        "quote": good.model_copy(update={"quote": "X" * len(good.quote)}),
        "outside": GraphSpan(
            chunk_id=chunk_a.id,
            extraction_view_id=chunk_a.extraction_view_id,
            start=chunk_a.end,
            end=chunk_a.end + 2,
            quote="xx",
        ),
        "other_tenant_chunk": GraphSpan(
            chunk_id=foreign_chunk.id,
            extraction_view_id=foreign_chunk.extraction_view_id,
            start=0,
            end=3,
            quote="for",
        ),
        # Same scope and same text as chunk_a, but the chunk belongs to a different index version.
        "sibling_version_chunk": GraphSpan(
            chunk_id=sibling_chunk.id,
            extraction_view_id=sibling_chunk.extraction_view_id,
            start=0,
            end=3,
            quote="Ada",
        ),
        # Valid quote and offsets, but the view is not the chunk's view.
        "wrong_view": good.model_copy(update={"extraction_view_id": views[1].id}),
    }
    for tag, bad in bad_spans.items():
        await _expect_error(repo.add_entity_graph(scope, version.id, bad_graph(bad, tag)), "graph_source_mismatch")
        await _expect_not_found(repo.get_entity_graph(scope, version.id, graph_key_for({"bad": tag})))

    await repo.add_entity_graph(scope, version.id, graph)
    if await repo.get_entity_graph(scope, version.id, graph.graph_key) != graph:
        raise AssertionError("entity graph must round-trip")
    await _expect_error(repo.add_entity_graph(scope, version.id, graph), "graph_exists")
    # The duplicate key is reported before span checks, so a duplicate with a bad span is still graph_exists.
    duplicate_bad = bad_graph(bad_spans["quote"], "dup").model_copy(update={"graph_key": graph.graph_key})
    await _expect_error(repo.add_entity_graph(scope, version.id, duplicate_bad), "graph_exists")
    for stranger in (other_tenant, Scope(tenant_id=scope.tenant_id, project_id=new_id())):
        await _expect_not_found(repo.get_entity_graph(stranger, version.id, graph.graph_key))

    arabic_graph = make_entity_graph(scope, version, (chunk_ar,))
    await repo.add_entity_graph(scope, version.id, arabic_graph)
    loaded = await repo.get_entity_graph(scope, version.id, arabic_graph.graph_key)
    quotes = {m.quote for e in loaded.entities for m in e.mentions}
    if loaded != arabic_graph or quotes != {arabic[: len(arabic) // 2], arabic[len(arabic) // 2 :]}:
        raise AssertionError("Arabic quotes must round-trip unchanged")


GRAPH_STORE_CHECKS = (check_entity_graphs,)
