import asyncio
import functools
import os

import pytest
from rag_foundry_contracts.resolve import ResolutionReason, ResolutionStatus
from rag_foundry_domain.errors import AnchorError
from rag_foundry_domain.testing import InMemoryDocumentRepository, InMemoryExtractionRepository, make_scope
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from rag_foundry_ingestion.intake import IntakeService
from test_extraction_service import MD, SpyStore, ingest, setup
from test_intake import MemoryObjectStore, RecordingStore


def sync(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


@sync
async def test_create_and_resolve_anchor(tmp_path):
    _svc, intake, spy, _store, docs, views, scope = setup(tmp_path)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    doc_id, version_id = await ingest(intake, scope, MD.encode(), name="a.md", media="text/markdown")
    view = (
        await ExtractionService(spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits()).extract(
            scope, doc_id, version_id
        )
    ).view
    service = AnchorService(spy, uow, provider_namespace="test")
    payload = await service.anchor(
        scope,
        view.id,
        (await ExtractionService(spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits()).load_payload(scope, view))
        .blocks[0]
        .block_id,
        0,
        5,
    )
    spy.reads.clear()
    result = await service.resolve(scope, [payload])
    assert result[0].status is ResolutionStatus.EXACT
    assert result[0].exact == payload.exact
    assert spy.reads == [view.payload_key]


@sync
async def test_batch_order_and_one_payload_read_per_view(tmp_path):
    _svc, intake, spy, _store, docs, views, scope = setup(tmp_path)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    extraction = ExtractionService(spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    service = AnchorService(spy, uow, provider_namespace="test")
    anchors = []
    view_keys = []
    for suffix in ("one", "two"):
        doc_id, version_id = await ingest(
            intake, scope, (MD + f"\n\n{suffix}\n").encode(), name=f"{suffix}.md", media="text/markdown"
        )
        view = (await extraction.extract(scope, doc_id, version_id)).view
        view_keys.append(view.payload_key)
        anchors.append(await service.anchor(scope, view.id, "b00000", 0, 5))
    spy.reads.clear()
    results = await service.resolve(scope, [anchors[1], anchors[0], anchors[1]])
    assert [r.anchor_id for r in results] == [anchors[1].anchor_id, anchors[0].anchor_id, anchors[1].anchor_id]
    assert [r.status for r in results] == [ResolutionStatus.EXACT] * 3
    assert sorted(spy.reads) == sorted(view_keys)


@sync
async def test_scope_and_foreign_namespace_do_not_read(tmp_path):
    owner, other = make_scope(), make_scope()
    _svc, intake, spy, _store, docs, views, _ = setup(tmp_path, owner, other)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    doc_id, version_id = await ingest(intake, owner, MD.encode(), name="a.md", media="text/markdown")
    extraction = ExtractionService(spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    view = (await extraction.extract(owner, doc_id, version_id)).view
    service = AnchorService(spy, uow, provider_namespace="test")
    anchor = await service.anchor(owner, view.id, "b00000", 0, 5)
    spy.reads.clear()
    result = await service.resolve(other, [anchor])
    assert result[0].reason is ResolutionReason.NOT_FOUND
    assert spy.reads == []
    foreign = anchor.model_copy(update={"source": anchor.source.model_copy(update={"provider_namespace": "elsewhere"})})

    class NeverLookup:
        def __call__(self, *, read_only=False):
            raise AssertionError("foreign namespace looked up")

    foreign_result = await AnchorService(spy, NeverLookup(), provider_namespace="test").resolve(owner, [foreign])
    assert foreign_result[0].reason is ResolutionReason.FOREIGN_NAMESPACE


@sync
async def test_tampered_payload_and_view_limit(tmp_path):
    _svc, intake, spy, _store, docs, views, scope = setup(tmp_path)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    extraction = ExtractionService(spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    service = AnchorService(spy, uow, provider_namespace="test")
    anchors = []
    keys = []
    for i in range(11):
        doc_id, version_id = await ingest(
            intake, scope, (MD + f"\n\n{i}\n").encode(), name=f"{i}.md", media="text/markdown"
        )
        view = (await extraction.extract(scope, doc_id, version_id)).view
        keys.append(view.payload_key)
        anchors.append(await service.anchor(scope, view.id, "b00000", 0, 5))
    path = os.path.join(tmp_path, *keys[0].split("/"))
    os.chmod(path, 0o600)
    with open(path, "ab") as stream:
        stream.write(b"x")
    spy.reads.clear()
    damaged = await service.resolve(scope, [anchors[0]])
    assert damaged[0].reason is ResolutionReason.INTEGRITY_ERROR
    spy.reads.clear()
    with pytest.raises(AnchorError) as exc:
        await service.resolve(scope, anchors)
    assert exc.value.code == "too_many_views"
    assert spy.reads == []


@sync
async def test_unknown_view_is_not_found(tmp_path):
    _svc, intake, spy, _store, docs, views, scope = setup(tmp_path)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    doc_id, version_id = await ingest(intake, scope, MD.encode(), name="a.md", media="text/markdown")
    extraction = ExtractionService(spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    view = (await extraction.extract(scope, doc_id, version_id)).view
    service = AnchorService(spy, uow, provider_namespace="test")
    anchor = await service.anchor(scope, view.id, "b00000", 0, 5)
    unknown = anchor.model_copy(update={"source": anchor.source.model_copy(update={"extraction_view_id": "bad"})})
    spy.reads.clear()
    result = await service.resolve(scope, [unknown])
    assert result[0].reason is ResolutionReason.NOT_FOUND
    assert spy.reads == []


@sync
async def test_all_anchor_methods_read_payloads_without_open_uow(tmp_path):
    scope = make_scope()
    docs = InMemoryDocumentRepository([scope])
    views = InMemoryExtractionRepository(docs)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    store = RecordingStore(MemoryObjectStore(), lambda: uow.open_now)
    spy = SpyStore(store)
    intake = IntakeService(store, uow, max_bytes=100_000)
    extraction = ExtractionService(spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    doc_id, version_id = await ingest(intake, scope, MD.encode(), name="a.md", media="text/markdown")
    view = (await extraction.extract(scope, doc_id, version_id)).view
    payload = await extraction.load_payload(scope, view)
    service = AnchorService(spy, uow, provider_namespace="test")
    block = payload.blocks[0]
    store.calls.clear()
    spy.calls.clear()
    anchor = await service.anchor(scope, view.id, block.block_id, block.start, block.end)
    assert await service.anchors_for_span(scope, view.id, block.start, block.end)
    assert await service.resolve(scope, [anchor])
    assert store.calls == ["open_read", "open_read", "open_read"]


@sync
async def test_resolve_reads_three_anchors_from_two_views_in_one_read_uow(tmp_path):
    scope = make_scope()
    docs = InMemoryDocumentRepository([scope])
    views = InMemoryExtractionRepository(docs)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    store = RecordingStore(MemoryObjectStore(), lambda: uow.open_now)
    spy = SpyStore(store)
    intake = IntakeService(store, uow, max_bytes=100_000)
    extraction = ExtractionService(spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    service = AnchorService(spy, uow, provider_namespace="test")
    anchors = []
    for suffix in ("one", "two"):
        doc_id, version_id = await ingest(
            intake, scope, (MD + f"\n{suffix}\n").encode(), name=f"{suffix}.md", media="text/markdown"
        )
        view = (await extraction.extract(scope, doc_id, version_id)).view
        anchors.append(await service.anchor(scope, view.id, "b00000", 0, 5))
    before = uow.opened
    results = await service.resolve(scope, [anchors[0], anchors[1], anchors[0]])
    assert len(results) == 3
    assert uow.opened - before == 1
