import asyncio
import functools
import os

import pytest
from rag_foundry_contracts.resolve import ResolutionReason, ResolutionStatus
from rag_foundry_domain.errors import AnchorError
from rag_foundry_domain.testing import make_scope
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from test_extraction_service import MD, ingest, setup


def sync(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


@sync
async def test_create_and_resolve_anchor(tmp_path):
    _svc, intake, spy, _store, docs, views, scope = setup(tmp_path)
    doc_id, version_id = await ingest(intake, scope, MD.encode(), name="a.md", media="text/markdown")
    view = (
        await ExtractionService(spy, docs, views, extractors=EXTRACTORS, limits=ExtractionLimits()).extract(
            scope, doc_id, version_id
        )
    ).view
    service = AnchorService(spy, views, provider_namespace="test")
    payload = await service.anchor(
        scope,
        view.id,
        (
            await ExtractionService(spy, docs, views, extractors=EXTRACTORS, limits=ExtractionLimits()).load_payload(
                scope, view
            )
        )
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
    extraction = ExtractionService(spy, docs, views, extractors=EXTRACTORS, limits=ExtractionLimits())
    service = AnchorService(spy, views, provider_namespace="test")
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
    _svc, intake, spy, _store, _docs, views, _ = setup(tmp_path, owner, other)
    doc_id, version_id = await ingest(intake, owner, MD.encode(), name="a.md", media="text/markdown")
    extraction = ExtractionService(spy, _docs, views, extractors=EXTRACTORS, limits=ExtractionLimits())
    view = (await extraction.extract(owner, doc_id, version_id)).view
    service = AnchorService(spy, views, provider_namespace="test")
    anchor = await service.anchor(owner, view.id, "b00000", 0, 5)
    spy.reads.clear()
    result = await service.resolve(other, [anchor])
    assert result[0].reason is ResolutionReason.NOT_FOUND
    assert spy.reads == []
    foreign = anchor.model_copy(update={"source": anchor.source.model_copy(update={"provider_namespace": "elsewhere"})})

    class NeverLookup:
        async def get_view(self, *_args):
            raise AssertionError("foreign namespace looked up")

    foreign_result = await AnchorService(spy, NeverLookup(), provider_namespace="test").resolve(owner, [foreign])
    assert foreign_result[0].reason is ResolutionReason.FOREIGN_NAMESPACE


@sync
async def test_tampered_payload_and_view_limit(tmp_path):
    _svc, intake, spy, _store, docs, views, scope = setup(tmp_path)
    extraction = ExtractionService(spy, docs, views, extractors=EXTRACTORS, limits=ExtractionLimits())
    service = AnchorService(spy, views, provider_namespace="test")
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
    doc_id, version_id = await ingest(intake, scope, MD.encode(), name="a.md", media="text/markdown")
    extraction = ExtractionService(spy, docs, views, extractors=EXTRACTORS, limits=ExtractionLimits())
    view = (await extraction.extract(scope, doc_id, version_id)).view
    service = AnchorService(spy, views, provider_namespace="test")
    anchor = await service.anchor(scope, view.id, "b00000", 0, 5)
    unknown = anchor.model_copy(update={"source": anchor.source.model_copy(update={"extraction_view_id": "bad"})})
    spy.reads.clear()
    result = await service.resolve(scope, [unknown])
    assert result[0].reason is ResolutionReason.NOT_FOUND
    assert spy.reads == []
