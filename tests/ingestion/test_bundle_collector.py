import asyncio
from datetime import UTC, datetime

import pytest
from rag_foundry_chunking import ParentChildChunker, ParentChildConfig
from rag_foundry_contracts.resolve import ResolutionStatus
from rag_foundry_domain import NotFoundError
from rag_foundry_domain.index_testing import InMemoryIndexRepository, make_new_index_version
from rag_foundry_domain.testing import make_scope
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
from rag_foundry_indexing import IndexService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.bundle import (
    BundleAnchorCollector,
    BundleError,
    BundleExportService,
    LocalBundleCatalog,
    import_bundle,
    read_bundle_dir,
    write_bundle_dir,
)
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from rag_foundry_providers.fake import FakeEmbeddingModel, fake_embedding_profile
from test_anchor_rechunk import MARKDOWN
from test_extraction_service import ingest, setup

NS = "lab.local"
CLOCK = lambda: datetime(2026, 11, 8, 10, 0, tzinfo=UTC)  # noqa: E731
SECOND = "# Second guide\n\nThe pump sends clean water.\n\nCheck the pressure every morning.\n"


async def world(tmp_path, owner=None):
    """Two extracted views for the owner, one for a stranger, and a ready C18 index over both owner views."""
    owner, stranger = owner or make_scope(), make_scope()
    _svc, intake, store, _real, docs, views, _ = setup(tmp_path / "store", owner, stranger)
    extraction = ExtractionService(
        store,
        InMemoryUnitOfWorkFactory(documents=docs, extractions=views),
        extractors=EXTRACTORS,
        limits=ExtractionLimits(),
    )
    owned = []
    for i, text in enumerate((MARKDOWN, SECOND)):
        doc_id, version_id = await ingest(intake, owner, text.encode("utf-8"), name=f"{i}.md", media="text/markdown")
        owned.append((await extraction.extract(owner, doc_id, version_id)).view.id)
    doc_id, version_id = await ingest(
        intake, stranger, b"# Other tenant\n\nSecret paragraph.\n", name="x.md", media="text/markdown"
    )
    foreign = (await extraction.extract(stranger, doc_id, version_id)).view.id
    index = InMemoryIndexRepository(views)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views, indexes=index)
    anchors = AnchorService(store, uow, provider_namespace=NS)
    built = await IndexService(store, uow, anchors).build(
        owner,
        tuple(owned),
        ParentChildChunker(ParentChildConfig(parent_size=300, child_size=80)),
        FakeEmbeddingModel(fake_embedding_profile(dim=4)),
    )
    return owner, store, uow, anchors, index, owned, foreign, built


async def _export(store, uow, owner, owned, collected):
    service = BundleExportService(store, uow, provider_namespace=NS, clock=CLOCK)
    return await service.export(owner, owned, collected)


def test_collected_block_chunk_and_parent_anchors_survive_a_round_trip(tmp_path):
    async def run():
        owner, store, uow, anchors, index, owned, _foreign, built = await world(tmp_path)
        collected = await BundleAnchorCollector(store, uow, anchors).anchors_for_views(owner, owned, built.id)
        chunks = await index.list_chunks(owner, built.id, limit=500)
        assert chunks and all(c.parent_start_cp is not None for c in chunks)
        assert len({(c.parent_start_cp, c.parent_end_cp) for c in chunks}) >= 2
        assert len({a.anchor_id for a in collected}) == len(collected)

        files = await _export(store, uow, owner, owned, collected)
        write_bundle_dir(files, tmp_path / "bundle")
        imported = import_bundle(read_bundle_dir(tmp_path / "bundle"))
        consumer = make_scope()
        catalog = LocalBundleCatalog()
        catalog.add(consumer, imported)
        assert all(r.status is ResolutionStatus.EXACT for r in catalog.resolve(consumer, collected))

        # Every chunk start and every parent start is the start of some collected anchor.
        def absolute(anchor):
            blocks = {b.block_id: b for b in imported.views[anchor.source.extraction_view_id].blocks}
            return anchor.source.extraction_view_id, blocks[anchor.block_id].start + anchor.text_range.start

        starts = {absolute(a) for a in collected}
        assert {(c.extraction_view_id, c.start) for c in chunks} <= starts
        assert {(c.extraction_view_id, c.parent_start_cp) for c in chunks} <= starts

        again = await BundleAnchorCollector(store, uow, anchors).anchors_for_views(owner, owned, built.id)
        assert await _export(store, uow, owner, owned, again) == files

    asyncio.run(run())


def test_other_scope_view_is_not_found_before_any_payload_read(tmp_path):
    async def run():
        owner, store, uow, anchors, _index, owned, foreign, _built = await world(tmp_path)
        store.reads.clear()
        with pytest.raises(NotFoundError):
            await BundleAnchorCollector(store, uow, anchors).anchors_for_views(owner, [*owned, foreign])
        assert store.reads == []

    asyncio.run(run())


def test_index_must_be_ready(tmp_path):
    async def run():
        owner, store, uow, anchors, index, owned, _foreign, _built = await world(tmp_path)
        building = await index.create_index_version(owner, make_new_index_version(tuple(owned)))
        with pytest.raises(BundleError) as caught:
            await BundleAnchorCollector(store, uow, anchors).anchors_for_views(owner, owned, building.id)
        assert caught.value.code == "index_not_ready"

    asyncio.run(run())


def test_every_index_view_must_be_exported(tmp_path):
    async def run():
        owner, store, uow, anchors, _index, owned, _foreign, built = await world(tmp_path)
        store.reads.clear()
        with pytest.raises(BundleError) as caught:
            await BundleAnchorCollector(store, uow, anchors).anchors_for_views(owner, owned[:1], built.id)
        assert caught.value.code == "index_views_not_exported"
        assert store.reads == []

    asyncio.run(run())


def test_chunk_limit(tmp_path):
    async def run():
        owner, store, uow, anchors, _index, owned, _foreign, built = await world(tmp_path)
        with pytest.raises(BundleError) as caught:
            await BundleAnchorCollector(store, uow, anchors, max_chunks=1).anchors_for_views(owner, owned, built.id)
        assert caught.value.code == "too_large"

    asyncio.run(run())


def test_same_anchor_id_with_different_content_is_a_collision(tmp_path):
    class SameIdAnchors:
        """Gives every block anchor one fixed id, so two different blocks collide."""

        def __init__(self, inner):
            self.inner = inner

        async def anchor(self, *args):
            made = await self.inner.anchor(*args)
            return made.model_copy(update={"anchor_id": "anc_" + "0" * 32})

        async def anchors_for_span(self, *args):
            return await self.inner.anchors_for_span(*args)

    async def run():
        owner, store, uow, anchors, _index, owned, _foreign, _built = await world(tmp_path)
        with pytest.raises(BundleError) as caught:
            await BundleAnchorCollector(store, uow, SameIdAnchors(anchors)).anchors_for_views(owner, owned[:1])
        assert caught.value.code == "anchor_collision"

    asyncio.run(run())
