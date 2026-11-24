import asyncio
import functools
import hashlib
import os
import unicodedata

import pytest
from _pdf_fixtures import build_pdf
from rag_foundry_domain import NotFoundError, ObjectStore
from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.errors import ExtractionError, ObjectStoreError
from rag_foundry_domain.extraction import ExtractionKey
from rag_foundry_domain.testing import InMemoryDocumentRepository, InMemoryExtractionRepository, make_scope
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
from rag_foundry_ingestion import IntakeService
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from rag_foundry_ingestion.normalize import CANONICAL_V1, SEARCH_AR_EN_V1, apply
from rag_foundry_ingestion.payload import ExtractionPayload
from rag_foundry_storage import LocalObjectStore
from test_intake import MemoryObjectStore, RecordingStore


def sync(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


class Stream:
    def __init__(self, *parts: bytes) -> None:
        self.parts = parts

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for p in self.parts:
            yield p


class SpyStore:
    """Delegates to a real store and records `open_read` calls."""

    def __init__(self, inner: LocalObjectStore) -> None:
        self.inner = inner
        self.reads: list[str] = []
        self.open_now = lambda: 0
        self.calls: list[str] = []

    def open_read(self, key):
        self._record("open_read")
        self.reads.append(key)
        return self.inner.open_read(key)

    async def stage(self, *args, **kwargs):
        self._record("stage")
        return await self.inner.stage(*args, **kwargs)

    async def commit(self, *args, **kwargs):
        self._record("commit")
        return await self.inner.commit(*args, **kwargs)

    async def discard(self, *args, **kwargs):
        self._record("discard")
        return await self.inner.discard(*args, **kwargs)

    def _record(self, name):
        assert self.open_now() == 0
        self.calls.append(name)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def setup(tmp_path, *scopes):
    scopes = scopes or (make_scope(),)
    store = LocalObjectStore(tmp_path)
    docs = InMemoryDocumentRepository(scopes)
    views = InMemoryExtractionRepository(docs)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    spy = SpyStore(store)
    intake = IntakeService(store, uow, max_bytes=100_000)
    svc = ExtractionService(spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    return svc, intake, spy, store, docs, views, scopes[0]


async def ingest(intake, scope, data, *, name, media):
    out = await intake.ingest(scope, Stream(data), media_type=media, filename=name)
    return out.document.id, out.version.id


# Source (decoded) offsets, counting code points:
#   0-10 "Hello World", 11-12 CRLF, 13-14 CRLF, 15 emoji (one code point), 16 space,
#   17 kaf, 18 kasra, 19 teh, 20 alef, 21 beh, 22 space, 23 meem, 24-25 tatweel, 26 noon, 27 space, 28-30 "end".
MD = "Hello World\r\n\r\n\U0001f600 \u0643\u0650\u062a\u0627\u0628 \u0645\u0640\u0640\u0646 end\r\n"

HTML = "<html><body><p>Caf&eacute; &amp; Tea</p><p>&#x645;&#x631;&#x62d;&#x628;&#x627;</p></body></html>"


def hit_to_source(payload: ExtractionPayload, needle: str) -> tuple[tuple[int, int], tuple[int, int], str]:
    """Find `needle` in the search text of the canonical text; return canonical range, source range, search text."""
    norm = apply(SEARCH_AR_EN_V1, payload.canonical_text)
    i = norm.text.index(needle)
    c = norm.map.to_original(i, i + len(needle))
    s = payload.source_map.to_original(c.start, c.end)
    return (c.start, c.end), (s.start, s.end), norm.text


@sync
async def test_done_criterion_markdown(tmp_path):
    svc, intake, _spy, _store, _docs, _views, scope = setup(tmp_path)
    doc_id, ver_id = await ingest(intake, scope, MD.encode("utf-8"), name="a.md", media="text/markdown")
    out = await svc.extract(scope, doc_id, ver_id)
    assert out.created
    p = await svc.load_payload(scope, out.view)
    assert p.canonical_text == "Hello World\n\n\U0001f600 \u0643\u0650\u062a\u0627\u0628 \u0645\u0640\u0640\u0646 end\n"
    assert p.source_map.o_len == len(MD)
    assert all(0 <= b.start < b.end <= len(p.canonical_text) for b in p.blocks)
    expected = {
        "hello": ((0, 5), (0, 5)),
        "world": ((6, 11), (6, 11)),
        "\U0001f600": ((13, 14), (15, 16)),
        "\u0643\u062a\u0627\u0628": ((15, 20), (17, 22)),
        "\u0645\u0646": ((21, 25), (23, 27)),
        "end": ((26, 29), (28, 31)),
    }
    for needle, (canon, src) in expected.items():
        got_canon, got_src, _text = hit_to_source(p, needle)
        assert (got_canon, got_src) == (canon, src), needle
        original_slice = MD[src[0] : src[1]]
        assert apply(SEARCH_AR_EN_V1, original_slice).text == needle


@sync
async def test_done_criterion_html(tmp_path):
    svc, intake, _spy, _store, _docs, _views, scope = setup(tmp_path)
    doc_id, ver_id = await ingest(intake, scope, HTML.encode("utf-8"), name="a.html", media="text/html")
    out = await svc.extract(scope, doc_id, ver_id)
    p = await svc.load_payload(scope, out.view)
    assert p.source_space == "decoded_text"
    assert p.source_map.o_len == len(HTML)
    _c, src, _t = hit_to_source(p, "tea")
    assert src == (HTML.index("Tea"), HTML.index("Tea") + 3)
    assert apply(SEARCH_AR_EN_V1, HTML[src[0] : src[1]]).text == "tea"
    _c, src, _t = hit_to_source(p, "&")
    assert HTML[src[0] : src[1]] == "&amp;"
    _c, src, _t = hit_to_source(p, "\u0645\u0631\u062d\u0628\u0627")
    assert src == (HTML.index("&#x645;"), HTML.index("&#x627;") + len("&#x627;"))
    assert p.canonical_text.count("\u0645\u0631\u062d\u0628\u0627") == 1


@sync
async def test_second_call_returns_same_view_without_reading(tmp_path):
    svc, intake, spy, _store, _docs, _views, scope = setup(tmp_path)
    doc_id, ver_id = await ingest(intake, scope, MD.encode("utf-8"), name="a.md", media="text/markdown")
    first = await svc.extract(scope, doc_id, ver_id)
    assert len(spy.reads) == 1
    spy.reads.clear()
    second = await svc.extract(scope, doc_id, ver_id)
    assert not second.created
    assert second.view == first.view
    assert spy.reads == []


@sync
async def test_payload_bytes_identical_across_runs(tmp_path):
    first = setup(tmp_path / "one")
    second = setup(tmp_path / "two")
    digests = []
    for svc, intake, _spy, _store, _docs, _views, scope in (first, second):
        doc_id, ver_id = await ingest(intake, scope, MD.encode("utf-8"), name="a.md", media="text/markdown")
        out = await svc.extract(scope, doc_id, ver_id)
        digests.append(out.view.payload_sha256)
        p = await svc.load_payload(scope, out.view)
        assert hashlib.sha256(p.dumps()).hexdigest() == out.view.payload_sha256
    assert digests[0] == digests[1]


def all_files(root):
    return [os.path.join(d, n) for d, _dirs, names in os.walk(root) for n in names]


def _object_path(root, key):
    return os.path.join(root, *key.split("/"))


@sync
async def test_tampered_or_truncated_object_is_integrity_mismatch(tmp_path):
    for mutate in (lambda b: b"X" + b[1:], lambda b: b[:-1], lambda b: b + b"!"):
        root = tmp_path / str(len(os.listdir(tmp_path)))
        svc, intake, _spy, store, docs, views, scope = setup(root)
        doc_id, ver_id = await ingest(intake, scope, MD.encode("utf-8"), name="a.md", media="text/markdown")
        version = await docs.get_version(scope, doc_id, ver_id)
        before = sorted(all_files(root))
        path = _object_path(root, version.object_key)
        os.chmod(path, 0o600)
        with open(path, "rb") as f:
            data = f.read()
        with open(path, "wb") as f:
            f.write(mutate(data))
        with pytest.raises(ExtractionError) as exc:
            await svc.extract(scope, doc_id, ver_id)
        assert exc.value.code == "integrity_mismatch"
        assert MD[:5] not in str(exc.value)
        extractor = EXTRACTORS[MediaType.TEXT_MARKDOWN]
        key = ExtractionKey(
            document_version_id=ver_id,
            extractor=extractor.name,
            extractor_version=extractor.version,
            profile_id=CANONICAL_V1.profile_id,
            unicode_version=unicodedata.unidata_version,
        )
        assert await views.find_view(scope, key) is None
        assert sorted(all_files(root)) == before  # no payload object was written


@sync
async def test_tampered_payload_is_integrity_mismatch(tmp_path):
    svc, intake, _spy, _store, _docs, _views, scope = setup(tmp_path)
    doc_id, ver_id = await ingest(intake, scope, MD.encode("utf-8"), name="a.md", media="text/markdown")
    view = (await svc.extract(scope, doc_id, ver_id)).view
    path = _object_path(tmp_path, view.payload_key)
    os.chmod(path, 0o600)
    with open(path, "ab") as f:
        f.write(b" ")
    with pytest.raises(ExtractionError) as exc:
        await svc.load_payload(scope, view)
    assert exc.value.code == "integrity_mismatch"


def test_payload_loads_rejects_inconsistent_payloads(tmp_path):
    import json

    svc_parts = asyncio.run(_one_payload(tmp_path))
    good = json.loads(svc_parts)
    ExtractionPayload.loads(svc_parts)

    def broken(**changes):
        return json.dumps({**good, **changes}, ensure_ascii=False).encode("utf-8")

    bad_block = [{**good["blocks"][0], "end": len(good["canonical_text"]) + 1}]
    for data in (
        broken(blocks=bad_block),
        broken(canonical_text=good["canonical_text"] + "x"),  # map no longer covers the text
        broken(schema_version=2),
        broken(extra=1),
        svc_parts.replace(b'"page_starts":null', b'"page_starts":NaN'),
        b"not json",
    ):
        with pytest.raises(ExtractionError) as exc:
            ExtractionPayload.loads(data)
        assert exc.value.code == "integrity_mismatch"


async def _one_payload(tmp_path) -> bytes:
    svc, intake, _spy, _store, _docs, _views, scope = setup(tmp_path)
    doc_id, ver_id = await ingest(intake, scope, MD.encode("utf-8"), name="a.md", media="text/markdown")
    view = (await svc.extract(scope, doc_id, ver_id)).view
    return (await svc.load_payload(scope, view)).dumps()


@sync
async def test_other_tenant_gets_not_found_and_nothing_is_read(tmp_path):
    owner, other = make_scope(), make_scope()
    svc, intake, spy, _store, _docs, _views, _ = setup(tmp_path, owner, other)
    doc_id, ver_id = await ingest(intake, owner, MD.encode("utf-8"), name="a.md", media="text/markdown")
    view = (await svc.extract(owner, doc_id, ver_id)).view
    spy.reads.clear()
    with pytest.raises(NotFoundError):
        await svc.extract(other, doc_id, ver_id)
    with pytest.raises(NotFoundError):
        await svc.load_payload(other, view)
    assert spy.reads == []


def test_local_store_satisfies_the_port(tmp_path):
    assert isinstance(LocalObjectStore(tmp_path), ObjectStore)


@sync
async def test_unsupported_media_type_without_extractor(tmp_path):
    store = LocalObjectStore(tmp_path)
    scope = make_scope()
    docs = InMemoryDocumentRepository([scope])
    views = InMemoryExtractionRepository(docs)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    intake = IntakeService(store, uow, max_bytes=1000)
    svc = ExtractionService(store, uow, extractors={}, limits=ExtractionLimits())
    doc_id, ver_id = await ingest(intake, scope, b"plain", name="a.txt", media="text/plain")
    with pytest.raises(ExtractionError) as exc:
        await svc.extract(scope, doc_id, ver_id)
    assert exc.value.code == "unsupported_extractor"


@sync
async def test_pdf_end_to_end(tmp_path):
    pytest.importorskip("pypdfium2")

    svc, intake, _spy, _store, _docs, _views, scope = setup(tmp_path)
    doc_id, ver_id = await ingest(
        intake, scope, build_pdf(["Hello page", "Second page"]), name="a.pdf", media="application/pdf"
    )
    out = await svc.extract(scope, doc_id, ver_id)
    p = await svc.load_payload(scope, out.view)
    assert p.source_space == "pdf_page_text"
    assert p.page_starts is not None and len(p.page_starts) == 2
    assert out.view.page_count == 2


class FailingCommitStore(SpyStore):
    def __init__(self, inner) -> None:
        super().__init__(inner)
        self.discarded = 0

    async def commit(self, staged, key):
        raise ObjectStoreError("boom")

    async def discard(self, staged):
        self.discarded += 1
        await self.inner.discard(staged)


@sync
async def test_commit_failure_discards_staged_object_and_surfaces_original_error(tmp_path):
    svc, intake, _spy, store, docs, views, scope = setup(tmp_path)
    doc_id, ver_id = await ingest(intake, scope, MD.encode("utf-8"), name="a.md", media="text/markdown")
    failing = FailingCommitStore(store)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    bad = ExtractionService(failing, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    with pytest.raises(ObjectStoreError) as exc:
        await bad.extract(scope, doc_id, ver_id)
    assert str(exc.value) == "boom"
    assert failing.discarded == 1
    assert len(all_files(tmp_path)) == 1  # only the original object; the staged file is gone
    assert await views.list_views(scope, ver_id) == []


@sync
async def test_concurrent_extractions_share_one_view_and_one_payload(tmp_path):
    svc, intake, _spy, _store, _docs, views, scope = setup(tmp_path)
    doc_id, ver_id = await ingest(intake, scope, MD.encode("utf-8"), name="a.md", media="text/markdown")
    a, b = await asyncio.gather(svc.extract(scope, doc_id, ver_id), svc.extract(scope, doc_id, ver_id))
    assert a.view.id == b.view.id
    assert sorted([a.created, b.created]) == [False, True]
    assert len(await views.list_views(scope, ver_id)) == 1
    assert len(all_files(tmp_path)) == 2  # original plus one payload object


@sync
async def test_invalid_stored_media_type_is_unsupported_extractor(tmp_path):
    svc, intake, _spy, store, docs, views, scope = setup(tmp_path)
    doc_id, ver_id = await ingest(intake, scope, MD.encode("utf-8"), name="a.md", media="text/markdown")
    good = await docs.get_version(scope, doc_id, ver_id)

    async def broken_get_version(_scope, _doc, _ver):
        return good.model_copy(update={"media_type": "text/x-unknown"})

    docs.get_version = broken_get_version
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    broken = ExtractionService(store, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    with pytest.raises(ExtractionError) as exc:
        await broken.extract(scope, doc_id, ver_id)
    assert exc.value.code == "unsupported_extractor"


@sync
async def test_extraction_object_store_calls_have_no_open_uow(tmp_path):
    scope = make_scope()
    docs = InMemoryDocumentRepository([scope])
    views = InMemoryExtractionRepository(docs)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    store = RecordingStore(MemoryObjectStore(), lambda: uow.open_now)
    spy = SpyStore(store)
    spy.open_now = lambda: uow.open_now
    intake = IntakeService(store, uow, max_bytes=100_000)
    extraction = ExtractionService(spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    doc_id, ver_id = await ingest(intake, scope, MD.encode("utf-8"), name="a.md", media="text/markdown")
    store.calls.clear()
    spy.calls.clear()
    outcome = await extraction.extract(scope, doc_id, ver_id)
    assert store.calls == ["open_read", "stage", "commit"]
    assert outcome.created


@sync
async def test_payload_write_failure_opens_no_write_uow_for_view(tmp_path):
    scope = make_scope()
    docs = InMemoryDocumentRepository([scope])
    views = InMemoryExtractionRepository(docs)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    recording_store = RecordingStore(MemoryObjectStore(), lambda: uow.open_now)
    spy = SpyStore(recording_store)
    spy.open_now = lambda: uow.open_now
    intake = IntakeService(recording_store, uow, max_bytes=100_000)
    doc_id, ver_id = await ingest(intake, scope, MD.encode("utf-8"), name="a.md", media="text/markdown")
    recording_store.calls.clear()

    async def fail_payload_commit(_staged, _key):
        spy._record("commit")
        raise ObjectStoreError("payload write failed")

    spy.commit = fail_payload_commit
    spy.open_now = lambda: uow.open_now
    extraction = ExtractionService(spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    opened_before = uow.opened
    committed_before = uow.committed
    with pytest.raises(ObjectStoreError, match="payload write failed"):
        await extraction.extract(scope, doc_id, ver_id)
    assert uow.opened - opened_before == 1
    assert uow.committed == committed_before
    assert spy.calls == ["open_read", "stage", "commit", "discard"]
    assert await views.list_views(scope, ver_id) == []
