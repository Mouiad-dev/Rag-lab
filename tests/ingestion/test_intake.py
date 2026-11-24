import asyncio
import functools
import hashlib
import os
import uuid

import pytest
from rag_foundry_domain import (
    InvalidFilenameError,
    InvalidTextEncodingError,
    MediaTypeMismatchError,
    NotFoundError,
    Scope,
    StagedObject,
    UnsupportedMediaTypeError,
    UploadTooLargeError,
    new_id,
)
from rag_foundry_domain.testing import InMemoryDocumentRepository, make_scope
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
from rag_foundry_ingestion import IntakeService
from rag_foundry_storage import LocalObjectStore


def sync(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


class Stream:
    def __init__(self, *parts: bytes) -> None:
        self.parts = parts
        self.started = False

    def __aiter__(self):
        self.started = True
        return self._gen()

    async def _gen(self):
        for p in self.parts:
            yield p


def files(root, include_staging=True):
    out = []
    for dirpath, dirnames, names in os.walk(root):
        if not include_staging and ".staging" in dirnames:
            dirnames.remove(".staging")
        out += [os.path.join(dirpath, n) for n in names]
    return out


def setup(tmp_path, *scopes, max_bytes=1000):
    scopes = scopes or (make_scope(),)
    store = LocalObjectStore(tmp_path)
    repo = InMemoryDocumentRepository(scopes)
    return IntakeService(store, InMemoryUnitOfWorkFactory(documents=repo), max_bytes=max_bytes), store, repo, scopes[0]


class RecordingStore:
    def __init__(self, inner, open_now):
        self.inner = inner
        self.open_now = open_now
        self.calls = []

    def key_for(self, *args):
        return self.inner.key_for(*args)

    def open_read(self, *args):
        self._record("open_read")
        return self.inner.open_read(*args)

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


class MemoryObjectStore:
    def __init__(self):
        self.staged = {}
        self.objects = {}

    def key_for(self, scope, sha256):
        return f"{scope.tenant_id}/{scope.project_id}/{sha256}"

    async def stage(self, chunks, *, max_bytes):
        body = bytearray()
        async for chunk in chunks:
            body.extend(chunk)
            if len(body) > max_bytes:
                raise RuntimeError("too large")
        digest = hashlib.sha256(body).hexdigest()
        obj = StagedObject(token=uuid.uuid4().hex, sha256=digest, size_bytes=len(body))
        self.staged[obj.token] = bytes(body)
        return obj

    async def commit(self, staged, key):
        self.objects[key] = self.staged.pop(staged.token)

    async def discard(self, staged):
        self.staged.pop(staged.token, None)

    def open_read(self, key):
        async def chunks():
            yield self.objects[key]

        return chunks()


A = b"alpha text"
A2 = b"alpha tex!"


async def up(svc, scope, data, *, name="a.txt", media="text/plain", document_id=None):
    return await svc.ingest(scope, Stream(data), media_type=media, filename=name, document_id=document_id)


@sync
async def test_done_criterion_dedup_and_versioning(tmp_path):
    svc, store, repo, scope = setup(tmp_path)
    first = await up(svc, scope, A)
    assert first.created_document and first.created_version and first.version.version_number == 1
    again = await up(svc, scope, A)
    assert again.deduplicated and not again.created_document
    assert (again.document.id, again.version.id) == (first.document.id, first.version.id)
    assert len(files(tmp_path, include_staging=False)) == 1
    v2 = await up(svc, scope, A2, document_id=first.document.id)
    assert v2.created_version and v2.version.version_number == 2
    assert v2.version.sha256 != first.version.sha256 and v2.version.object_key != first.version.object_key
    assert len(files(tmp_path, include_staging=False)) == 2
    assert await repo.get_version(scope, first.document.id, first.version.id) == first.version
    for key, expected in ((first.version.object_key, A), (v2.version.object_key, A2)):
        assert b"".join([c async for c in store.open_read(key)]) == expected
    back = await up(svc, scope, A, document_id=first.document.id)
    assert back.deduplicated and back.version.id == first.version.id
    assert files(tmp_path / ".staging") == []


@sync
async def test_same_bytes_other_project_is_separate(tmp_path):
    s1, s2 = make_scope(), make_scope()
    svc, _, _, _ = setup(tmp_path, s1, s2)
    a = await up(svc, s1, A)
    b = await up(svc, s2, A)
    assert b.created_document and b.document.id != a.document.id
    assert a.version.object_key != b.version.object_key


@sync
async def test_unknown_project_and_foreign_document_never_read_stream(tmp_path):
    svc, store, repo, scope = setup(tmp_path)
    other = Scope(tenant_id=new_id(), project_id=new_id())
    stream = Stream(A)
    with pytest.raises(NotFoundError):
        await svc.ingest(other, stream, media_type="text/plain", filename="a.txt")
    assert not stream.started
    both = InMemoryDocumentRepository([scope, other])
    svc2 = IntakeService(store, InMemoryUnitOfWorkFactory(documents=both), max_bytes=1000)
    doc = await up(svc2, other, A)
    for doc_id in (doc.document.id, new_id()):
        stream = Stream(A2)
        with pytest.raises(NotFoundError):
            await svc2.ingest(scope, stream, media_type="text/plain", filename="a.txt", document_id=doc_id)
        assert not stream.started
    assert len(files(tmp_path, include_staging=False)) == 1  # only the other tenant's object
    assert files(tmp_path / ".staging") == []


@sync
@pytest.mark.parametrize(
    ("data", "kwargs", "error"),
    [
        (A, {"media": "image/png"}, UnsupportedMediaTypeError),
        (A, {"media": "text/plain; charset=latin-1"}, UnsupportedMediaTypeError),
        (A, {"name": "a/b.txt"}, InvalidFilenameError),
        (A, {"name": "a.md"}, MediaTypeMismatchError),
        (b"ab\xffcd", {}, InvalidTextEncodingError),
        (b"ab\xd9", {}, InvalidTextEncodingError),  # truncated: fails in finish(), after staging
        (b"x" * 1001, {}, UploadTooLargeError),
        (b"not a pdf", {"name": "a.pdf", "media": "application/pdf"}, MediaTypeMismatchError),
        (b"%PD", {"name": "a.pdf", "media": "application/pdf"}, MediaTypeMismatchError),  # short: fails in finish()
    ],
)
async def test_rejected_uploads_leave_no_files(tmp_path, data, kwargs, error):
    svc, _, _, scope = setup(tmp_path)
    with pytest.raises(error):
        await up(svc, scope, data, **kwargs)
    assert files(tmp_path, include_staging=False) == []
    if (tmp_path / ".staging").exists():
        assert files(tmp_path / ".staging") == []


@sync
async def test_pre_staging_rejection_does_not_read_stream(tmp_path):
    svc, _, _, scope = setup(tmp_path)
    stream = Stream(A)
    with pytest.raises(UnsupportedMediaTypeError):
        await svc.ingest(scope, stream, media_type="image/png", filename="a.txt")
    assert not stream.started


@sync
async def test_commit_failure_discards_staged(tmp_path):
    svc, store, _, scope = setup(tmp_path)

    async def bad_commit(staged, key):
        raise RuntimeError("boom")

    store.commit = bad_commit
    with pytest.raises(RuntimeError):
        await up(svc, scope, A)
    assert files(tmp_path) == []


@sync
async def test_pdf_and_arabic_filename_stored(tmp_path):
    svc, _, _, scope = setup(tmp_path)
    name = "تقرير report.PDF"
    out = await up(svc, scope, b"%PDF-1.4 x", name=name, media="application/pdf")
    assert out.version.text_encoding is None
    assert out.document.display_name == name


@sync
async def test_ingest_stream_and_object_store_calls_have_no_open_uow(tmp_path):
    scope = make_scope()
    repo = InMemoryDocumentRepository([scope])
    uow = InMemoryUnitOfWorkFactory(documents=repo)
    store = RecordingStore(MemoryObjectStore(), lambda: uow.open_now)
    service = IntakeService(store, uow, max_bytes=1000)

    async def chunks():
        assert uow.open_now == 0
        yield A

    outcome = await service.ingest(scope, chunks(), media_type="text/plain", filename="a.txt")
    assert store.calls == ["stage", "commit"]
    assert uow.committed == 1
    assert b"".join([part async for part in store.open_read(outcome.version.object_key)]) == A
    assert store.calls[-1] == "open_read"
    with pytest.raises(InvalidTextEncodingError):
        await service.ingest(scope, Stream(b"ab\xd9"), media_type="text/plain", filename="bad.txt")
    assert store.calls[-2:] == ["stage", "discard"]


@sync
async def test_unknown_project_opens_no_write_uow(tmp_path):
    scope = make_scope()
    factory = InMemoryUnitOfWorkFactory(documents=InMemoryDocumentRepository([scope]))
    service = IntakeService(RecordingStore(MemoryObjectStore(), lambda: factory.open_now), factory, max_bytes=1000)
    with pytest.raises(NotFoundError):
        await service.ingest(
            Scope(tenant_id=new_id(), project_id=new_id()), Stream(A), media_type="text/plain", filename="a.txt"
        )
    assert factory.opened == 1
    assert factory.committed == 0
    assert factory.rolled_back == 0


@sync
async def test_successful_ingest_commits_one_write_uow(tmp_path):
    scope = make_scope()
    factory = InMemoryUnitOfWorkFactory(documents=InMemoryDocumentRepository([scope]))
    service = IntakeService(RecordingStore(MemoryObjectStore(), lambda: factory.open_now), factory, max_bytes=1000)
    await service.ingest(scope, Stream(A), media_type="text/plain", filename="a.txt")
    assert factory.opened == 2
    assert factory.committed == 1
