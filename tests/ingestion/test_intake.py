import asyncio
import functools
import os

import pytest
from rag_foundry_domain import (
    InvalidFilenameError,
    InvalidTextEncodingError,
    MediaTypeMismatchError,
    NotFoundError,
    Scope,
    UnsupportedMediaTypeError,
    UploadTooLargeError,
    new_id,
)
from rag_foundry_domain.testing import InMemoryDocumentRepository, make_scope
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
    return IntakeService(store, repo, max_bytes=max_bytes), store, repo, scopes[0]


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
    svc2 = IntakeService(store, both, max_bytes=1000)
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
