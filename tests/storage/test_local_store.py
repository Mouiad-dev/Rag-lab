import asyncio
import functools
import hashlib
import os

import pytest
from rag_foundry_domain import (
    LOCAL_DEV_SCOPE,
    EmptyUploadError,
    ObjectStore,
    ObjectStoreError,
    StagedObject,
    UploadTooLargeError,
)
from rag_foundry_storage import LocalObjectStore


def sync(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


DATA = b"hello object store" * 100
HEX = hashlib.sha256(DATA).hexdigest()


async def agen(*parts: bytes):
    for p in parts:
        yield p


def files_under(root, skip_staging=False):
    out = []
    for dirpath, dirnames, names in os.walk(root):
        if skip_staging and ".staging" in dirnames:
            dirnames.remove(".staging")
        out += [os.path.join(dirpath, n) for n in names]
    return out


def staging_files(root):
    d = root / ".staging"
    return list(d.iterdir()) if d.exists() else []


@pytest.fixture
def store(tmp_path):
    return LocalObjectStore(tmp_path / "objects", chunk_size=64)


def test_is_object_store(store):
    assert isinstance(store, ObjectStore)


def test_key_format(store):
    s = LOCAL_DEV_SCOPE
    assert store.key_for(s, HEX) == f"t/{s.tenant_id}/p/{s.project_id}/sha256/{HEX[:2]}/{HEX}"


@pytest.mark.parametrize("bad", ["", "abc", "A" * 64, "g" * 64, "../" + "a" * 61, "a" * 63, "a" * 65])
def test_key_for_bad_hash(store, bad):
    with pytest.raises(ObjectStoreError):
        store.key_for(LOCAL_DEV_SCOPE, bad)


def test_key_for_bad_ids(store):
    bad = LOCAL_DEV_SCOPE.model_construct(tenant_id="../x", project_id=LOCAL_DEV_SCOPE.project_id)
    with pytest.raises(ObjectStoreError):
        store.key_for(bad, HEX)


@sync
async def test_stage_hash_and_size(store, tmp_path):
    staged = await store.stage(agen(DATA[:50], b"", DATA[50:]), max_bytes=len(DATA))
    assert staged.sha256 == HEX
    assert staged.size_bytes == len(DATA)
    assert len(staging_files(tmp_path / "objects")) == 1


@sync
async def test_max_bytes_boundary(store, tmp_path):
    await store.stage(agen(b"x" * 10), max_bytes=10)
    before = len(staging_files(tmp_path / "objects"))
    with pytest.raises(UploadTooLargeError):
        await store.stage(agen(b"x" * 5, b"x" * 6), max_bytes=10)
    assert len(staging_files(tmp_path / "objects")) == before


@sync
async def test_too_large_stops_reading(store):
    consumed = 0

    async def gen():
        nonlocal consumed
        for _ in range(100):
            consumed += 1
            yield b"x" * 10

    with pytest.raises(UploadTooLargeError):
        await store.stage(gen(), max_bytes=25)
    assert consumed == 3


@sync
async def test_empty_upload(store, tmp_path):
    with pytest.raises(EmptyUploadError):
        await store.stage(agen(), max_bytes=10)
    with pytest.raises(EmptyUploadError):
        await store.stage(agen(b""), max_bytes=10)
    assert staging_files(tmp_path / "objects") == []


@sync
async def test_commit_and_read(store, tmp_path):
    key = store.key_for(LOCAL_DEV_SCOPE, HEX)
    staged = await store.stage(agen(DATA), max_bytes=10_000)
    await store.commit(staged, key)
    assert staging_files(tmp_path / "objects") == []
    got = b"".join([c async for c in store.open_read(key)])
    assert got == DATA
    target = tmp_path / "objects" / key
    assert not os.access(target, os.W_OK)


@sync
async def test_same_bytes_twice_one_file(store, tmp_path):
    key = store.key_for(LOCAL_DEV_SCOPE, HEX)
    for _ in range(2):
        await store.commit(await store.stage(agen(DATA), max_bytes=10_000), key)
    root = tmp_path / "objects"
    assert len(files_under(root, skip_staging=True)) == 1
    assert staging_files(root) == []


@sync
async def test_concurrent_commits(store, tmp_path):
    key = store.key_for(LOCAL_DEV_SCOPE, HEX)
    a = await store.stage(agen(DATA), max_bytes=10_000)
    b = await store.stage(agen(DATA), max_bytes=10_000)
    await asyncio.gather(store.commit(a, key), store.commit(b, key))
    root = tmp_path / "objects"
    found = files_under(root, skip_staging=True)
    assert len(found) == 1
    assert hashlib.sha256(open(found[0], "rb").read()).hexdigest() == HEX
    assert staging_files(root) == []


@sync
async def test_different_bytes_at_existing_key(store, tmp_path):
    key = store.key_for(LOCAL_DEV_SCOPE, HEX)
    await store.commit(await store.stage(agen(DATA), max_bytes=10_000), key)
    target = tmp_path / "objects" / key
    os.chmod(target, 0o644)
    target.write_bytes(b"corrupted")
    again = await store.stage(agen(DATA), max_bytes=10_000)
    with pytest.raises(ObjectStoreError):
        await store.commit(again, key)


@sync
async def test_key_hash_mismatch(store):
    staged = await store.stage(agen(b"abc"), max_bytes=10)
    with pytest.raises(ObjectStoreError):
        await store.commit(staged, store.key_for(LOCAL_DEV_SCOPE, HEX))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda k: "../" + k,
        lambda k: "/" + k,
        lambda k: k.replace("/", "\\"),
        lambda k: k.replace("/sha256/", "/../sha256/"),
        lambda k: k[:-1] + "z",
        lambda k: k.replace("t/", "t/x", 1),
        lambda k: k + "/",
        lambda k: k + "\n",
        lambda k: "",
    ],
)
@sync
async def test_bad_keys_rejected(store, mutate):
    key = store.key_for(LOCAL_DEV_SCOPE, HEX)
    staged = await store.stage(agen(DATA), max_bytes=10_000)
    with pytest.raises(ObjectStoreError):
        await store.commit(staged, mutate(key))
    with pytest.raises(ObjectStoreError):
        async for _ in store.open_read(mutate(key)):
            pass


@sync
async def test_bad_token_rejected(store):
    bad = StagedObject(token="../../etc", sha256=HEX, size_bytes=1)
    with pytest.raises(ObjectStoreError):
        await store.discard(bad)
    with pytest.raises(ObjectStoreError):
        await store.commit(bad, store.key_for(LOCAL_DEV_SCOPE, HEX))


@sync
async def test_discard_idempotent(store, tmp_path):
    staged = await store.stage(agen(DATA), max_bytes=10_000)
    await store.discard(staged)
    await store.discard(staged)
    assert staging_files(tmp_path / "objects") == []


def _symlink(link, target):
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not permitted on this system")


@sync
async def test_symlinked_dir_below_root_rejected(tmp_path):
    root = tmp_path / "objects"
    store = LocalObjectStore(root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    root.mkdir()
    _symlink(root / "t", elsewhere)
    staged = await store.stage(agen(DATA), max_bytes=10_000)
    with pytest.raises(ObjectStoreError):
        await store.commit(staged, store.key_for(LOCAL_DEV_SCOPE, HEX))
    assert list(elsewhere.iterdir()) == []


@sync
async def test_symlinked_root_accepted(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    _symlink(link, real)
    store = LocalObjectStore(link)
    key = store.key_for(LOCAL_DEV_SCOPE, HEX)
    await store.commit(await store.stage(agen(DATA), max_bytes=10_000), key)
    assert (real / key).is_file()


@sync
async def test_cancellation_mid_stage(store, tmp_path):
    started = asyncio.Event()

    async def slow():
        yield b"first"
        started.set()
        await asyncio.sleep(60)
        yield b"never"

    task = asyncio.create_task(store.stage(slow(), max_bytes=1000))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert staging_files(tmp_path / "objects") == []


@sync
async def test_cancellation_before_first_chunk(store, tmp_path):
    started = asyncio.Event()

    async def silent():
        started.set()
        await asyncio.sleep(60)
        yield b"never"

    task = asyncio.create_task(store.stage(silent(), max_bytes=1000))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert staging_files(tmp_path / "objects") == []
