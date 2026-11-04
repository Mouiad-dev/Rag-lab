"""Local filesystem, content-addressed object store (ADR-0009 D4, D6, D7).

Nothing client-supplied becomes a path: keys are rebuilt from validated IDs and a hex hash.
Errors carry fixed text only; no content, filenames or full paths are logged or raised.

Accepted limit: there is a time-of-check/time-of-use window between the symlink checks and
`os.replace`. An attacker with write access to the local object-store directory is out of scope.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path

from rag_foundry_domain import EmptyUploadError, ObjectStoreError, Scope, StagedObject, UploadTooLargeError

_UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
_ID_RE = re.compile(_UUID_RE)
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[0-9a-f]{32}")
_KEY_RE = re.compile(rf"t/({_UUID_RE})/p/({_UUID_RE})/sha256/([0-9a-f]{{2}})/([0-9a-f]{{64}})")
_STAGING = ".staging"


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except PermissionError:
        try:
            os.chmod(path, 0o600)
            path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            raise ObjectStoreError("could not remove file") from None
    except OSError:
        raise ObjectStoreError("could not remove file") from None


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _hash_file(path: Path, chunk_size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()


class LocalObjectStore:
    def __init__(self, root: Path, *, chunk_size: int = 1024 * 1024) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self._root = Path(root)
        self._chunk_size = chunk_size

    # -- keys and paths -------------------------------------------------

    def key_for(self, scope: Scope, sha256: str) -> str:
        if not _ID_RE.fullmatch(scope.tenant_id) or not _ID_RE.fullmatch(scope.project_id):
            raise ObjectStoreError("invalid scope")
        if not _HASH_RE.fullmatch(sha256):
            raise ObjectStoreError("invalid hash")
        return f"t/{scope.tenant_id}/p/{scope.project_id}/sha256/{sha256[:2]}/{sha256}"

    def _path_for(self, key: str) -> tuple[Path, str]:
        m = _KEY_RE.fullmatch(key) if isinstance(key, str) else None
        if m is None or m.group(4)[:2] != m.group(3):
            raise ObjectStoreError("invalid key")
        path = self._root.joinpath("t", m.group(1), "p", m.group(2), "sha256", m.group(3), m.group(4))
        return path, m.group(4)

    def _check_below_root(self, path: Path) -> None:
        """Refuse symlinked components below root (root itself may be a symlink) and escapes."""
        try:
            rel = path.relative_to(self._root)
        except ValueError:
            raise ObjectStoreError("invalid key") from None
        if ".." in rel.parts:
            raise ObjectStoreError("invalid key")
        current = self._root
        for part in rel.parts:
            current = current / part
            if _is_link(current):
                raise ObjectStoreError("symbolic links are not allowed")
        try:
            path.resolve().relative_to(self._root.resolve())
        except (ValueError, OSError):
            raise ObjectStoreError("invalid key") from None

    def _staging_dir(self) -> Path:
        d = self._root / _STAGING
        d.mkdir(parents=True, exist_ok=True)
        if _is_link(d):
            raise ObjectStoreError("symbolic links are not allowed")
        return d

    def _staged_path(self, staged: StagedObject) -> Path:
        if not _TOKEN_RE.fullmatch(staged.token):
            raise ObjectStoreError("invalid staged object")
        return self._root / _STAGING / staged.token

    # -- port -----------------------------------------------------------

    async def stage(self, chunks: AsyncIterable[bytes], *, max_bytes: int) -> StagedObject:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        token = uuid.uuid4().hex
        try:
            # Created and opened synchronously (two short calls, no await) so cancellation cannot
            # leave an orphan staging file or a leaked handle.
            path = self._staging_dir() / token
            f = open(path, "xb")
        except OSError:
            raise ObjectStoreError("could not create staging file") from None
        h = hashlib.sha256()
        size = 0
        try:
            try:
                async for chunk in chunks:
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        raise UploadTooLargeError()
                    h.update(chunk)
                    await asyncio.to_thread(f.write, chunk)
                if size == 0:
                    raise EmptyUploadError()
                await asyncio.to_thread(f.flush)
            finally:
                f.close()
        except BaseException as exc:
            try:
                _unlink(path)
            except ObjectStoreError:
                pass
            if isinstance(exc, OSError):
                raise ObjectStoreError("could not write staging file") from None
            raise
        return StagedObject(token=token, sha256=h.hexdigest(), size_bytes=size)

    async def commit(self, staged: StagedObject, key: str) -> None:
        """Publish under `key`; no-op (staged file discarded) if the key holds the same hash.

        On any error, including a hash mismatch, the staged file is kept: the caller must call `discard`.
        """
        await asyncio.to_thread(self._commit_sync, staged, key)

    def _commit_sync(self, staged: StagedObject, key: str) -> None:
        target, key_hash = self._path_for(key)
        src = self._staged_path(staged)
        if key_hash != staged.sha256:
            raise ObjectStoreError("key does not match content hash")
        if _is_link(self._root / _STAGING) or not src.is_file() or _is_link(src):
            raise ObjectStoreError("staged object not found")
        try:
            self._check_below_root(target)
            self._mkdirs_synced(target.parent)
            self._check_below_root(target)
            if target.exists():
                self._dedup(src, target, key_hash)
                return
            fd = os.open(src, os.O_RDWR)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            try:
                os.chmod(src, 0o444)
            except OSError:
                pass
            try:
                os.replace(src, target)
            except OSError:
                if target.exists():
                    self._dedup(src, target, key_hash)
                    return
                raise ObjectStoreError("could not publish object") from None
            _fsync_dir(target.parent)
        except OSError:
            raise ObjectStoreError("could not publish object") from None

    def _mkdirs_synced(self, directory: Path) -> None:
        """Create missing directories below root and fsync each new one and its parent."""
        self._root.mkdir(parents=True, exist_ok=True)
        current = self._root
        for part in directory.relative_to(self._root).parts:
            current = current / part
            if _is_link(current):
                raise ObjectStoreError("symbolic links are not allowed")
            if not current.exists():
                current.mkdir(exist_ok=True)
                _fsync_dir(current)
                _fsync_dir(current.parent)

    def _dedup(self, src: Path, target: Path, key_hash: str) -> None:
        if _is_link(target) or _hash_file(target, self._chunk_size) != key_hash:
            raise ObjectStoreError("key already holds different content")
        _unlink(src)

    async def discard(self, staged: StagedObject) -> None:
        path = self._staged_path(staged)
        await asyncio.to_thread(_unlink, path)

    async def open_read(self, key: str) -> AsyncIterator[bytes]:
        path, _ = self._path_for(key)
        await asyncio.to_thread(self._check_below_root, path)
        try:
            f = await asyncio.to_thread(open, path, "rb")
        except FileNotFoundError:
            raise ObjectStoreError("object not found") from None
        except OSError:
            raise ObjectStoreError("could not read object") from None
        try:
            while True:
                block = await asyncio.to_thread(f.read, self._chunk_size)
                if not block:
                    return
                yield block
        finally:
            f.close()
