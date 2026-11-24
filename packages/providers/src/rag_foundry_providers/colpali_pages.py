"""Optional local page-image multivector adapter for vidore/colSmol-256M (R41, ColPali style).

Verified 2026-11-23 (PyPI metadata, local model files, installed packages):
- colpali-engine 0.3.18 (MIT) is NOT used: it requires ``torch<2.14``, ``torchvision`` and ``peft``, which
  conflicts with the locked ``torch 2.15.0+cpu``. Older releases pin torch even lower. No pin was changed.
- Owner-approved path (2026-11-23): the installed ``sentence-transformers 6.2.0`` ``MultiVectorEncoder``
  loads the colSmol sentence-transformers layout (Transformer, Dense 576->128, Normalize, MultiVectorMask).
  The Transformer module loads the LoRA adapter through ``peft`` (0.22.0, Apache-2.0, direct dependency of
  the ``[visual]`` extra; pulls ``accelerate`` and ``psutil``). Pillow 13.0.0 (MIT-CMU) decodes the PNGs.
- colSmol-256M is a PEFT LoRA adapter. Its ``adapter_config.json`` names
  ``vidore/ColSmolVLM-Instruct-256M-base`` (same commit as ``vidore/ColSmolVLM-Instruct-256M``). A name would
  be resolved over the network, so the adapter files are mirrored into a temporary overlay directory whose
  ``adapter_config.json`` points at the local base directory (``RAG_VISUAL_BASE_DIR``, default: the sibling
  directory ``ColSmolVLM-Instruct-256M``). Files under ``var/models`` are never modified; the overlay is
  deleted after loading.
- Weights: safetensors only (``use_safetensors=True``), ``trust_remote_code=False``, ``local_files_only=True``,
  CPU, float32, batch size 1. Nothing is downloaded.

PNG safety: width and height are read from the IHDR chunk with the stdlib. Non-PNG input or more than
4_000_000 pixels raises ``ExtractionError("page_image_invalid")`` before Pillow opens anything. This module
never reads or changes ``Image.MAX_IMAGE_PIXELS`` or any other Pillow global.
"""

import asyncio
import json
import os
import shutil
import struct
import tempfile
import threading
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from time import monotonic
from typing import Any

from rag_foundry_domain.errors import ExtractionError

from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.messages import Multivector, MultivectorResult
from rag_foundry_providers.ports import PageImageEncoder
from rag_foundry_providers.usage import Operation, TokenSource, Usage

MAX_PAGE_PIXELS = 4_000_000
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_EXTRA = "visual"
_FORBIDDEN_SUFFIXES = (".bin", ".pt", ".pth", ".pkl")
_OVERLAY_FILES = ("adapter_model.safetensors", "adapter_config.json")


def png_pixel_count(png: bytes) -> int:
    """Return width * height from the IHDR chunk. Raises ExtractionError("page_image_invalid") otherwise."""
    if not isinstance(png, bytes | bytearray | memoryview):
        raise ExtractionError("page_image_invalid", "page image must be bytes")
    head = bytes(png[:33])
    if len(head) < 33 or head[:8] != _PNG_SIGNATURE or head[12:16] != b"IHDR" or head[8:12] != b"\x00\x00\x00\x0d":
        raise ExtractionError("page_image_invalid", "page image is not a PNG")
    width, height = struct.unpack(">II", head[16:24])
    if width < 1 or height < 1 or width * height > MAX_PAGE_PIXELS:
        raise ExtractionError("page_image_invalid", "page image size is outside the allowed range")
    return width * height


class ColPaliPageEncoder(PageImageEncoder):
    def __init__(self, model_dir: str | Path | None = None, *, base_dir: str | Path | None = None) -> None:
        configured = model_dir or os.environ.get("RAG_VISUAL_MODEL_DIR")
        self._model_dir = Path(configured) if configured else None
        base = base_dir or os.environ.get("RAG_VISUAL_BASE_DIR")
        if base:
            self._base_dir: Path | None = Path(base)
        elif self._model_dir is not None:
            self._base_dir = self._model_dir.parent / "ColSmolVLM-Instruct-256M"
        else:
            self._base_dir = None
        self._model: Any = None
        self._lock = threading.Lock()
        self.revision: str | None = None
        self.profile = self._make_profile(None)

    @staticmethod
    def _make_profile(revision: str | None) -> ModelProfile:
        model = "vidore/colSmol-256M" if revision is None else f"vidore/colSmol-256M@{revision}"
        return ModelProfile(
            id="colsmol-256m-pages",
            provider="sentence-transformers",
            model=model,
            kind=ModelKind.EMBEDDING,
            capabilities=Capabilities(local=True, embedding_dim=128),
        )

    def _load(self) -> None:
        with self._lock:  # concurrent to_thread calls load once
            self._load_locked()

    def _load_locked(self) -> None:
        if self._model is not None:
            return
        adapter, base = self._model_dir, self._base_dir
        if (
            adapter is None
            or base is None
            or not all((adapter / name).is_file() for name in (*_OVERLAY_FILES, "modules.json"))
            or not (base / "model.safetensors").is_file()
        ):
            raise AdapterDependencyError(_EXTRA)  # model files missing or RAG_VISUAL_MODEL_DIR unset
        for directory in (adapter, base):  # safetensors only: refuse pickle-style weights and any symlink
            # rglob does not descend into symlinked folders but copytree would follow them, so refuse symlinks.
            entries = list(directory.rglob("*"))
            if any(p.is_symlink() for p in entries) or any(
                p.suffix.lower() in _FORBIDDEN_SUFFIXES for p in entries if p.is_file()
            ):
                raise AdapterDependencyError(_EXTRA)
        try:
            import peft  # noqa: F401  (the adapter needs it; fail early with the typed error)
            import PIL  # noqa: F401
            from safetensors import SafetensorError
            from sentence_transformers import MultiVectorEncoder
        except ImportError as exc:
            raise AdapterDependencyError(_EXTRA) from exc
        try:
            with tempfile.TemporaryDirectory(prefix="ragf-visual-") as tmp:
                overlay = Path(tmp)
                for item in adapter.iterdir():
                    if item.is_file() and item.name not in _OVERLAY_FILES:
                        shutil.copyfile(item, overlay / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, overlay / item.name)
                shutil.copyfile(adapter / "adapter_model.safetensors", overlay / "adapter_model.safetensors")
                config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
                config["base_model_name_or_path"] = str(base)
                (overlay / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
                model = MultiVectorEncoder(
                    str(overlay),
                    device="cpu",
                    local_files_only=True,
                    trust_remote_code=False,
                    model_kwargs={"use_safetensors": True, "dtype": "float32"},
                )
        except (OSError, ValueError, RuntimeError, KeyError, ImportError, SafetensorError) as exc:
            raise AdapterDependencyError(_EXTRA) from exc
        model.eval()
        self._model = model
        revision_file = adapter / "REVISION"
        if revision_file.is_file():
            self.revision = revision_file.read_text(encoding="ascii", errors="replace").strip() or None
            self.profile = self._make_profile(self.revision)

    async def encode_pages(self, pngs: Sequence[bytes]) -> MultivectorResult:
        data = [bytes(p) for p in pngs]
        for png in data:  # validate every page before any model work
            png_pixel_count(png)
        return await self._run(self._encode_pages, data)

    async def encode_queries(self, texts: Sequence[str]) -> MultivectorResult:
        return await self._run(self._encode_queries, list(texts))

    async def _run(self, fn: Any, items: list[Any]) -> MultivectorResult:
        start = monotonic()
        vectors, dim = await asyncio.to_thread(fn, items)
        elapsed = (monotonic() - start) * 1000
        return MultivectorResult(
            items=tuple(Multivector(vectors=v) for v in vectors),
            dim=dim,
            usage=Usage(
                profile_id=self.profile.id,
                provider=self.profile.provider,
                model=self.profile.model,
                operation=Operation.EMBED,
                input_tokens=sum(len(v) for v in vectors),
                output_tokens=0,
                token_source=TokenSource.ESTIMATED,
                latency_ms=elapsed,
                estimated_cost_usd=0,
            ),
        )

    def _empty_dim(self) -> int:
        return int(self.profile.capabilities.embedding_dim or 0)

    def _to_rows(self, tensors: Any) -> tuple[tuple[tuple[tuple[float, ...], ...], ...], int]:
        rows = tuple(tuple(tuple(float(x) for x in row) for row in t.tolist()) for t in tensors)
        return rows, (len(rows[0][0]) if rows and rows[0] else self._empty_dim())

    def _encode_pages(self, pngs: list[bytes]) -> tuple[tuple[tuple[tuple[float, ...], ...], ...], int]:
        if not pngs:
            return (), self._empty_dim()
        self._load()
        import torch
        from PIL import Image

        out: list[Any] = []
        for png in pngs:
            try:
                with Image.open(BytesIO(png)) as image:
                    image.load()
                    rgb = image.convert("RGB")
            except Exception:
                raise ExtractionError("page_image_invalid", "page image cannot be decoded") from None
            with torch.no_grad():
                out.append(self._model.encode_document([rgb], batch_size=1)[0])
        return self._to_rows(out)

    def _encode_queries(self, texts: list[str]) -> tuple[tuple[tuple[tuple[float, ...], ...], ...], int]:
        if not texts:
            return (), self._empty_dim()
        self._load()
        import torch

        out: list[Any] = []
        for text in texts:
            with torch.no_grad():
                out.append(self._model.encode_query([text], batch_size=1)[0])
        return self._to_rows(out)
