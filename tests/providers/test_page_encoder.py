import asyncio
import builtins
import struct
import types
import zlib

import pytest
from rag_foundry_domain.errors import ExtractionError
from rag_foundry_providers.colpali_pages import ColPaliPageEncoder, png_pixel_count
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.ports import PageImageEncoder

from tests.providers._fake_page_encoder import FakePageEncoder

SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def header_only_png(width: int, height: int) -> bytes:
    return SIG + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))


def run(coro):
    return asyncio.run(coro)


def test_fake_and_adapter_satisfy_protocol():
    assert isinstance(FakePageEncoder(), PageImageEncoder)
    assert isinstance(ColPaliPageEncoder(model_dir="does-not-exist"), PageImageEncoder)


def test_fake_shapes_and_determinism():
    fake = FakePageEncoder(dim=4, patches=3)
    png = header_only_png(10, 10)
    a = run(fake.encode_pages([png, png + b"x"]))
    b = run(fake.encode_pages([png, png + b"x"]))
    assert a.items == b.items and a.dim == 4
    assert [len(i.vectors) for i in a.items] == [3, 3] and a.items[0] != a.items[1]
    q = run(fake.encode_queries(["quarterly revenue", ""]))
    assert [len(i.vectors) for i in q.items] == [2, 0]


def test_missing_dir_raises_dependency_error(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_VISUAL_MODEL_DIR", raising=False)
    for encoder in (ColPaliPageEncoder(), ColPaliPageEncoder(model_dir=tmp_path / "nope")):
        with pytest.raises(AdapterDependencyError):
            run(encoder.encode_queries(["x"]))
        with pytest.raises(AdapterDependencyError):
            run(encoder.encode_pages([header_only_png(1, 1)]))


def test_missing_extra_raises_dependency_error(tmp_path, monkeypatch):
    for name in ("adapter_model.safetensors", "adapter_config.json", "modules.json"):
        (tmp_path / name).write_bytes(b"{}")
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("peft", "PIL"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(AdapterDependencyError):
        run(ColPaliPageEncoder(model_dir=tmp_path, base_dir=base).encode_queries(["x"]))


def test_pixel_limit_boundary():
    assert png_pixel_count(header_only_png(4_000_000, 1)) == 4_000_000
    assert png_pixel_count(header_only_png(2000, 2000)) == 4_000_000
    for bad in (header_only_png(4_000_001, 1), header_only_png(0, 5)):
        with pytest.raises(ExtractionError) as err:
            png_pixel_count(bad)
        assert err.value.code == "page_image_invalid"


def test_oversize_rejected_before_model_or_pillow(tmp_path):
    # An unconfigured adapter raises AdapterDependencyError if a page gets past validation.
    encoder = ColPaliPageEncoder(model_dir=tmp_path / "nope")
    with pytest.raises(ExtractionError) as err:
        run(encoder.encode_pages([header_only_png(4_000_001, 1)]))
    assert err.value.code == "page_image_invalid"
    with pytest.raises(AdapterDependencyError):  # 4_000_000 pixels passes the check and reaches model loading
        run(encoder.encode_pages([header_only_png(4_000_000, 1)]))


def test_pillow_global_unchanged(tmp_path):
    pil = pytest.importorskip("PIL.Image")
    before = pil.MAX_IMAGE_PIXELS
    encoder = ColPaliPageEncoder(model_dir=tmp_path / "nope")
    for png in (header_only_png(4_000_001, 1), header_only_png(10, 10)):
        with pytest.raises((ExtractionError, AdapterDependencyError)):
            run(encoder.encode_pages([png]))
    assert pil.MAX_IMAGE_PIXELS == before


@pytest.mark.parametrize("data", [b"", b"not a png at all, just text bytes here....", SIG, SIG + b"\x00" * 40])
def test_non_png_rejected(data):
    with pytest.raises(ExtractionError) as err:
        png_pixel_count(data)
    assert err.value.code == "page_image_invalid"
    with pytest.raises(ExtractionError):
        run(ColPaliPageEncoder(model_dir="nope").encode_pages([data]))


def _fake_model_dirs(tmp_path):
    for name in ("adapter_model.safetensors", "adapter_config.json", "modules.json"):
        (tmp_path / name).write_bytes(b"{}")
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"")
    return base


@pytest.mark.parametrize("where", ["adapter", "base"])
def test_symlink_in_model_dir_refused_before_load(tmp_path, monkeypatch, where):
    base = _fake_model_dirs(tmp_path)
    target = tmp_path if where == "adapter" else base
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "w.bin").write_bytes(b"x")
    try:
        (target / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted on this system")
    # The scan must refuse before anything is copied or loaded.
    monkeypatch.setattr(
        "rag_foundry_providers.colpali_pages.tempfile",
        types.SimpleNamespace(TemporaryDirectory=lambda *a, **k: pytest.fail("reached copy/load")),
    )
    encoder = ColPaliPageEncoder(model_dir=tmp_path, base_dir=base)
    with pytest.raises(AdapterDependencyError):
        run(encoder.encode_queries(["x"]))
    assert encoder._model is None


@pytest.mark.parametrize("where", ["adapter", "base"])
@pytest.mark.parametrize("name", ["w.bin", "sub/w.PT", "w.pth", "w.pkl"])
def test_pickle_style_weights_refused_before_load(tmp_path, monkeypatch, where, name):
    base = _fake_model_dirs(tmp_path)
    target = tmp_path if where == "adapter" else base
    (target / name).parent.mkdir(exist_ok=True)
    (target / name).write_bytes(b"x")
    # The scan must refuse before anything is copied or loaded.
    monkeypatch.setattr(
        "rag_foundry_providers.colpali_pages.tempfile",
        types.SimpleNamespace(TemporaryDirectory=lambda *a, **k: pytest.fail("reached copy/load")),
    )
    encoder = ColPaliPageEncoder(model_dir=tmp_path, base_dir=base)
    with pytest.raises(AdapterDependencyError):
        run(encoder.encode_queries(["x"]))
    assert encoder._model is None


def test_concurrent_loads_happen_once(tmp_path, monkeypatch):
    import threading
    import time

    encoder = ColPaliPageEncoder(model_dir=tmp_path / "nope")
    calls = []

    def slow_locked():
        if encoder._model is not None:
            return
        calls.append(1)
        time.sleep(0.05)
        encoder._model = object()

    monkeypatch.setattr(encoder, "_load_locked", slow_locked)
    threads = [threading.Thread(target=encoder._load) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1


def test_empty_input_uses_profile_dim():
    encoder = ColPaliPageEncoder(model_dir="nope")
    for result in (run(encoder.encode_pages([])), run(encoder.encode_queries([]))):
        assert result.items == () and result.dim == 128
