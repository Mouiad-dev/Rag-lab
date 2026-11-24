import importlib
import sys
from pathlib import Path

import pytest
from rag_foundry_techniques import (
    ChunkInput,
    TechniqueRegistry,
    TechniqueUnavailableError,
    TextBlock,
    check_chunker_run,
    check_plugin_contract,
    discover_plugins,
    load_catalogue,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "config" / "techniques.json"
SAMPLE_SRC = ROOT / "examples" / "plugins" / "rag_foundry_sample_plugin" / "src"


@pytest.fixture
def sample_env(tmp_path, monkeypatch):
    dist = tmp_path / "rag_foundry_sample_plugin-0.1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rag-foundry-sample-plugin\nVersion: 0.1.0\n", encoding="utf-8"
    )
    (dist / "entry_points.txt").write_text(
        "[rag_foundry.techniques]\nC11 = rag_foundry_sample_plugin:PageBlockChunker\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(SAMPLE_SRC))
    # no test may see a cached sample module; the module imported during the test is dropped at teardown
    monkeypatch.delitem(sys.modules, "rag_foundry_sample_plugin", raising=False)
    importlib.invalidate_caches()
    yield
    sys.modules.pop("rag_foundry_sample_plugin", None)


def _doc(pages=3):
    text = "".join(f"page{i}\n" for i in range(pages))
    blocks = tuple(
        TextBlock(block_id=f"b{i}", start=i * 6, end=i * 6 + 5, kind="page", page_index=i) for i in range(pages)
    )
    return ChunkInput(text=text, blocks=blocks)


def _discovered():
    registry = TechniqueRegistry(load_catalogue(CATALOGUE))
    report = discover_plugins(registry)
    return registry, report


def test_discover_and_run_sample(sample_env):
    registry, report = _discovered()
    c11 = [r for r in report if r.technique_id == "C11"]
    assert len(c11) == 1
    assert c11[0].distribution == "rag-foundry-sample-plugin"
    assert c11[0].distribution_version == "0.1.0"
    cls = registry.plugin("C11")
    assert len(check_chunker_run(cls, {"max_pages": 1}, _doc(3))) == 3
    assert len(check_chunker_run(cls, {"max_pages": 2}, _doc(3))) == 2


def test_sample_passes_contract_kit(sample_env):
    registry, _ = _discovered()
    check_plugin_contract(registry.plugin("C11"), load_catalogue(CATALOGUE), sample_config={"max_pages": 2})


def test_no_page_blocks_is_unavailable(sample_env):
    registry, _ = _discovered()
    cls = registry.plugin("C11")
    with pytest.raises(TechniqueUnavailableError):
        check_chunker_run(cls, {}, ChunkInput(text="no pages here"))


def test_catalogue_status_unchanged():
    assert load_catalogue(CATALOGUE).get("C11").status == "planned"
