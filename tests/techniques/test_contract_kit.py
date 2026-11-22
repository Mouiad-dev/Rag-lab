from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from rag_foundry_cache import register_builtin_plugins as register_cache
from rag_foundry_chunking import register_builtin_plugins as register_chunking
from rag_foundry_indexing import register_builtin_plugins as register_indexing
from rag_foundry_techniques import (
    ChunkInput,
    ChunkSpan,
    Fidelity,
    PluginContractError,
    PluginKind,
    PluginManifest,
    TechniqueRegistry,
    check_chunker_run,
    check_plugin_contract,
    load_catalogue,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = load_catalogue(ROOT / "config" / "techniques.json")


def _builtins():
    registry = TechniqueRegistry(CATALOGUE)
    register_chunking(registry)
    register_indexing(registry)
    register_cache(registry)
    return {tid: registry.plugin(tid) for tid in registry.registered_ids()}


BUILTINS = _builtins()


@pytest.mark.parametrize("tid", sorted(BUILTINS))
def test_builtin_passes_contract(tid):
    manifest = check_plugin_contract(BUILTINS[tid], CATALOGUE)
    assert manifest.technique_id == tid


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Lax(BaseModel):
    pass


def _plugin(tid="C01", kind=PluginKind.CHUNKER, cfg=Strict, chunk=None):
    manifest = PluginManifest(
        technique_id=tid, kind=kind, version="0.1.0", fidelity=Fidelity.SIMULATION, config_model=cfg
    )
    ns = {"manifest": manifest, "__init__": lambda self, config: None}
    if chunk is not None:
        ns["chunk"] = chunk
    return type("P", (), ns)


def test_good_plugin_with_sample_config():
    check_plugin_contract(_plugin(), CATALOGUE, sample_config={})


def test_lax_config_rejected():
    with pytest.raises(PluginContractError):
        check_plugin_contract(_plugin(cfg=Lax), CATALOGUE)


def test_wrong_kind_rejected():
    with pytest.raises(PluginContractError):
        check_plugin_contract(_plugin(kind=PluginKind.RERANKER), CATALOGUE)


def test_unknown_id_rejected():
    with pytest.raises(PluginContractError):
        check_plugin_contract(_plugin(tid="X99"), CATALOGUE)


class NoJsonSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fn: Callable[[], int] = lambda: 1


# The registry-behaviour checks (fresh lookup, duplicate registration) and the unknown-key
# check depend on core registry and pydantic behaviour; a fake plugin cannot trigger them.


def test_missing_manifest_rejected():
    with pytest.raises(PluginContractError, match="no valid PluginManifest"):
        check_plugin_contract(type("P", (), {}), CATALOGUE)


def test_config_without_json_schema_rejected():
    with pytest.raises(PluginContractError, match="model_json_schema"):
        check_plugin_contract(_plugin(cfg=NoJsonSchema), CATALOGUE)


def test_invalid_sample_config_rejected():
    with pytest.raises(PluginContractError, match="sample_config does not validate"):
        check_plugin_contract(_plugin(), CATALOGUE, sample_config={"bad": 1})


def test_chunker_run_rejects_other_kinds_and_non_spans():
    doc = ChunkInput(text="abcdef")
    with pytest.raises(PluginContractError, match="not 'chunker'"):
        check_chunker_run(_plugin(kind=PluginKind.RERANKER), {}, doc)
    with pytest.raises(PluginContractError, match="not a ChunkSpan"):
        check_chunker_run(_plugin(chunk=lambda self, d: ((0, 3),)), {}, doc)


def test_bad_chunker_span_rejected():
    doc = ChunkInput(text="abcdef")
    out_of_bounds = _plugin(chunk=lambda self, d: (ChunkSpan(start=0, end=99, split_level=0),))
    with pytest.raises(PluginContractError):
        check_chunker_run(out_of_bounds, {}, doc)
    descending = _plugin(
        chunk=lambda self, d: (ChunkSpan(start=3, end=5, split_level=0), ChunkSpan(start=0, end=2, split_level=0))
    )
    with pytest.raises(PluginContractError):
        check_chunker_run(descending, {}, doc)
    ok = _plugin(chunk=lambda self, d: (ChunkSpan(start=0, end=3, split_level=0),))
    assert len(check_chunker_run(ok, {}, doc)) == 1
