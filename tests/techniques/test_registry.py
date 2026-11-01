from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from rag_foundry_techniques import (
    DuplicatePluginError,
    Fidelity,
    PluginKind,
    PluginManifest,
    TechniqueError,
    TechniquePlugin,
    TechniqueRegistry,
    TechniqueUnavailableError,
    UnknownTechniqueError,
    load_catalogue,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "config" / "techniques.json"


class DummyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: int = 10


class LaxConfig(BaseModel):
    size: int = 10


def _manifest(tid="C01", config=DummyConfig):
    return PluginManifest(
        technique_id=tid,
        kind=PluginKind.CHUNKER,
        version="0.1.0",
        fidelity=Fidelity.SIMULATION,
        config_model=config,
    )


class DummyPlugin:
    manifest: ClassVar[PluginManifest] = _manifest()


def _registry():
    return TechniqueRegistry(load_catalogue(CATALOGUE))


def test_fresh_registry_every_entry_known_but_unavailable():
    reg = _registry()
    cat = load_catalogue(CATALOGUE)
    assert len(cat.ids()) == 72
    assert reg.registered_ids() == ()
    for tid in cat.ids():
        assert reg.entry(tid).id == tid
        with pytest.raises(TechniqueUnavailableError, match=tid) as info:
            reg.plugin(tid)
        assert reg.entry(tid).status in str(info.value)
        assert not isinstance(info.value, KeyError | LookupError)


def test_unknown_id_raises_everywhere():
    reg = _registry()
    with pytest.raises(UnknownTechniqueError):
        reg.plugin("X99")
    with pytest.raises(UnknownTechniqueError):
        reg.entry("X99")
    bad = type("Bad", (), {"manifest": _manifest("X99")})
    with pytest.raises(UnknownTechniqueError):
        reg.register(bad)


def test_register_and_duplicate():
    reg = _registry()
    reg.register(DummyPlugin)
    assert reg.plugin("C01") is DummyPlugin
    assert isinstance(DummyPlugin, type) and isinstance(DummyPlugin, TechniquePlugin)
    assert reg.registered_ids() == ("C01",)
    with pytest.raises(DuplicatePluginError) as info:
        reg.register(DummyPlugin)
    assert not isinstance(info.value, KeyError | LookupError)
    assert not isinstance(UnknownTechniqueError("x"), KeyError | LookupError)


def test_validate_config():
    reg = _registry()
    reg.register(DummyPlugin)
    assert reg.validate_config("C01", {"size": 5}).size == 5
    with pytest.raises(ValidationError):
        reg.validate_config("C01", {"size": 5, "surprise": 1})
    with pytest.raises(TechniqueUnavailableError):
        reg.validate_config("C02", {})


def test_lax_config_model_refused():
    plugin = type("Lax", (), {"manifest": _manifest("C01", LaxConfig)})
    with pytest.raises(TechniqueError):
        _registry().register(plugin)


def test_class_without_manifest_refused():
    with pytest.raises(TechniqueError):
        _registry().register(type("NoManifest", (), {}))


def test_manifest_version_pattern():
    with pytest.raises(ValidationError):
        PluginManifest(
            technique_id="C01",
            kind=PluginKind.CHUNKER,
            version="1.0",
            fidelity=Fidelity.FAITHFUL,
            config_model=DummyConfig,
        )


def test_plugin_kind_has_twelve_members():
    assert len(PluginKind) == 12
    assert {k.value for k in PluginKind} == {
        "chunker", "representation", "retriever", "fusion", "query_transform", "reranker",
        "context_builder", "workflow", "graph_builder", "knowledge_store", "cache", "evaluator",
    }  # fmt: skip
