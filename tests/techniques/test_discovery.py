import subprocess
import sys
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from rag_foundry_techniques import (
    ENTRY_POINT_GROUP,
    DiscoveredPlugin,
    Fidelity,
    PluginDiscoveryError,
    PluginKind,
    PluginManifest,
    TechniqueRegistry,
    discover_plugins,
    discovery,
    load_catalogue,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "config" / "techniques.json"


class Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _plugin(tid, kind=PluginKind.CHUNKER, name="P"):
    manifest = PluginManifest(
        technique_id=tid, kind=kind, version="0.2.0", fidelity=Fidelity.SIMULATION, config_model=Cfg
    )
    return type(name, (), {"manifest": manifest})


class FakeDist:
    def __init__(self, name, version):
        self.name = name
        self.version = version


class FakeEP:
    def __init__(self, name, target, value="fake:Plugin", group=ENTRY_POINT_GROUP, dist=None):
        self.name = name
        self.value = value
        self.group = group
        self.dist = dist
        self._target = target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def _registry():
    return TechniqueRegistry(load_catalogue(CATALOGUE))


def test_success_report_fields():
    reg = _registry()
    dist = FakeDist("my-dist", "1.2.3")
    eps = [FakeEP("C01", _plugin("C01"), value="m:C01", dist=dist)]
    report = discover_plugins(reg, entry_points=eps)
    assert report == (
        DiscoveredPlugin(
            entry_point="C01",
            value="m:C01",
            distribution="my-dist",
            distribution_version="1.2.3",
            technique_id="C01",
            kind="chunker",
            plugin_version="0.2.0",
        ),
    )
    assert reg.registered_ids() == ("C01",)


def test_no_dist_gives_none():
    report = discover_plugins(_registry(), entry_points=[FakeEP("C01", _plugin("C01"))])
    assert report[0].distribution is None and report[0].distribution_version is None


def test_reversed_input_sorted_report():
    reg = _registry()
    eps = [FakeEP("C02", _plugin("C02")), FakeEP("C01", _plugin("C01"))]
    report = discover_plugins(reg, entry_points=eps)
    assert [r.entry_point for r in report] == ["C01", "C02"]


def _fail(eps, reg=None):
    reg = reg or _registry()
    with pytest.raises(PluginDiscoveryError) as info:
        discover_plugins(reg, entry_points=eps)
    return info.value


def test_unknown_id():
    err = _fail([FakeEP("X99", _plugin("X99"))])
    assert err.failures[0].error_type == "UnknownTechniqueError"


def test_wrong_kind():
    err = _fail([FakeEP("C01", _plugin("C01", PluginKind.RERANKER))])
    assert err.failures[0].error_type == "PluginKindError"


def test_name_id_mismatch():
    err = _fail([FakeEP("C02", _plugin("C01"))])
    assert "does not match" in err.failures[0].reason


def test_duplicate_with_builtin():
    reg = _registry()
    reg.register(_plugin("C05"))
    err = _fail([FakeEP("C05", _plugin("C05"))], reg)
    assert err.failures[0].error_type == "DuplicatePluginError"


def test_duplicate_between_entry_points():
    reg = _registry()
    eps = [
        FakeEP("C01", _plugin("C01"), value="b:P"),
        FakeEP("C01", _plugin("C01"), value="a:P"),
    ]
    err = _fail(eps, reg)
    assert [f.error_type for f in err.failures] == ["DuplicatePluginError"] * 2
    assert [f.value for f in err.failures] == ["a:P", "b:P"]
    assert "duplicate entry point for C01" in err.failures[0].reason
    assert reg.registered_ids() == ()


def test_real_entry_point_import_failure():
    ep = EntryPoint(name="C11", value="ragf_missing_module_xyz:Plugin", group=ENTRY_POINT_GROUP)
    err = _fail([ep])
    assert err.failures[0].reason.startswith("import failed: ModuleNotFoundError")


def test_real_entry_point_no_manifest():
    ep = EntryPoint(name="C01", value="rag_foundry_techniques.protocol:PluginKind", group=ENTRY_POINT_GROUP)
    err = _fail([ep])
    assert "no valid PluginManifest" in err.failures[0].reason


def test_wrong_group_and_non_class():
    err = _fail([FakeEP("C01", _plugin("C01"), group="other.group")])
    assert "group" in err.failures[0].reason
    err = _fail([FakeEP("C01", object())])
    assert "not a class" in err.failures[0].reason


class _RaisingMeta(type):
    @property
    def manifest(cls):
        raise RuntimeError("broken manifest lookup")


class _BadRepr:
    def __repr__(self):
        raise RuntimeError("repr must not be called")


def test_non_technique_error_from_check_becomes_typed_failure():
    reg = _registry()
    broken = _RaisingMeta("Broken", (), {})
    eps = [FakeEP("C01", _plugin("C01")), FakeEP("C02", broken)]
    err = _fail(eps, reg)
    assert [(f.entry_point, f.error_type) for f in err.failures] == [("C02", "RuntimeError")]
    assert err.failures[0].reason.startswith("check failed: RuntimeError")
    assert reg.registered_ids() == ()


def test_non_class_with_raising_repr():
    err = _fail([FakeEP("C01", _BadRepr())])
    assert "type _BadRepr is not a class" in err.failures[0].reason


def test_mixed_batch_reports_all_and_registers_none():
    reg = _registry()
    eps = [
        FakeEP("C01", _plugin("C01")),
        FakeEP("X99", _plugin("X99")),
        FakeEP("C02", _plugin("C02", PluginKind.RERANKER)),
    ]
    with pytest.raises(PluginDiscoveryError) as info:
        discover_plugins(reg, entry_points=eps)
    assert [f.entry_point for f in info.value.failures] == ["C02", "X99"]
    assert reg.registered_ids() == ()


def test_default_path_uses_group(monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return [FakeEP("C01", _plugin("C01"))]

    monkeypatch.setattr(discovery.metadata, "entry_points", fake)
    reg = _registry()
    discover_plugins(reg)
    assert calls == [{"group": "rag_foundry.techniques"}]
    assert reg.registered_ids() == ("C01",)


def test_import_does_not_call_entry_points():
    code = (
        "import importlib.metadata as m\n"
        "def boom(*a, **k):\n    raise SystemExit(3)\n"
        "m.entry_points = boom\n"
        "import rag_foundry_techniques\n"
        "print('ok')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0
    assert "ok" in res.stdout
