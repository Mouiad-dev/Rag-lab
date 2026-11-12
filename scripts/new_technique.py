"""Scaffold an external technique plugin package for a catalogue ID that already exists.

Usage (from the repository root):
    uv run python scripts/new_technique.py C11 --kind chunker --fidelity simulation --out <dir>

The catalogue is opened read-only; adding IDs stays a human catalogue change.
Invalid input prints one line to stderr and exits with code 2.
"""

import argparse
import re
import sys
from pathlib import Path

from rag_foundry_techniques import TechniqueRegistry, load_catalogue
from rag_foundry_techniques.errors import CatalogueError

ROOT = Path(__file__).resolve().parents[1]
FIDELITIES = ("faithful", "adaptation", "simulation")
PACKAGE_RE = r"[a-z][a-z0-9_]{2,60}"

_METHODS = {
    "chunker": ("    def chunk(self, doc):", "chunk(None)"),
    "retriever": ("    async def retrieve(self, query):", "retrieve('q')"),
    "fusion": ("    def fuse(self, results):", "fuse([])"),
}


class Refusal(Exception):
    pass


def _builtin_ids() -> set[str]:
    from rag_foundry_chunking import register_builtin_plugins as register_chunking
    from rag_foundry_indexing import register_builtin_plugins as register_indexing

    catalogue = load_catalogue(ROOT / "config/techniques.json")
    registry = TechniqueRegistry(catalogue)
    register_chunking(registry)
    register_indexing(registry)
    return set(registry.registered_ids())


def _validate(args, catalogue) -> str:
    tid = args.technique_id
    if tid not in catalogue:
        raise Refusal(f"unknown technique ID {tid!r} (IDs are case-sensitive; adding IDs is a human catalogue change)")
    status = catalogue.get(tid).status
    if status in ("implemented", "validated"):
        raise Refusal(f"{tid} has status '{status}'; scaffolds are only for IDs that are not implemented yet")
    allowed = catalogue.kinds_for(tid)
    kind = args.kind
    if kind is None:
        if len(allowed) != 1:
            raise Refusal(f"--kind is required for {tid}; allowed kinds: {', '.join(allowed)}")
        kind = allowed[0]
    if kind not in allowed:
        raise Refusal(f"kind '{kind}' is not allowed for {tid}; allowed kinds: {', '.join(allowed)}")
    if args.package is not None and re.fullmatch(PACKAGE_RE, args.package) is None:
        raise Refusal(f"invalid package name {args.package!r}; must match {PACKAGE_RE}")
    out = Path(args.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise Refusal(f"output path {out} exists and is not an empty directory")
    if tid in _builtin_ids():
        raise Refusal(f"a built-in plugin already exists for {tid}; discovery would fail with DuplicatePluginError")
    return kind


def _render(tid: str, kind: str, fidelity: str, pkg: str, catalogue_path: Path) -> dict[str, str]:
    if kind in _METHODS:
        method_line, call = _METHODS[kind]
        method = f'\n{method_line}\n        raise TechniqueUnavailableError("{tid} scaffold: not implemented")\n'
    else:
        call = None
        method = (
            f"\n    # The '{kind}' kind protocol is not frozen yet: no method is generated.\n"
            f'    # Raise TechniqueUnavailableError("{tid} scaffold: not implemented") from your own entry point.\n'
        )
    extra_import = ", TechniqueUnavailableError" if kind in _METHODS else ""
    init = f'''"""Scaffold for technique {tid} ({kind}, {fidelity}). Not implemented."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest{extra_import}


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Plugin:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="{tid}",
        kind=PluginKind("{kind}"),
        version="0.1.0",
        fidelity=Fidelity("{fidelity}"),
        config_model=Config,
    )

    def __init__(self, config: Config) -> None:
        self.config = config
{method}'''
    if call is None:
        lines = _contract_only_test(pkg, catalogue_path)
    else:
        lines = [
            *(["import asyncio"] if kind == "retriever" else []),
            "import os",
            "from pathlib import Path",
            "",
            "import pytest",
            f"from {pkg} import Config, Plugin",
            "from rag_foundry_techniques import TechniqueUnavailableError, check_plugin_contract, load_catalogue",
            "",
            f"DEFAULT_CATALOGUE = {ascii(str(catalogue_path))}",
            "",
            "",
            "def _catalogue():",
            '    return load_catalogue(Path(os.environ.get("RAG_FOUNDRY_CATALOGUE", DEFAULT_CATALOGUE)))',
            "",
            "",
            "def test_contract():",
            "    check_plugin_contract(Plugin, _catalogue(), sample_config={})",
            "",
            "",
            "def test_scaffold_is_unavailable():",
            "    plugin = Plugin(Config())",
            "    with pytest.raises(TechniqueUnavailableError):",
            f"        asyncio.run(plugin.{call})" if kind == "retriever" else f"        plugin.{call}",
        ]
    test = "\n".join(lines) + "\n"
    pyproject = f'''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{pkg.replace("_", "-")}"
version = "0.1.0"
description = "Scaffold for RAG Foundry technique {tid}"
requires-python = ">=3.12"
dependencies = ["rag-foundry-techniques"]

[project.entry-points."rag_foundry.techniques"]
{tid} = "{pkg}:Plugin"

[tool.hatch.build.targets.wheel]
packages = ["src/{pkg}"]
'''
    readme = f"""# {pkg}

Scaffold for technique {tid} ({kind}, fidelity {fidelity}). It is not implemented yet.

- Follow `docs/templates/TECHNIQUE.md` in the RAG Foundry repository.
- Plugins are trusted code: installing this package and enabling plugin discovery runs it in the
  RAG Foundry process. See the trusted-code note in the plugin documentation before enabling it.
- Run the contract test: `PYTHONPATH=src python -m pytest -q tests/test_contract.py`
  (set `RAG_FOUNDRY_CATALOGUE` to override the catalogue path).
"""
    return {
        "pyproject.toml": pyproject,
        f"src/{pkg}/__init__.py": init,
        "tests/test_contract.py": test,
        "README.md": readme,
    }


def _contract_only_test(pkg: str, catalogue_path: Path) -> list[str]:
    return [
        "import os",
        "from pathlib import Path",
        "",
        f"from {pkg} import Plugin",
        "from rag_foundry_techniques import check_plugin_contract, load_catalogue",
        "",
        f"DEFAULT_CATALOGUE = {ascii(str(catalogue_path))}",
        "",
        "",
        "def test_contract():",
        '    path = Path(os.environ.get("RAG_FOUNDRY_CATALOGUE", DEFAULT_CATALOGUE))',
        "    check_plugin_contract(Plugin, load_catalogue(path), sample_config={})",
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Scaffold an external technique plugin package.")
    parser.add_argument("technique_id")
    parser.add_argument("--kind")
    parser.add_argument("--fidelity", required=True, choices=FIDELITIES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--package")
    parser.add_argument("--catalogue", default=str(ROOT / "config/techniques.json"))
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    catalogue_path = Path(args.catalogue).resolve()
    try:
        catalogue = load_catalogue(catalogue_path)
        kind = _validate(args, catalogue)
    except Refusal as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except CatalogueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    pkg = args.package or f"rag_foundry_plugin_{args.technique_id.lower()}"
    if re.fullmatch(PACKAGE_RE, pkg) is None:
        print(f"refused: invalid package name {pkg!r}; must match {PACKAGE_RE}", file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    for rel, content in _render(args.technique_id, kind, args.fidelity, pkg, catalogue_path).items():
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="ascii", newline="\n")
        print(f"created {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
