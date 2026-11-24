"""Static guard: vendor SDKs only inside the providers adapters package; no LangChain in providers.

Stdlib only. AST-based import scan plus a manifest scan (ADR-0001).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVIDERS_ROOT = REPO_ROOT / "packages" / "providers"
ADAPTERS_DIR = PROVIDERS_ROOT / "src" / "rag_foundry_providers" / "adapters"
PROVIDERS_MANIFEST = PROVIDERS_ROOT / "pyproject.toml"

FORBIDDEN_MODULES = (
    "anthropic",
    "openai",
    "litellm",
    "ollama",
    "cohere",
    "mistralai",
    "voyageai",
    "google.genai",
    "google.generativeai",
)
FORBIDDEN_DISTRIBUTIONS = (
    "anthropic",
    "openai",
    "litellm",
    "ollama",
    "cohere",
    "mistralai",
    "voyageai",
    "google-genai",
    "google-generativeai",
)


def _imported_names(source: str) -> list[str]:
    """Dotted module names imported by `source`; `from a import b` also yields `a.b`."""
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
            names += [f"{node.module}.{alias.name}" for alias in node.names]
    return names


def _matches(name: str, forbidden: str) -> bool:
    return name == forbidden or name.startswith(forbidden + ".")


def imports_vendor_sdk(source: str) -> bool:
    return any(_matches(n, f) for n in _imported_names(source) for f in FORBIDDEN_MODULES)


def imports_langchain(source: str) -> bool:
    return any(n.split(".")[0].lower().startswith("langchain") for n in _imported_names(source))


def lists_forbidden_distribution(manifest_text: str) -> list[str]:
    """Forbidden distributions named in a dependency string or as a TOML key (comments ignored)."""
    found: list[str] = []
    lines = [ln for ln in manifest_text.splitlines() if not ln.lstrip().startswith("#")]
    for dist in FORBIDDEN_DISTRIBUTIONS:
        quoted = re.compile(rf"[\"']{re.escape(dist)}(?![\w.-])", re.IGNORECASE)  # TOML basic or literal string
        key = re.compile(rf"\s*{re.escape(dist)}\s*=", re.IGNORECASE)
        if any(quoted.search(ln) or key.match(ln) for ln in lines):
            found.append(dist)
    return found


def python_sources() -> list[Path]:
    found: list[Path] = []
    for top in ("packages", "apps"):
        for path in sorted((REPO_ROOT / top).rglob("*.py")):
            if {".venv", "__pycache__"} & set(path.parts):
                continue
            found.append(path)
    return found


def test_vendor_sdks_only_in_adapters() -> None:
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in python_sources()
        if not p.is_relative_to(ADAPTERS_DIR) and imports_vendor_sdk(p.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"vendor SDK imported outside providers adapters: {offenders}"


def test_providers_do_not_import_langchain() -> None:
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in python_sources()
        if p.is_relative_to(PROVIDERS_ROOT) and imports_langchain(p.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"LangChain imported in providers: {offenders}"


def test_manifests_do_not_list_vendor_distributions() -> None:
    manifests = [REPO_ROOT / "pyproject.toml"]
    manifests += sorted(REPO_ROOT.glob("packages/*/pyproject.toml"))
    manifests += sorted(REPO_ROOT.glob("apps/*/pyproject.toml"))
    offenders = {
        str(m.relative_to(REPO_ROOT)): found
        for m in manifests
        if m != PROVIDERS_MANIFEST and (found := lists_forbidden_distribution(m.read_text(encoding="utf-8")))
    }
    assert offenders == {}, f"vendor distributions listed outside providers: {offenders}"


def test_vendor_detector_flags_imports() -> None:
    # Negative self-tests: a broken detector must not pass silently.
    assert imports_vendor_sdk("import anthropic")
    assert imports_vendor_sdk("from openai import OpenAI")
    assert imports_vendor_sdk("import litellm.router")
    assert imports_vendor_sdk("import google.genai")
    assert imports_vendor_sdk("from google import genai")
    assert imports_vendor_sdk("from google import generativeai as g")
    assert imports_vendor_sdk("from google.generativeai import types")
    assert not imports_vendor_sdk("import os\nfrom pathlib import Path\nimport google.protobuf")
    assert not imports_vendor_sdk("from google import protobuf")
    assert not imports_vendor_sdk("import openai_compat_local")


def test_langchain_detector() -> None:
    assert imports_langchain("import langchain_core")
    assert imports_langchain("from langchain.chat_models import X")
    assert imports_langchain("from langchain_community.llms import Y")
    assert not imports_langchain("import langgraph")


def test_manifest_detector() -> None:
    assert lists_forbidden_distribution('dependencies = ["litellm==1.0"]') == ["litellm"]
    assert lists_forbidden_distribution("dependencies = ['openai>=1']") == ["openai"]
    assert lists_forbidden_distribution('dependencies = [\n  "google-genai>=1",\n]') == ["google-genai"]
    assert lists_forbidden_distribution("openai = { workspace = true }") == ["openai"]
    assert lists_forbidden_distribution('dependencies = ["pydantic>=2"]') == []
    assert lists_forbidden_distribution('# "openai" is banned here') == []
