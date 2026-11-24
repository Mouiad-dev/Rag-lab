"""Reusable contract checks for technique plugins (plain functions, no pytest)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from rag_foundry_techniques.catalogue import Catalogue
from rag_foundry_techniques.errors import DuplicatePluginError, TechniqueUnavailableError
from rag_foundry_techniques.kinds import ChunkInput, ChunkSpan
from rag_foundry_techniques.protocol import PluginKind, PluginManifest
from rag_foundry_techniques.registry import TechniqueRegistry


class PluginContractError(AssertionError):
    """A plugin violates the technique plugin contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PluginContractError(message)


def check_plugin_contract(
    plugin_cls: type, catalogue: Catalogue, *, sample_config: Mapping[str, Any] | None = None
) -> PluginManifest:
    manifest = getattr(plugin_cls, "manifest", None)
    _require(isinstance(manifest, PluginManifest), f"{plugin_cls!r} has no valid PluginManifest in 'manifest'")
    tid = manifest.technique_id
    _require(tid in catalogue, f"technique ID {tid} is not in the catalogue")
    allowed = catalogue.kinds_for(tid)
    _require(manifest.kind.value in allowed, f"{tid}: kind '{manifest.kind.value}' not in allowed kinds {allowed}")
    config_model = manifest.config_model
    _require(config_model.model_config.get("extra") == "forbid", f"{tid}: config model must set extra='forbid'")
    try:
        config_model.model_json_schema()
    except Exception as exc:
        raise PluginContractError(f"{tid}: model_json_schema() failed: {type(exc).__name__}: {exc}") from exc

    registry = TechniqueRegistry(catalogue)
    try:
        registry.plugin(tid)
    except TechniqueUnavailableError:
        pass
    except Exception as exc:
        raise PluginContractError(
            f"{tid}: unregistered lookup must raise TechniqueUnavailableError, got {type(exc).__name__}"
        ) from exc
    else:
        raise PluginContractError(f"{tid}: fresh registry unexpectedly has a plugin")
    registry.register(plugin_cls)
    try:
        registry.register(plugin_cls)
    except DuplicatePluginError:
        pass
    else:
        raise PluginContractError(f"{tid}: second registration must raise DuplicatePluginError")

    if sample_config is not None:
        try:
            config_model.model_validate(dict(sample_config))
        except ValidationError as exc:
            raise PluginContractError(f"{tid}: sample_config does not validate: {exc}") from exc
        try:
            config_model.model_validate({**dict(sample_config), "__contract_unknown__": 1})
        except ValidationError:
            pass
        else:
            raise PluginContractError(f"{tid}: unknown config key was accepted")
    return manifest


def check_chunker_run(plugin_cls: type, config: Mapping[str, Any], doc: ChunkInput) -> tuple[ChunkSpan, ...]:
    manifest = getattr(plugin_cls, "manifest", None)
    _require(isinstance(manifest, PluginManifest), f"{plugin_cls!r} has no valid PluginManifest in 'manifest'")
    _require(manifest.kind == PluginKind.CHUNKER, f"{manifest.technique_id}: kind is not 'chunker'")
    model = manifest.config_model.model_validate(dict(config))
    spans = plugin_cls(model).chunk(doc)
    previous_start = -1
    for span in spans:
        _require(isinstance(span, ChunkSpan), f"span {span!r} is not a ChunkSpan")
        _require(0 <= span.start < span.end <= len(doc.text), f"span {span.start}-{span.end} is out of bounds")
        _require(span.start >= previous_start, "spans are not in ascending order")
        previous_start = span.start
    return tuple(spans)
