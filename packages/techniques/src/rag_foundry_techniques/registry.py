"""Registry of technique plugins keyed by catalogue ID. No built-ins, no discovery (v0)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from rag_foundry_techniques.catalogue import Catalogue, TechniqueEntry
from rag_foundry_techniques.errors import (
    DuplicatePluginError,
    TechniqueError,
    TechniqueUnavailableError,
    UnknownTechniqueError,
)
from rag_foundry_techniques.protocol import PluginManifest


class TechniqueRegistry:
    def __init__(self, catalogue: Catalogue) -> None:
        self._catalogue = catalogue
        self._plugins: dict[str, type] = {}

    def register(self, plugin_cls: type) -> type:
        manifest = getattr(plugin_cls, "manifest", None)
        if not isinstance(manifest, PluginManifest):
            raise TechniqueError(f"plugin {plugin_cls!r} has no valid PluginManifest in 'manifest'")
        tid = manifest.technique_id
        if tid not in self._catalogue:
            raise UnknownTechniqueError(tid)
        if manifest.config_model.model_config.get("extra") != "forbid":
            raise TechniqueError(f"plugin for {tid}: config model must set extra='forbid'")
        if tid in self._plugins:
            raise DuplicatePluginError(f"a plugin is already registered for technique {tid}")
        self._plugins[tid] = plugin_cls
        return plugin_cls

    def entry(self, tid: str) -> TechniqueEntry:
        return self._catalogue.get(tid)

    def plugin(self, tid: str) -> type:
        entry = self._catalogue.get(tid)
        try:
            return self._plugins[tid]
        except KeyError:
            raise TechniqueUnavailableError(
                f"technique {tid} has no registered plugin (catalogue status '{entry.status}')"
            ) from None

    def validate_config(self, tid: str, raw: Mapping[str, Any]) -> BaseModel:
        return self.plugin(tid).manifest.config_model.model_validate(dict(raw))

    def registered_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))
