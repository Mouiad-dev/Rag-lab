"""Opt-in discovery of third-party technique plugins via entry points.

Nothing here runs at import time; call ``discover_plugins`` explicitly.

``EntryPoint.load()`` imports third-party modules. Import side effects (module state,
``sys.modules`` entries) are NOT rolled back when discovery fails; "unchanged" refers to
the registry only. Plugins are trusted in-process code (ADR-0002).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from rag_foundry_techniques.errors import (
    DuplicatePluginError,
    PluginDiscoveryError,
    PluginDiscoveryFailure,
    TechniqueError,
)
from rag_foundry_techniques.registry import TechniqueRegistry

ENTRY_POINT_GROUP = "rag_foundry.techniques"


@dataclass(frozen=True)
class DiscoveredPlugin:
    entry_point: str
    value: str
    distribution: str | None
    distribution_version: str | None
    technique_id: str
    kind: str
    plugin_version: str


def _dist(ep: Any) -> tuple[str | None, str | None]:
    dist = getattr(ep, "dist", None)
    if dist is None:
        return None, None
    return dist.name, dist.version


def discover_plugins(
    registry: TechniqueRegistry, *, entry_points: Iterable[Any] | None = None
) -> tuple[DiscoveredPlugin, ...]:
    """Load, validate and register plugins from the ``rag_foundry.techniques`` group.

    All-or-nothing: on any failure a ``PluginDiscoveryError`` is raised and the registry is
    left unchanged. Imported modules stay imported (see module docstring).
    """
    if entry_points is None:
        entry_points = metadata.entry_points(group=ENTRY_POINT_GROUP)
    candidates = sorted(entry_points, key=lambda e: (e.name, e.value, _dist(e)[0] or ""))

    failures: list[PluginDiscoveryFailure] = []
    valid: list[tuple[Any, type, Any]] = []

    def fail(ep: Any, reason: str, error_type: str) -> None:
        failures.append(PluginDiscoveryFailure(ep.name, ep.value, _dist(ep)[0], reason, error_type))

    for ep in candidates:
        if ep.group != ENTRY_POINT_GROUP:
            fail(ep, f"entry point group '{ep.group}' is not '{ENTRY_POINT_GROUP}'", "TechniqueError")
            continue
        try:
            obj = ep.load()
        except Exception as exc:
            fail(ep, f"import failed: {type(exc).__name__}: {exc}", type(exc).__name__)
            continue
        if not isinstance(obj, type):
            # type name only: a third-party __repr__ may raise
            fail(ep, f"loaded object of type {type(obj).__name__} is not a class", "TechniqueError")
            continue
        try:
            manifest = registry.check(obj)
        except TechniqueError as exc:
            fail(ep, str(exc), type(exc).__name__)
            continue
        except Exception as exc:
            # a broken third-party class (e.g. a raising manifest lookup) still becomes a typed failure
            fail(ep, f"check failed: {type(exc).__name__}: {exc}", type(exc).__name__)
            continue
        if ep.name != manifest.technique_id:
            fail(
                ep,
                f"entry point name '{ep.name}' does not match technique ID '{manifest.technique_id}'",
                "TechniqueError",
            )
            continue
        valid.append((ep, obj, manifest))

    by_id: dict[str, list[Any]] = defaultdict(list)
    for ep, _cls, manifest in valid:
        by_id[manifest.technique_id].append(ep)
    for tid, eps in by_id.items():
        if len(eps) > 1:
            for ep in eps:
                fail(ep, f"duplicate entry point for {tid}", DuplicatePluginError.__name__)

    if failures:
        raise PluginDiscoveryError(failures)

    report: list[DiscoveredPlugin] = []
    for ep, cls, manifest in valid:
        registry.register(cls)
        name, version = _dist(ep)
        report.append(
            DiscoveredPlugin(
                entry_point=ep.name,
                value=ep.value,
                distribution=name,
                distribution_version=version,
                technique_id=manifest.technique_id,
                kind=manifest.kind.value,
                plugin_version=manifest.version,
            )
        )
    return tuple(report)
