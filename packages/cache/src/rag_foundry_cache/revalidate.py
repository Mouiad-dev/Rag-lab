"""Revalidation of a cache hit before it may be served (R50/R51, ADR-0021).

A failed check is a miss with a named reason. It is never an access shortcut.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from rag_foundry_contracts.runs import RunResponse
from rag_foundry_contracts.source import SourceAnchor
from rag_foundry_domain.documents import Scope
from rag_foundry_domain.errors import AnchorError, NotFoundError
from rag_foundry_domain.response_cache import CacheEntry
from rag_foundry_ingestion.anchor_service import AnchorService


class SourceGate(Protocol):
    """Reports which source versions are no longer allowed to back a cached answer."""

    checks_revocation: bool
    """False when the gate cannot report revocation; a hit then says so in its trace."""

    async def denied(self, scope: Scope, version_ids: Sequence[str]) -> frozenset[str]: ...


class AllowAllSourceGate:
    """Denies nothing. There is no product source deletion yet; wiring a real gate is an open item."""

    checks_revocation = False

    async def denied(self, scope: Scope, version_ids: Sequence[str]) -> frozenset[str]:
        return frozenset()


def response_anchors(response: RunResponse) -> tuple[SourceAnchor, ...]:
    found = [a for c in response.citations for a in c.anchors]
    found += [a for e in response.evidence for a in e.anchors]
    return tuple(found)


async def revalidate(
    scope: Scope,
    entry: CacheEntry,
    response: RunResponse,
    *,
    now: datetime,
    expected_index_version_id: str | None,
    anchors: AnchorService,
    gate: SourceGate,
) -> str | None:
    """Return None when the entry may be served, else the miss reason. Order is part of the contract."""
    if entry.tenant_id != scope.tenant_id or entry.project_id != scope.project_id:
        return "scope_mismatch"
    if now >= entry.expires_at:
        return "expired"
    if response.pins.index_version_id != expected_index_version_id:
        return "index_version_changed"
    if await gate.denied(scope, entry.source_version_ids):
        return "source_denied"
    found = response_anchors(response)
    if found:
        try:
            resolved = await anchors.resolve(scope, found)
        except (NotFoundError, AnchorError):
            return "anchor_unresolved"
        if any(item.status != "exact" for item in resolved):
            return "anchor_unresolved"
    return None
