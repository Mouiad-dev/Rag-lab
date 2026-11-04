"""Dev-only scope resolver (ADR-0009 D10). Real authentication arrives in P10.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from rag_foundry_domain import LOCAL_DEV_SCOPE, NotFoundError, Scope

from rag_foundry_api.errors import api_error


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    project_ids: tuple[str, ...]

    def scope_for(self, project_id: str) -> Scope:
        """Out-of-scope and unknown projects raise the same NotFoundError."""
        if project_id not in self.project_ids:
            raise NotFoundError()
        return Scope(tenant_id=self.tenant_id, project_id=project_id)

    def scopes(self) -> list[Scope]:
        return [Scope(tenant_id=self.tenant_id, project_id=p) for p in sorted(self.project_ids)]


def get_auth_context(request: Request) -> AuthContext:
    if request.app.state.settings.environment == "prod":
        raise api_error(401, "auth_not_configured", "authentication is not configured")
    return AuthContext(tenant_id=LOCAL_DEV_SCOPE.tenant_id, project_ids=(LOCAL_DEV_SCOPE.project_id,))


Auth = Annotated[AuthContext, Depends(get_auth_context)]
