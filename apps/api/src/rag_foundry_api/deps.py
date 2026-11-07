"""FastAPI dependency providers for application settings and services."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from rag_foundry_answering.runner import LocalRunner
from rag_foundry_core.settings import Settings
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.intake import IntakeService

from rag_foundry_api.container import AppContainer


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_container(request: Request) -> AppContainer | None:
    return getattr(request.state, "container", None)


ContainerDep = Annotated[AppContainer | None, Depends(get_container)]


def get_intake_service(container: ContainerDep):
    return container.intake if container else None


def get_anchor_service(container: ContainerDep):
    return container.anchors if container else None


def get_run_service(container: ContainerDep):
    return container.runs if container else None


SettingsDep = Annotated[Settings, Depends(get_settings)]
IntakeDep = Annotated[IntakeService | None, Depends(get_intake_service)]
AnchorsDep = Annotated[AnchorService | None, Depends(get_anchor_service)]
RunsDep = Annotated[LocalRunner | None, Depends(get_run_service)]
