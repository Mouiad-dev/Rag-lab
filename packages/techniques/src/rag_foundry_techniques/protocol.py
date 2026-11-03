"""Plugin protocol v0 (ADR-0002): kinds, fidelity and the manifest every plugin declares."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION = "0"


class PluginKind(StrEnum):
    CHUNKER = "chunker"
    REPRESENTATION = "representation"
    RETRIEVER = "retriever"
    FUSION = "fusion"
    QUERY_TRANSFORM = "query_transform"
    RERANKER = "reranker"
    CONTEXT_BUILDER = "context_builder"
    WORKFLOW = "workflow"
    GRAPH_BUILDER = "graph_builder"
    KNOWLEDGE_STORE = "knowledge_store"
    CACHE = "cache"
    EVALUATOR = "evaluator"


class Fidelity(StrEnum):
    FAITHFUL = "faithful"
    ADAPTATION = "adaptation"
    SIMULATION = "simulation"


class PluginManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    technique_id: str
    kind: PluginKind
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    fidelity: Fidelity
    required_capabilities: frozenset[str] = frozenset()
    config_model: type[BaseModel]


@runtime_checkable
class TechniquePlugin(Protocol):
    """A technique plugin class declares a class-level manifest.

    Kind-specific methods (for example ``chunk(...)``) are explicitly deferred to later
    steps (P3.4 validates the API with two built-in plugins; v1 freezes at P7.1).
    Plugins are trusted in-process code (ADR-0002).
    """

    manifest: ClassVar[PluginManifest]
