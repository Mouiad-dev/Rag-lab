"""Scoped RAPTOR summary tree value object (R37).

Leaves hold their source span. Summaries are synthetic and hold only child IDs; lineage (the
original leaf spans under a node) is computed on demand. Summaries are never evidence.
"""

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag_foundry_domain.documents import Scope, Uuid4
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.graphs import GraphSpan

_FROZEN = ConfigDict(frozen=True, extra="forbid")

NodeId = Annotated[str, Field(pattern=r"^rnd_[0-9a-f]{16}$")]
Finite = Annotated[float, Field(allow_inf_nan=False)]
TREE_SCHEMA = "rag_foundry.summary_tree/1"
MAX_SUMMARY_CHARS = 4000


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def node_id_for(level: int, payload: str | list[str] | tuple[str, ...]) -> str:
    """Leaf payload is the chunk ID; summary payload is the child IDs (order-insensitive)."""
    body = payload if isinstance(payload, str) else sorted(payload)
    return "rnd_" + hashlib.sha256(_canonical([level, body])).hexdigest()[:16]


def tree_key_for(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


class TreeNode(BaseModel):
    model_config = _FROZEN
    node_id: NodeId
    level: Annotated[int, Field(ge=0, le=8)]
    kind: Literal["leaf", "summary"]
    text: str
    span: GraphSpan | None = None
    children: Annotated[tuple[NodeId, ...], Field(max_length=50)] = ()
    embedding: tuple[Finite, ...] = ()
    synthetic: bool

    @model_validator(mode="after")
    def _validate_node(self) -> "TreeNode":
        if self.kind == "leaf":
            if self.level != 0 or self.span is None or self.children or self.synthetic:
                raise ValueError("invalid leaf")
            if self.text != self.span.quote:
                raise ValueError("leaf text must equal span quote")
        else:
            if self.level < 1 or self.span is not None or not 1 <= len(self.children) <= 50 or not self.synthetic:
                raise ValueError("invalid summary")
            if not 1 <= len(self.text) <= MAX_SUMMARY_CHARS:
                raise ValueError("summary text length")
        return self


class SummaryTree(BaseModel):
    model_config = _FROZEN
    schema_: Literal["rag_foundry.summary_tree/1"] = Field(alias="schema")
    tree_key: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    tenant_id: Uuid4
    project_id: Uuid4
    index_version_id: Uuid4
    technique_id: Literal["R37"] = "R37"
    embedding_key: Annotated[str, Field(min_length=1, max_length=500)]
    generator_json: Annotated[str, Field(max_length=8000)]
    nodes: Annotated[tuple[TreeNode, ...], Field(min_length=1, max_length=4000)]

    @model_validator(mode="after")
    def _validate_tree(self) -> "SummaryTree":
        ids = [n.node_id for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate node id")
        if [(n.level, n.node_id) for n in self.nodes] != sorted((n.level, n.node_id) for n in self.nodes):
            raise ValueError("nodes must be sorted by level and id")
        dims = {len(n.embedding) for n in self.nodes}
        if len(dims) != 1 or min(dims) < 1:
            raise ValueError("embedding dimension")
        by_id = {n.node_id: n for n in self.nodes}
        parents: dict[str, int] = {}
        for n in self.nodes:
            for c in n.children:
                child = by_id.get(c)
                if child is None or child.level >= n.level:
                    raise ValueError("child missing or not at a lower level")
                parents[c] = parents.get(c, 0) + 1
        if any(count != 1 for count in parents.values()):
            raise ValueError("node has more than one parent")
        return self  # strictly lower child levels also rule out cycles

    def check_scope(self, scope: Scope, index_version_id: str) -> None:
        if (
            self.tenant_id != scope.tenant_id
            or self.project_id != scope.project_id
            or self.index_version_id != index_version_id
        ):
            raise IndexingError("tree_scope_mismatch")

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json", by_alias=True), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "SummaryTree":
        try:
            data = json.loads(text)
            if not isinstance(data, dict) or data.get("schema") != TREE_SCHEMA:
                raise IndexingError("tree_invalid")
            return cls.model_validate(data)
        except (ValueError, ValidationError) as exc:
            if isinstance(exc, IndexingError):
                raise
            raise IndexingError("tree_invalid") from exc

    def _node(self, node_id: str) -> TreeNode:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise IndexingError("tree_invalid")

    def roots(self) -> tuple[str, ...]:
        child_ids = {c for n in self.nodes for c in n.children}
        return tuple(n.node_id for n in self.nodes if n.node_id not in child_ids)

    def children_of(self, node_id: str) -> tuple[str, ...]:
        return self._node(node_id).children

    def leaf_ids(self, node_id: str) -> tuple[str, ...]:
        found: list[str] = []
        stack = [node_id]
        while stack:
            n = self._node(stack.pop())
            if n.kind == "leaf":
                found.append(n.node_id)
            else:
                stack.extend(n.children)
        return tuple(sorted(found))

    def lineage(self, node_id: str) -> tuple[GraphSpan, ...]:
        spans = [self._node(i).span for i in self.leaf_ids(node_id)]
        return tuple(sorted((s for s in spans if s is not None), key=lambda s: (s.extraction_view_id, s.start, s.end)))

    def levels(self) -> tuple[tuple[str, ...], ...]:
        top = max(n.level for n in self.nodes)
        return tuple(tuple(n.node_id for n in self.nodes if n.level == lv) for lv in range(top + 1))
