"""Scoped entity graph value object (R38 GraphRAG, R39 communities).

Every entity mention and relation evidence is a source span. Graph content is derived data,
never evidence by itself; answers cite the spans.
"""

import hashlib
import json
import unicodedata
from collections import deque
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag_foundry_domain.documents import Scope, Uuid4
from rag_foundry_domain.errors import IndexingError

_FROZEN = ConfigDict(frozen=True, extra="forbid")

EntityId = Annotated[str, Field(pattern=r"^ent_[0-9a-f]{16}$")]
RelationId = Annotated[str, Field(pattern=r"^rel_[0-9a-f]{16}$")]
Direction = Literal["forward", "reverse"]
GRAPH_SCHEMA = "rag_foundry.entity_graph/1"


class GraphSpan(BaseModel):
    model_config = _FROZEN
    chunk_id: Uuid4
    extraction_view_id: Uuid4
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]
    quote: str

    @model_validator(mode="after")
    def _validate_span(self) -> "GraphSpan":
        if self.end <= self.start or len(self.quote) != self.end - self.start:
            raise ValueError("quote length and span disagree")
        return self


class Entity(BaseModel):
    model_config = _FROZEN
    entity_id: EntityId
    name: Annotated[str, Field(min_length=1, max_length=500)]
    type: Annotated[str, Field(min_length=1, max_length=64)]
    mentions: Annotated[tuple[GraphSpan, ...], Field(min_length=1, max_length=50)]


class Relation(BaseModel):
    model_config = _FROZEN
    relation_id: RelationId
    source_id: EntityId
    target_id: EntityId
    label: Annotated[str, Field(min_length=1, max_length=200)]
    evidence: Annotated[tuple[GraphSpan, ...], Field(min_length=1, max_length=50)]


def normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def entity_id_for(name: str, type: str) -> str:
    digest = hashlib.sha256(_canonical([normalize_name(name), normalize_name(type)])).hexdigest()
    return "ent_" + digest[:16]


def relation_id_for(source_id: str, label: str, target_id: str) -> str:
    digest = hashlib.sha256(_canonical([source_id, normalize_name(label), target_id])).hexdigest()
    return "rel_" + digest[:16]


def graph_key_for(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


class EntityGraph(BaseModel):
    model_config = _FROZEN
    schema_: Literal["rag_foundry.entity_graph/1"] = Field(alias="schema")
    graph_key: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    tenant_id: Uuid4
    project_id: Uuid4
    index_version_id: Uuid4
    technique_id: Literal["R38"] = "R38"
    generator_json: Annotated[str, Field(max_length=8000)]
    entities: Annotated[tuple[Entity, ...], Field(max_length=5000)]
    relations: Annotated[tuple[Relation, ...], Field(max_length=10000)]

    @model_validator(mode="after")
    def _validate_graph(self) -> "EntityGraph":
        entity_ids = [e.entity_id for e in self.entities]
        relation_ids = [r.relation_id for r in self.relations]
        if len(set(entity_ids)) != len(entity_ids) or len(set(relation_ids)) != len(relation_ids):
            raise ValueError("duplicate id")
        known = set(entity_ids)
        if any(r.source_id not in known or r.target_id not in known for r in self.relations):
            raise ValueError("relation endpoint missing")
        if entity_ids != sorted(entity_ids) or relation_ids != sorted(relation_ids):
            raise ValueError("entities and relations must be sorted by id")
        return self

    def check_scope(self, scope: Scope, index_version_id: str) -> None:
        if (
            self.tenant_id != scope.tenant_id
            or self.project_id != scope.project_id
            or self.index_version_id != index_version_id
        ):
            raise IndexingError("graph_scope_mismatch")

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json", by_alias=True), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "EntityGraph":
        try:
            data = json.loads(text)
            if not isinstance(data, dict) or data.get("schema") != GRAPH_SCHEMA:
                raise IndexingError("graph_invalid")
            return cls.model_validate(data)
        except (ValueError, ValidationError) as exc:
            if isinstance(exc, IndexingError):
                raise
            raise IndexingError("graph_invalid") from exc

    def neighbors(self, entity_id: str) -> tuple[tuple[str, str, Direction], ...]:
        found: list[tuple[str, str, Direction]] = []
        for r in self.relations:
            if r.source_id == entity_id:
                found.append((r.relation_id, r.target_id, "forward"))
            if r.target_id == entity_id:
                found.append((r.relation_id, r.source_id, "reverse"))
        return tuple(sorted(found))

    def shortest_path(self, a: str, b: str, max_hops: int) -> tuple[tuple[str, Direction], ...] | None:
        known = {e.entity_id for e in self.entities}
        if a not in known or b not in known:
            return None
        if a == b:
            return ()
        parents: dict[str, tuple[str, str, Direction]] = {}
        seen = {a}
        queue: deque[tuple[str, int]] = deque([(a, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for rel_id, other, direction in self.neighbors(node):
                if other in seen:
                    continue
                seen.add(other)
                parents[other] = (node, rel_id, direction)
                if other == b:
                    path: list[tuple[str, Direction]] = []
                    cur = b
                    while cur != a:
                        prev, rid, d = parents[cur]
                        path.append((rid, d))
                        cur = prev
                    return tuple(reversed(path))
                queue.append((other, depth + 1))
        return None
