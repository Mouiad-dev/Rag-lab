"""Scoped R48 KAG domain schema and grounded knowledge graph value object.

The schema declares entity types, typed properties and relation types. Every fact links to source
spans and every chunk links back to its facts. Conflicts are reported, never resolved. The graph
is derived data; evidence is always the original chunk text.
"""

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError, model_validator

from rag_foundry_domain.documents import Scope, Uuid4
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.graphs import GraphSpan, normalize_name

_FROZEN = ConfigDict(frozen=True, extra="forbid")

KG_SCHEMA = "rag_foundry.knowledge_graph/1"
MAX_SCHEMA_JSON_CHARS = 64000
MAX_INT_LITERAL = 2**63  # exclusive bound, SQL bigint
MAX_TEXT_LITERAL = 500

PropertyName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
RelationName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
TypeName = Annotated[str, Field(pattern=r"^[A-Z][A-Za-z0-9]{0,39}$")]
KEntityId = Annotated[str, Field(pattern=r"^ken_[0-9a-f]{16}$")]
FactId = Annotated[str, Field(pattern=r"^fct_[0-9a-f]{16}$")]
Cardinality = Literal["one", "many"]


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def kentity_id_for(name: str, type: str) -> str:
    return "ken_" + hashlib.sha256(_canonical([normalize_name(name), type])).hexdigest()[:16]


def fact_id_for(subject_id: str, predicate: str, object_or_literal: str | int) -> str:
    return "fct_" + hashlib.sha256(_canonical([subject_id, predicate, object_or_literal])).hexdigest()[:16]


def kg_key_for(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


class PropertyDef(BaseModel):
    model_config = _FROZEN
    name: PropertyName
    value_kind: Literal["text", "int", "year"]
    cardinality: Cardinality


class EntityTypeDef(BaseModel):
    model_config = _FROZEN
    name: TypeName
    properties: tuple[PropertyDef, ...] = ()


class RelationTypeDef(BaseModel):
    model_config = _FROZEN
    name: RelationName
    domain: TypeName
    range: TypeName
    cardinality: Cardinality


class KnowledgeSchema(BaseModel):
    model_config = _FROZEN
    schema_id: Annotated[str, Field(min_length=1, max_length=64)]
    version: Annotated[str, Field(min_length=1, max_length=16)]
    entity_types: Annotated[tuple[EntityTypeDef, ...], Field(min_length=1)]
    relation_types: tuple[RelationTypeDef, ...] = ()

    @model_validator(mode="after")
    def _validate_schema(self) -> "KnowledgeSchema":
        names = [t.name for t in self.entity_types]
        if len(set(names)) != len(names):
            raise IndexingError("kag_schema_invalid")
        declared = set(names)
        for rel in self.relation_types:
            if rel.domain not in declared or rel.range not in declared:
                raise IndexingError("kag_schema_invalid")
        for t in self.entity_types:
            used = [p.name for p in t.properties] + [r.name for r in self.relation_types if r.domain == t.name]
            if len(set(used)) != len(used):
                raise IndexingError("kag_schema_invalid")
        if len(self._canonical_json()) > MAX_SCHEMA_JSON_CHARS:
            raise IndexingError("kag_schema_invalid")
        return self

    def _canonical_json(self) -> str:
        return _canonical(self.model_dump(mode="json")).decode("utf-8")

    def schema_sha256(self) -> str:
        return hashlib.sha256(self._canonical_json().encode("utf-8")).hexdigest()

    def predicate(self, subject_type: str, name: str) -> PropertyDef | RelationTypeDef | None:
        for t in self.entity_types:
            if t.name == subject_type:
                for p in t.properties:
                    if p.name == name:
                        return p
        for r in self.relation_types:
            if r.domain == subject_type and r.name == name:
                return r
        return None


class KEntity(BaseModel):
    model_config = _FROZEN
    entity_id: KEntityId
    type: TypeName
    name: Annotated[str, Field(min_length=1, max_length=500)]
    mentions: Annotated[tuple[GraphSpan, ...], Field(min_length=1, max_length=50)]


class Fact(BaseModel):
    model_config = _FROZEN
    fact_id: FactId
    subject_id: KEntityId
    predicate: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
    object_id: KEntityId | None = None
    literal: StrictStr | StrictInt | None = None
    evidence: Annotated[tuple[GraphSpan, ...], Field(min_length=1, max_length=50)]

    @model_validator(mode="after")
    def _validate_fact(self) -> "Fact":
        if (self.object_id is None) == (self.literal is None):
            raise IndexingError("kag_invalid")
        if isinstance(self.literal, int) and abs(self.literal) >= MAX_INT_LITERAL:
            raise IndexingError("kag_invalid")
        if isinstance(self.literal, str) and not 1 <= len(self.literal) <= MAX_TEXT_LITERAL:
            raise IndexingError("kag_invalid")
        return self

    @property
    def value(self) -> str | int:
        return self.object_id if self.object_id is not None else self.literal  # type: ignore[return-value]


class Conflict(BaseModel):
    model_config = _FROZEN
    subject_id: KEntityId
    predicate: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
    fact_ids: Annotated[tuple[FactId, ...], Field(min_length=2)]


def _literal_ok(kind: str, value: str | int) -> bool:
    if kind == "text":
        return isinstance(value, str) and 1 <= len(value) <= MAX_TEXT_LITERAL
    if kind == "int":
        return isinstance(value, int) and abs(value) < MAX_INT_LITERAL
    if kind == "year":
        return isinstance(value, int) and 1000 <= value <= 2999
    return False


class KnowledgeGraph(BaseModel):
    model_config = _FROZEN
    schema_: Literal["rag_foundry.knowledge_graph/1"] = Field(alias="schema")
    kg_key: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    tenant_id: Uuid4
    project_id: Uuid4
    index_version_id: Uuid4
    technique_id: Literal["R48"] = "R48"
    knowledge_schema: KnowledgeSchema
    generator_json: Annotated[str, Field(max_length=8000)]
    entities: Annotated[tuple[KEntity, ...], Field(max_length=5000)] = ()
    facts: Annotated[tuple[Fact, ...], Field(max_length=20000)] = ()

    @model_validator(mode="after")
    def _validate_graph(self) -> "KnowledgeGraph":
        e_ids = [e.entity_id for e in self.entities]
        f_ids = [f.fact_id for f in self.facts]
        if len(set(e_ids)) != len(e_ids) or len(set(f_ids)) != len(f_ids):
            raise IndexingError("kag_invalid")
        if e_ids != sorted(e_ids) or f_ids != sorted(f_ids):
            raise IndexingError("kag_invalid")
        by_id = {e.entity_id: e for e in self.entities}
        declared = {t.name for t in self.knowledge_schema.entity_types}
        if any(e.type not in declared for e in self.entities):
            raise IndexingError("kag_invalid")
        # IDs are derived from content, so one name+type and one (subject, predicate, value) exist once.
        if any(e.entity_id != kentity_id_for(e.name, e.type) for e in self.entities):
            raise IndexingError("kag_invalid")
        if any(f.fact_id != fact_id_for(f.subject_id, f.predicate, f.value) for f in self.facts):
            raise IndexingError("kag_invalid")
        for f in self.facts:
            subject = by_id.get(f.subject_id)
            if subject is None:
                raise IndexingError("kag_invalid")
            pred = self.knowledge_schema.predicate(subject.type, f.predicate)
            if pred is None:
                raise IndexingError("kag_invalid")
            if isinstance(pred, PropertyDef):
                if f.literal is None or f.object_id is not None or not _literal_ok(pred.value_kind, f.literal):
                    raise IndexingError("kag_invalid")
            else:
                if f.object_id is None or f.literal is not None:
                    raise IndexingError("kag_invalid")
                obj = by_id.get(f.object_id)
                if obj is None or obj.type != pred.range:
                    raise IndexingError("kag_invalid")
        return self

    def check_scope(self, scope: Scope, index_version_id: str) -> None:
        if (
            self.tenant_id != scope.tenant_id
            or self.project_id != scope.project_id
            or self.index_version_id != index_version_id
        ):
            raise IndexingError("kag_scope_mismatch")

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json", by_alias=True), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "KnowledgeGraph":
        try:
            data = json.loads(text)
            if not isinstance(data, dict) or data.get("schema") != KG_SCHEMA:
                raise IndexingError("kag_invalid")
            return cls.model_validate(data)
        except (ValueError, ValidationError) as exc:
            if isinstance(exc, IndexingError):
                raise
            raise IndexingError("kag_invalid") from exc

    def entity_by_name(self, name: str, type: str | None = None) -> tuple[KEntity, ...]:
        key = normalize_name(name)
        return tuple(e for e in self.entities if normalize_name(e.name) == key and (type is None or e.type == type))

    def facts_about(self, entity_id: str, predicate: str | None = None, as_object: bool = False) -> tuple[Fact, ...]:
        return tuple(
            f
            for f in self.facts
            if (f.object_id if as_object else f.subject_id) == entity_id
            and (predicate is None or f.predicate == predicate)
        )

    def facts_for_chunk(self, chunk_id: str) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if any(s.chunk_id == chunk_id for s in f.evidence))

    def chunks_for_fact(self, fact_id: str) -> tuple[str, ...]:
        for f in self.facts:
            if f.fact_id == fact_id:
                return tuple(sorted({s.chunk_id for s in f.evidence}))
        return ()

    def conflicts(self) -> tuple[Conflict, ...]:
        by_id = {e.entity_id: e for e in self.entities}
        groups: dict[tuple[str, str], dict[str, str | int]] = {}
        for f in self.facts:
            pred = self.knowledge_schema.predicate(by_id[f.subject_id].type, f.predicate)
            if pred is not None and pred.cardinality == "one":
                groups.setdefault((f.subject_id, f.predicate), {})[f.fact_id] = f.value
        return tuple(
            Conflict(subject_id=s, predicate=p, fact_ids=tuple(sorted(ids)))
            for (s, p), ids in sorted(groups.items())
            if len(set(map(repr, ids.values()))) >= 2
        )
