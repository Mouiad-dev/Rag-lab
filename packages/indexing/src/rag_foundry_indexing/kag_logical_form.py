"""operators named after the KAG paper's logical forms; our own JSON grammar, not OpenSPG's"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_foundry_domain.graphs import GraphSpan
from rag_foundry_domain.knowledge import Conflict, KnowledgeGraph, KnowledgeSchema, PropertyDef, RelationTypeDef


@dataclass(frozen=True)
class PlanRejection:
    reason: str
    detail: str


@dataclass(frozen=True)
class Step:
    op: str
    id: str
    subject: dict[str, Any] | None = None
    predicate: str | None = None
    object: dict[str, Any] | None = None
    input: str | None = None
    cmp: str | None = None
    value: str | int | None = None
    order: str | None = None
    limit: int | None = None
    inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class LogicalForm:
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Binding:
    value: str | int
    fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class StepResult:
    step_id: str
    op: str
    status: str
    bindings: tuple[Binding, ...]
    examined_fact_ids: tuple[str, ...] = ()
    missing: tuple[tuple[str, str], ...] = ()
    conflicts: tuple[Conflict, ...] = ()


@dataclass(frozen=True)
class Execution:
    status: str
    steps: tuple[StepResult, ...]
    answer: tuple[Binding, ...]
    fact_ids: tuple[str, ...]
    sources: tuple[GraphSpan, ...]


def _reject(reason: str, detail: str) -> PlanRejection:
    return PlanRejection(reason, detail)


def _predicate_for_side(schema: KnowledgeSchema, entity_type: str, name: str, reverse: bool):
    if reverse:
        return next(
            (relation for relation in schema.relation_types if relation.name == name and relation.range == entity_type),
            None,
        )
    return schema.predicate(entity_type, name)


def _reverse_domains(schema: KnowledgeSchema, range_type: str, name: str) -> set[str]:
    # A relation name may exist on several domains; a reverse retrieve can return subjects of each.
    return {r.domain for r in schema.relation_types if r.name == name and r.range == range_type}


def _earlier_ref(value: object, ids: set[str], current_id: str) -> bool:
    return isinstance(value, str) and value in ids and value != current_id


def _single_property(pred: object) -> bool:
    return isinstance(pred, PropertyDef) and pred.cardinality == "one"


def _value_matches_kind(kind: str, value: object) -> bool:
    if kind == "text":
        return isinstance(value, str)
    return isinstance(value, int) and not isinstance(value, bool)


def _is_ref(value: object) -> bool:
    if not isinstance(value, dict) or len(value) != 1:
        return False
    if "step" in value:
        return isinstance(value["step"], str)
    if "entity" in value:
        entity = value["entity"]
        return (
            isinstance(entity, dict)
            and set(entity) == {"name", "type"}
            and all(isinstance(entity[k], str) and entity[k] for k in ("name", "type"))
        )
    return False


def parse_logical_form(data: object, schema: KnowledgeSchema) -> LogicalForm | PlanRejection:
    """Validate the versioned JSON grammar against the supplied knowledge schema."""
    if not isinstance(data, dict) or set(data) != {"steps"} or not isinstance(data["steps"], list):
        return _reject("invalid_logical_form", "expected an object with a steps array")
    raw_steps = data["steps"]
    if not 1 <= len(raw_steps) <= 8:
        return _reject("invalid_logical_form", "steps must contain between one and eight entries")

    ids: set[str] = set()
    output_count = 0
    parsed: list[Step] = []
    known_types: dict[str, set[str]] = {}
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict) or not isinstance(raw.get("op"), str):
            return _reject("invalid_logical_form", "each step must be an object with an op")
        op = raw["op"]
        if op not in {"retrieve", "filter", "count", "sort", "output"}:
            return _reject("unsupported_operator", f"unsupported operator: {op}")
        step_id = raw.get("id")
        if step_id != f"s{index + 1}" or step_id in ids:
            return _reject("invalid_logical_form", "step ids must be consecutive s1 through s8")
        ids.add(step_id)

        if op == "retrieve":
            if set(raw) != {"op", "id", "subject", "predicate", "object"}:
                return _reject("invalid_logical_form", "retrieve has missing or extra fields")
            subject, obj, predicate = raw["subject"], raw["object"], raw["predicate"]
            if not isinstance(predicate, str) or (subject is None) == (obj is None):
                return _reject("invalid_logical_form", "retrieve requires a predicate and exactly one null side")
            if subject is not None and not _is_ref(subject) or obj is not None and not _is_ref(obj):
                return _reject("invalid_logical_form", "invalid retrieve reference")
            if subject is None and not _is_ref(obj):
                return _reject("invalid_logical_form", "reverse retrieve requires an entity or step object reference")
            ref = obj if subject is None else subject
            if "step" in ref and not _earlier_ref(ref["step"], ids, step_id):
                return _reject("invalid_logical_form", "references may only name earlier steps")
            if "entity" in ref:
                entity_type = ref["entity"]["type"]
                if not any(t.name == entity_type for t in schema.entity_types):
                    return _reject("schema_violation", f"unknown entity type: {entity_type}")
                pred_def = _predicate_for_side(schema, entity_type, predicate, subject is None)
                if pred_def is None:
                    return _reject("schema_violation", f"unknown predicate {predicate} for {entity_type}")
                if subject is None:
                    out_types = _reverse_domains(schema, entity_type, predicate)
                else:
                    out_types = {pred_def.range} if isinstance(pred_def, RelationTypeDef) else set()
            else:
                src = known_types.get(ref["step"], set())
                matches = [_predicate_for_side(schema, t, predicate, subject is None) for t in src]
                if not src or any(p is None for p in matches):
                    return _reject("schema_violation", f"unknown predicate {predicate} for referenced step")
                out_types = (
                    {p.range for p in matches if isinstance(p, RelationTypeDef)}
                    if subject is not None
                    else {d for t in src for d in _reverse_domains(schema, t, predicate)}
                )
            parsed.append(Step(op, step_id, subject=subject, predicate=predicate, object=obj))
            known_types[step_id] = out_types
        elif op == "filter":
            if set(raw) != {"op", "id", "input", "predicate", "cmp", "value"}:
                return _reject("invalid_logical_form", "filter has missing or extra fields")
            input_id, predicate, cmp, value = raw["input"], raw["predicate"], raw["cmp"], raw["value"]
            if (
                not _earlier_ref(input_id, ids, step_id)
                or not isinstance(predicate, str)
                or not isinstance(cmp, str)
                or cmp not in {"eq", "ne", "gt", "ge", "lt", "le"}
                or not isinstance(value, (str, int))
            ):
                return _reject("invalid_logical_form", "invalid filter fields or reference")
            types = known_types.get(input_id, set())
            preds = [schema.predicate(t, predicate) for t in types]
            if not types or any(not _single_property(p) for p in preds):
                return _reject("schema_violation", "filter requires a single-valued property on its input type")
            if any(not _value_matches_kind(p.value_kind, value) for p in preds):
                return _reject("schema_violation", "filter value does not match the property value kind")
            parsed.append(Step(op, step_id, input=input_id, predicate=predicate, cmp=cmp, value=value))
            known_types[step_id] = types
        elif op == "sort":
            if set(raw) != {"op", "id", "input", "predicate", "order", "limit"}:
                return _reject("invalid_logical_form", "sort has missing or extra fields")
            input_id, predicate, order, limit = raw["input"], raw["predicate"], raw["order"], raw["limit"]
            if (
                not _earlier_ref(input_id, ids, step_id)
                or not isinstance(predicate, str)
                or not isinstance(order, str)
                or order not in {"asc", "desc"}
                or isinstance(limit, bool)
                or not isinstance(limit, int)
                or not 1 <= limit <= 20
            ):
                return _reject("invalid_logical_form", "invalid sort fields or reference")
            types = known_types.get(input_id, set())
            sort_preds = [schema.predicate(t, predicate) for t in types]
            if not types or any(not _single_property(p) for p in sort_preds):
                return _reject("schema_violation", "sort requires a single-valued property on its input type")
            if len({"text" if p.value_kind == "text" else "number" for p in sort_preds}) != 1:
                return _reject("schema_violation", "sort property must have one comparable value kind across types")
            parsed.append(Step(op, step_id, input=input_id, predicate=predicate, order=order, limit=limit))
            known_types[step_id] = types
        elif op == "count":
            if set(raw) != {"op", "id", "input"} or not _earlier_ref(raw["input"], ids, step_id):
                return _reject("invalid_logical_form", "count has invalid fields or reference")
            parsed.append(Step(op, step_id, input=raw["input"]))
            known_types[step_id] = set()
        else:
            output_count += 1
            if (
                index != len(raw_steps) - 1
                or set(raw) != {"op", "id", "inputs"}
                or not isinstance(raw["inputs"], list)
                or not raw["inputs"]
                or any(not _earlier_ref(x, ids, step_id) for x in raw["inputs"])
                or len(set(raw["inputs"])) != len(raw["inputs"])
            ):
                return _reject("invalid_logical_form", "output must be last and reference earlier step ids")
            parsed.append(Step(op, step_id, inputs=tuple(raw["inputs"])))
            known_types[step_id] = set().union(*(known_types.get(x, set()) for x in raw["inputs"]))
    if output_count != 1:
        return _reject("invalid_logical_form", "exactly one output step is required")
    return LogicalForm(tuple(parsed))


def _unique(values: Any) -> tuple[Any, ...]:
    return tuple(sorted(set(values)))


def execute(form: LogicalForm, graph: KnowledgeGraph) -> Execution:
    """Run a validated form with stable ordering and source fact provenance."""
    entity_by_id = {entity.entity_id: entity for entity in graph.entities}
    facts_by_id = {fact.fact_id: fact for fact in graph.facts}
    conflicts = {(c.subject_id, c.predicate): c for c in graph.conflicts()}
    results: dict[str, StepResult] = {}
    for step in form.steps:
        bindings: list[Binding] = []
        examined: set[str] = set()
        missing: set[tuple[str, str]] = set()
        step_conflicts: dict[tuple[str, str], Conflict] = {}
        status = "ok"
        if step.op == "retrieve":
            ref = step.subject if step.subject is not None else step.object
            reverse = step.subject is None
            if "entity" in ref:
                ent = ref["entity"]
                entities = graph.entity_by_name(ent["name"], ent["type"])
                if not entities:
                    status = "unknown_entity"
                candidates = tuple(Binding(e.entity_id, ()) for e in entities)
            else:
                candidates = results[ref["step"]].bindings
            merged: dict[str | int, set[str]] = {}
            for binding in candidates:
                if not isinstance(binding.value, str) or binding.value not in entity_by_id:
                    continue
                entity_id = binding.value
                if not reverse:
                    conflict = conflicts.get((entity_id, step.predicate))
                    if conflict is not None:
                        step_conflicts[(entity_id, step.predicate)] = conflict
                        examined.update(conflict.fact_ids)
                        continue
                matches = tuple(
                    f
                    for f in graph.facts
                    if f.predicate == step.predicate
                    and ((f.object_id == entity_id) if reverse else (f.subject_id == entity_id))
                )
                examined.update(f.fact_id for f in matches)
                if not matches:
                    missing.add((entity_id, step.predicate))
                    continue
                for fact in matches:
                    value = fact.subject_id if reverse else fact.value
                    merged.setdefault(value, set()).update(binding.fact_ids)
                    merged[value].add(fact.fact_id)
            bindings = [Binding(v, _unique(merged[v])) for v in sorted(merged, key=lambda v: (isinstance(v, str), v))]
            if status != "unknown_entity" and step_conflicts:
                status = "conflict"
            elif status != "unknown_entity" and not bindings:
                status = "missing" if missing else "empty"
        elif step.op in {"filter", "sort"}:
            inputs = results[step.input].bindings
            props: list[tuple[Binding, str | int]] = []
            for binding in inputs:
                entity_id = binding.value
                pair = (entity_id, step.predicate)
                conflict = conflicts.get(pair)
                if conflict is not None:
                    step_conflicts[pair] = conflict
                    examined.update(conflict.fact_ids)
                    continue
                fs = tuple(f for f in graph.facts if f.subject_id == entity_id and f.predicate == step.predicate)
                examined.update(f.fact_id for f in fs)
                if not fs:
                    missing.add(pair)
                    continue
                f = fs[0]
                props.append((Binding(binding.value, _unique(set(binding.fact_ids) | {f.fact_id})), f.value))
            for enriched, value in props:
                if step.op == "filter":
                    if step.cmp == "eq":
                        passed = value == step.value
                    elif step.cmp == "ne":
                        passed = value != step.value
                    elif step.cmp == "gt":
                        passed = value > step.value
                    elif step.cmp == "ge":
                        passed = value >= step.value
                    elif step.cmp == "lt":
                        passed = value < step.value
                    else:
                        passed = value <= step.value
                    if passed:
                        bindings.append(enriched)
                else:
                    bindings.append(enriched)
            if step.op == "sort":
                ordered = sorted(props, key=lambda p: (p[1], str(p[0].value)), reverse=step.order == "desc")
                bindings = [b for b, _ in ordered][: step.limit]
            if step_conflicts:
                status = "conflict"
            elif missing:
                status = "missing"
            elif not bindings:
                status = "empty"
        elif step.op == "count":
            source = results[step.input]
            count_facts = set(f for b in source.bindings for f in b.fact_ids)
            if not source.bindings:
                count_facts.update(source.examined_fact_ids)
            bindings.append(Binding(len(source.bindings), _unique(count_facts)))
        elif step.op == "output":
            input_ids = sorted(step.inputs)
            bindings.extend(b for input_id in input_ids for b in results[input_id].bindings)
            examined.update(f for input_id in input_ids for f in results[input_id].examined_fact_ids)
            missing.update(m for input_id in input_ids for m in results[input_id].missing)
            for input_id in input_ids:
                for c in results[input_id].conflicts:
                    step_conflicts[(c.subject_id, c.predicate)] = c
            if not bindings:
                status = next((results[x].status for x in input_ids if results[x].status != "empty"), "empty")
            elif any(results[x].status == "conflict" for x in input_ids):
                status = "conflict"
        if step.op != "output" and not bindings and step_conflicts:
            status = "conflict"
        results[step.id] = StepResult(
            step_id=step.id,
            op=step.op,
            status=status,
            bindings=tuple(bindings),
            examined_fact_ids=_unique(examined),
            missing=tuple(sorted(missing)),
            conflicts=tuple(step_conflicts[k] for k in sorted(step_conflicts)),
        )

    steps = tuple(results[s.id] for s in form.steps)
    errors = [s.status for s in steps if s.status not in {"ok", "empty"}]
    overall = "conflict" if "conflict" in errors else "incomplete" if errors else "answered"
    output = results[form.steps[-1].id]
    answer = output.bindings if overall == "answered" else ()
    fact_ids = _unique(f for b in answer for f in b.fact_ids)
    spans = {span for fid in fact_ids for span in facts_by_id[fid].evidence}
    sources = tuple(sorted(spans, key=lambda s: (s.chunk_id, s.start, s.end, s.quote)))
    return Execution(overall, steps, answer, fact_ids, sources)
