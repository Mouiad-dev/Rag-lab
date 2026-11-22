import pytest
from rag_foundry_domain.knowledge import PropertyDef, RelationTypeDef, fact_id_for, kentity_id_for
from rag_foundry_indexing.kag_logical_form import LogicalForm, PlanRejection, execute, parse_logical_form

from tests.domain.test_knowledge import ent, fact, make_kg, make_schema, span


def cedar_graph():
    cedar = ent("Cedar Labs", "Organization")
    harbor = ent("Harbor Authority", "Organization")
    omar = ent("Omar Nasser", "Person")
    delta = ent("Delta Works", "Organization")
    evidence = {
        "supply": span("supplies"),
        "lead": span("leads"),
        "cedar_year": span("2020"),
        "delta_year": span("2010"),
    }
    facts = [
        fact(cedar, "supplies", harbor, [evidence["supply"]]),
        fact(omar, "leads", harbor, [evidence["lead"]]),
        fact(cedar, "founded_year", 2020, [evidence["cedar_year"]]),
        fact(delta, "supplies", harbor, [evidence["supply"]]),
        fact(delta, "founded_year", 2010, [evidence["delta_year"]]),
    ]
    entities = [cedar, harbor, omar, delta]
    return make_kg(entities, facts), {"cedar": cedar, "harbor": harbor, "omar": omar, "delta": delta}, facts


def form(*steps, schema=None):
    parsed = parse_logical_form({"steps": list(steps)}, schema or make_schema())
    assert isinstance(parsed, LogicalForm)
    return parsed


def entity_ref(name, type="Organization"):
    return {"entity": {"name": name, "type": type}}


def test_multihop_retrieve_returns_omar_with_fact_and_span_provenance():
    graph, entities, _ = cedar_graph()
    result = execute(
        form(
            {
                "op": "retrieve",
                "id": "s1",
                "subject": entity_ref("Cedar Labs"),
                "predicate": "supplies",
                "object": None,
            },
            {"op": "retrieve", "id": "s2", "subject": None, "predicate": "leads", "object": {"step": "s1"}},
            {"op": "output", "id": "s3", "inputs": ["s2"]},
        ),
        graph,
    )
    expected = tuple(
        sorted(
            fact_id_for(f.subject_id, f.predicate, f.value)
            for f in graph.facts
            if f.predicate in {"supplies", "leads"}
            and f.subject_id in {entities["cedar"].entity_id, entities["omar"].entity_id}
        )
    )
    assert result.status == "answered"
    assert result.answer[0].value == entities["omar"].entity_id
    assert result.fact_ids == expected
    assert {s.quote for s in result.sources} == {"supplies", "leads"}


def test_constraint_and_count_drops_delta_and_lists_support_facts():
    graph, entities, _ = cedar_graph()
    result = execute(
        form(
            {
                "op": "retrieve",
                "id": "s1",
                "subject": None,
                "predicate": "supplies",
                "object": entity_ref("Harbor Authority"),
            },
            {"op": "filter", "id": "s2", "input": "s1", "predicate": "founded_year", "cmp": "gt", "value": 2015},
            {"op": "count", "id": "s3", "input": "s2"},
            {"op": "output", "id": "s4", "inputs": ["s3"]},
        ),
        graph,
    )
    assert result.status == "answered"
    assert result.answer == (result.steps[2].bindings[0],)
    assert result.answer[0].value == 1
    assert tuple(b.value for b in result.steps[1].bindings) == (entities["cedar"].entity_id,)
    delta_facts = {f.fact_id for f in graph.facts if f.subject_id == entities["delta"].entity_id}
    assert delta_facts and not delta_facts & set(result.fact_ids)
    assert set(result.fact_ids) == {
        f.fact_id
        for f in graph.facts
        if f.subject_id == entities["cedar"].entity_id and f.predicate in {"supplies", "founded_year"}
    }


def test_count_of_empty_filter_is_traceable_and_answered_zero():
    graph, _, _ = cedar_graph()
    result = execute(
        form(
            {
                "op": "retrieve",
                "id": "s1",
                "subject": None,
                "predicate": "supplies",
                "object": entity_ref("Harbor Authority"),
            },
            {"op": "filter", "id": "s2", "input": "s1", "predicate": "founded_year", "cmp": "gt", "value": 2030},
            {"op": "count", "id": "s3", "input": "s2"},
            {"op": "output", "id": "s4", "inputs": ["s3"]},
        ),
        graph,
    )
    assert result.steps[1].status == "empty"
    assert result.steps[2].status == "ok"
    assert result.status == "answered" and result.answer[0].value == 0
    assert result.answer[0].fact_ids == result.steps[1].examined_fact_ids
    assert set(result.fact_ids) == set(result.steps[1].examined_fact_ids)


def test_one_property_conflict_is_reported_and_answer_is_empty():
    reef = ent("Blue Reef", "Organization")
    first, second = fact(reef, "founded_year", 2012), fact(reef, "founded_year", 2014)
    graph = make_kg([reef], [first, second])
    result = execute(
        form(
            {
                "op": "retrieve",
                "id": "s1",
                "subject": entity_ref("Blue Reef"),
                "predicate": "founded_year",
                "object": None,
            },
            {"op": "output", "id": "s2", "inputs": ["s1"]},
        ),
        graph,
    )
    assert result.status == "conflict"
    assert result.steps[0].status == "conflict"
    assert result.steps[0].conflicts[0].fact_ids == tuple(sorted((first.fact_id, second.fact_id)))
    assert result.answer == () and result.fact_ids == () and result.sources == ()


def test_missing_retrieve_and_unknown_entity_statuses_are_incomplete():
    graph, _, _ = cedar_graph()
    missing = execute(
        form(
            {
                "op": "retrieve",
                "id": "s1",
                "subject": entity_ref("Omar Nasser", "Person"),
                "predicate": "founded",
                "object": None,
            },
            {"op": "output", "id": "s2", "inputs": ["s1"]},
        ),
        graph,
    )
    assert missing.status == "incomplete" and missing.steps[0].status == "missing"
    assert missing.steps[0].missing == ((kentity_id_for("Omar Nasser", "Person"), "founded"),)
    assert missing.answer == ()
    unknown = execute(
        form(
            {
                "op": "retrieve",
                "id": "s1",
                "subject": entity_ref("No Such Organization"),
                "predicate": "supplies",
                "object": None,
            },
            {"op": "output", "id": "s2", "inputs": ["s1"]},
        ),
        graph,
    )
    assert unknown.status == "incomplete" and unknown.steps[0].status == "unknown_entity"
    assert unknown.answer == ()


def test_output_sources_deduplicate_shared_spans():
    cedar = ent("Cedar Labs", "Organization")
    harbor = ent("Harbor Authority", "Organization")
    shared = span("same source")
    graph = make_kg(
        [cedar, harbor], [fact(cedar, "supplies", harbor, [shared]), fact(cedar, "founded_year", 2020, [shared])]
    )
    result = execute(
        form(
            {
                "op": "retrieve",
                "id": "s1",
                "subject": None,
                "predicate": "supplies",
                "object": entity_ref("Harbor Authority"),
            },
            {"op": "filter", "id": "s2", "input": "s1", "predicate": "founded_year", "cmp": "gt", "value": 2015},
            {"op": "output", "id": "s3", "inputs": ["s2"]},
        ),
        graph,
    )
    assert result.sources == (shared,)
    assert len(result.sources) == 1


def test_shuffled_graph_construction_produces_equal_execution():
    graph, _, facts = cedar_graph()
    shuffled = make_kg(reversed(graph.entities), reversed(graph.facts), schema=graph.knowledge_schema)
    plan = form(
        {"op": "retrieve", "id": "s1", "subject": entity_ref("Cedar Labs"), "predicate": "supplies", "object": None},
        {"op": "retrieve", "id": "s2", "subject": None, "predicate": "leads", "object": {"step": "s1"}},
        {"op": "output", "id": "s3", "inputs": ["s2"]},
    )
    assert execute(plan, graph) == execute(plan, shuffled)
    assert all(f.fact_id == fact_id_for(f.subject_id, f.predicate, f.value) for f in facts)


def reject(data, schema=None):
    rejected = parse_logical_form(data, schema or make_schema())
    assert isinstance(rejected, PlanRejection)
    return rejected.reason


S1 = {"op": "retrieve", "id": "s1", "subject": entity_ref("Cedar Labs"), "predicate": "supplies", "object": None}


@pytest.mark.parametrize(
    "data",
    [
        None,
        {},
        {"steps": []},
        {"steps": [{}]},
        {"steps": [{"op": "count", "id": "s2", "input": "s1"}, {"op": "output", "id": "s2", "inputs": ["s1"]}]},
        {"steps": [{"op": "count", "id": "s1", "input": "s0"}, {"op": "output", "id": "s2", "inputs": ["s1"]}]},
        {"steps": [{"op": "count", "id": "s1", "input": "s1"}]},
        {"steps": [{"op": "output", "id": "s1", "inputs": []}]},
        # a valid s1 retrieve plus exactly one defect
        {"steps": [S1, {"op": "output", "id": "s2", "inputs": ["s1"]}, {"op": "output", "id": "s3", "inputs": ["s1"]}]},
        {"steps": [S1, {"op": "output", "id": "s2", "inputs": ["s1"]}, {"op": "count", "id": "s3", "input": "s1"}]},
        {"steps": [S1]},
        {"steps": [{**S1, "subject": {"step": "s2"}}, {"op": "output", "id": "s2", "inputs": ["s1"]}]},
        {"steps": [{**S1, "subject": {"step": "s1"}}, {"op": "output", "id": "s2", "inputs": ["s1"]}]},
        {"steps": [S1, {"op": "count", "id": "s2", "input": "s3"}, {"op": "output", "id": "s3", "inputs": ["s2"]}]},
        {"steps": [S1, {"op": "output", "id": "s2", "inputs": ["s1", "s1"]}]},
        {
            "steps": [
                S1,
                {"op": "sort", "id": "s2", "input": "s1", "predicate": "founded_year", "order": "asc", "limit": True},
                {"op": "output", "id": "s3", "inputs": ["s2"]},
            ]
        },
    ],
)
def test_invalid_logical_form_rejections(data):
    assert reject(data) == "invalid_logical_form"


@pytest.mark.parametrize("op", ["deduce", "math", "other"])
def test_unsupported_operators_are_rejected(op):
    assert (
        reject({"steps": [{"op": op, "id": "s1"}, {"op": "output", "id": "s2", "inputs": ["s1"]}]})
        == "unsupported_operator"
    )


@pytest.mark.parametrize(
    "data",
    [
        {
            "steps": [
                {
                    "op": "retrieve",
                    "id": "s1",
                    "subject": {"entity": {"name": "Cedar", "type": "Organization"}},
                    "predicate": "not_a_predicate",
                    "object": None,
                },
                {"op": "output", "id": "s2", "inputs": ["s1"]},
            ]
        },
        {
            "steps": [
                {
                    "op": "retrieve",
                    "id": "s1",
                    "subject": None,
                    "predicate": "supplies",
                    "object": entity_ref("Omar Nasser", "Person"),
                },
                {"op": "output", "id": "s2", "inputs": ["s1"]},
            ]
        },
        {
            "steps": [
                {
                    "op": "retrieve",
                    "id": "s1",
                    "subject": entity_ref("Cedar Labs"),
                    "predicate": "supplies",
                    "object": None,
                },
                {"op": "filter", "id": "s2", "input": "s1", "predicate": "supplies", "cmp": "eq", "value": 2},
                {"op": "output", "id": "s3", "inputs": ["s2"]},
            ]
        },
        {
            "steps": [
                {
                    "op": "retrieve",
                    "id": "s1",
                    "subject": None,
                    "predicate": "founded_year",
                    "object": entity_ref("Cedar Labs"),
                },
                {"op": "output", "id": "s2", "inputs": ["s1"]},
            ]
        },
        {
            "steps": [
                {
                    "op": "retrieve",
                    "id": "s1",
                    "subject": None,
                    "predicate": "supplies",
                    "object": entity_ref("Harbor Authority"),
                },
                {"op": "sort", "id": "s2", "input": "s1", "predicate": "supplies", "order": "asc", "limit": 3},
                {"op": "output", "id": "s3", "inputs": ["s2"]},
            ]
        },
    ],
)
def test_schema_violations_are_rejected(data):
    assert reject(data) == "schema_violation"


PROP_EXTRA = (
    PropertyDef(name="sector", value_kind="text", cardinality="one"),
    PropertyDef(name="tags", value_kind="text", cardinality="many"),
)
SUPPLIERS = {
    "op": "retrieve",
    "id": "s1",
    "subject": None,
    "predicate": "supplies",
    "object": entity_ref("Harbor Authority"),
}


def filter_steps(predicate, value, cmp="eq"):
    return [
        SUPPLIERS,
        {"op": "filter", "id": "s2", "input": "s1", "predicate": predicate, "cmp": cmp, "value": value},
        {"op": "output", "id": "s3", "inputs": ["s2"]},
    ]


@pytest.mark.parametrize(
    "predicate,value",
    [("founded_year", "2015"), ("founded_year", True), ("sector", 5), ("sector", False), ("tags", "x")],
)
def test_filter_value_kind_and_cardinality_checked_at_parse_time(predicate, value):
    data = {"steps": filter_steps(predicate, value)}
    assert reject(data, make_schema(PROP_EXTRA)) == "schema_violation"


def test_filter_with_matching_kind_parses():
    data = {"steps": filter_steps("sector", "energy")}
    assert isinstance(parse_logical_form(data, make_schema(PROP_EXTRA)), LogicalForm)


def test_sort_on_many_property_is_rejected():
    data = {
        "steps": [
            SUPPLIERS,
            {"op": "sort", "id": "s2", "input": "s1", "predicate": "tags", "order": "asc", "limit": 2},
            {"op": "output", "id": "s3", "inputs": ["s2"]},
        ]
    }
    assert reject(data, make_schema(PROP_EXTRA)) == "schema_violation"


def supplier_graph():
    harbor = ent("Harbor Authority", "Organization")
    orgs = {
        "cedar": ent("Cedar Labs", "Organization"),
        "delta": ent("Delta Works", "Organization"),
        "fjord": ent("Fjord Co", "Organization"),
        "echo": ent("Echo Inc", "Organization"),
    }
    facts = [fact(o, "supplies", harbor) for o in orgs.values()]
    facts += [
        fact(orgs["cedar"], "founded_year", 2020),
        fact(orgs["delta"], "founded_year", 2010),
        fact(orgs["fjord"], "founded_year", 2015),
    ]
    return make_kg([harbor, *orgs.values()], facts), orgs


def sort_plan(order, limit):
    return form(
        SUPPLIERS,
        {"op": "sort", "id": "s2", "input": "s1", "predicate": "founded_year", "order": order, "limit": limit},
        {"op": "output", "id": "s3", "inputs": ["s2"]},
    )


def test_sort_orders_limits_and_lists_missing_input():
    graph, orgs = supplier_graph()
    ids = {k: o.entity_id for k, o in orgs.items()}
    asc_run = execute(sort_plan("asc", 10), graph)
    asc = asc_run.steps[1]
    assert [b.value for b in asc.bindings] == [ids["delta"], ids["fjord"], ids["cedar"]]
    assert asc.status == "missing" and asc.missing == ((ids["echo"], "founded_year"),)
    assert asc_run.status == "incomplete"
    desc = execute(sort_plan("desc", 10), graph).steps[1]
    assert [b.value for b in desc.bindings] == [ids["cedar"], ids["fjord"], ids["delta"]]
    limited = execute(sort_plan("asc", 2), graph).steps[1]
    assert [b.value for b in limited.bindings] == [ids["delta"], ids["fjord"]]


def test_filter_drops_input_missing_property_and_run_is_incomplete():
    graph, orgs = supplier_graph()
    run = execute(form(*filter_steps("founded_year", 2012, "gt")), graph)
    flt = run.steps[1]
    assert [b.value for b in flt.bindings] == sorted([orgs["cedar"].entity_id, orgs["fjord"].entity_id])
    assert flt.status == "missing" and flt.missing == ((orgs["echo"].entity_id, "founded_year"),)
    assert run.status == "incomplete" and run.answer == ()


def reef_graph():
    harbor = ent("Harbor Authority", "Organization")
    reef = ent("Blue Reef", "Organization")
    cedar = ent("Cedar Labs", "Organization")
    facts = [
        fact(reef, "supplies", harbor),
        fact(cedar, "supplies", harbor),
        fact(reef, "founded_year", 2012),
        fact(reef, "founded_year", 2014),
        fact(cedar, "founded_year", 2020),
    ]
    return make_kg([harbor, reef, cedar], facts), reef, cedar


def test_filter_over_conflicting_input_is_dropped_and_listed():
    graph, reef, cedar = reef_graph()
    run = execute(form(*filter_steps("founded_year", 2000, "gt")), graph)
    flt = run.steps[1]
    assert [b.value for b in flt.bindings] == [cedar.entity_id]
    assert flt.status == "conflict"
    assert [(c.subject_id, c.predicate) for c in flt.conflicts] == [(reef.entity_id, "founded_year")]
    assert run.status == "conflict" and run.answer == ()


def test_conflict_and_missing_steps_in_one_run_report_conflict():
    graph, _, _ = reef_graph()
    omar = ent("Omar Nasser", "Person")
    graph = make_kg([*graph.entities, omar], graph.facts)
    run = execute(
        form(
            {
                "op": "retrieve",
                "id": "s1",
                "subject": entity_ref("Blue Reef"),
                "predicate": "founded_year",
                "object": None,
            },
            {
                "op": "retrieve",
                "id": "s2",
                "subject": entity_ref("Omar Nasser", "Person"),
                "predicate": "founded",
                "object": None,
            },
            {"op": "output", "id": "s3", "inputs": ["s1", "s2"]},
        ),
        graph,
    )
    assert {run.steps[0].status, run.steps[1].status} == {"conflict", "missing"}
    assert run.status == "conflict"


def test_who_leads_cedar_labs_reports_missing_pair():
    graph, entities, _ = cedar_graph()
    run = execute(
        form(
            {
                "op": "retrieve",
                "id": "s1",
                "subject": None,
                "predicate": "leads",
                "object": entity_ref("Cedar Labs"),
            },
            {"op": "output", "id": "s2", "inputs": ["s1"]},
        ),
        graph,
    )
    assert run.status == "incomplete" and run.steps[0].status == "missing"
    assert run.steps[0].missing == ((entities["cedar"].entity_id, "leads"),)
    assert run.answer == ()


def test_two_paths_to_one_entity_count_once_and_union_facts():
    a, b, z, root = (ent(n, "Organization") for n in ("A Co", "B Co", "Z Co", "Root Co"))
    facts = [fact(root, "supplies", a), fact(root, "supplies", b), fact(a, "supplies", z), fact(b, "supplies", z)]
    graph = make_kg([a, b, z, root], facts)
    run = execute(
        form(
            {"op": "retrieve", "id": "s1", "subject": entity_ref("Root Co"), "predicate": "supplies", "object": None},
            {"op": "retrieve", "id": "s2", "subject": {"step": "s1"}, "predicate": "supplies", "object": None},
            {"op": "count", "id": "s3", "input": "s2"},
            {"op": "output", "id": "s4", "inputs": ["s3"]},
        ),
        graph,
    )
    assert [x.value for x in run.steps[1].bindings] == [z.entity_id]
    assert len(run.steps[1].bindings[0].fact_ids) == 4
    assert run.answer[0].value == 1


def test_reverse_retrieve_does_not_check_subject_conflicts_on_the_object():
    schema = make_schema(
        relation_types=(
            RelationTypeDef(name="reports_to", domain="Organization", range="Organization", cardinality="one"),
        )
    )
    child, x, p, q = (ent(n, "Organization") for n in ("Child Co", "X Co", "P Co", "Q Co"))
    facts = [fact(child, "reports_to", x), fact(x, "reports_to", p), fact(x, "reports_to", q)]
    graph = make_kg([child, x, p, q], facts, schema=schema)
    assert graph.conflicts()
    run = execute(
        form(
            {"op": "retrieve", "id": "s1", "subject": None, "predicate": "reports_to", "object": entity_ref("X Co")},
            {"op": "output", "id": "s2", "inputs": ["s1"]},
            schema=schema,
        ),
        graph,
    )
    assert run.status == "answered" and [b.value for b in run.answer] == [child.entity_id]


def _shared_relation_schema(person_kind):
    from rag_foundry_domain.knowledge import EntityTypeDef, KnowledgeSchema

    return KnowledgeSchema(
        schema_id="shared",
        version="1",
        entity_types=(
            EntityTypeDef(
                name="Person", properties=(PropertyDef(name="x", value_kind=person_kind, cardinality="one"),)
            ),
            EntityTypeDef(name="Org", properties=(PropertyDef(name="x", value_kind="year", cardinality="one"),)),
        ),
        relation_types=(
            RelationTypeDef(name="member_of", domain="Person", range="Org", cardinality="many"),
            RelationTypeDef(name="member_of", domain="Org", range="Org", cardinality="many"),
        ),
    )


def _members_then(step, person_kind):
    steps = [
        {"op": "retrieve", "id": "s1", "subject": None, "predicate": "member_of", "object": entity_ref("Hub", "Org")},
        step,
        {"op": "output", "id": "s3", "inputs": ["s2"]},
    ]
    return parse_logical_form({"steps": steps}, _shared_relation_schema(person_kind))


def test_reverse_retrieve_on_shared_relation_name_tracks_every_domain():
    sort = {"op": "sort", "id": "s2", "input": "s1", "predicate": "x", "order": "asc", "limit": 5}
    mixed = _members_then(sort, "text")
    assert isinstance(mixed, PlanRejection) and mixed.reason == "schema_violation"
    year_filter = {"op": "filter", "id": "s2", "input": "s1", "predicate": "x", "cmp": "ge", "value": 2000}
    assert isinstance(_members_then(year_filter, "text"), PlanRejection)
    assert isinstance(_members_then(sort, "year"), LogicalForm)
    assert isinstance(_members_then(year_filter, "year"), LogicalForm)
