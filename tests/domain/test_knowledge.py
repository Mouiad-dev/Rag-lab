import json
import uuid

import pytest
from pydantic import ValidationError
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.graphs import GraphSpan
from rag_foundry_domain.knowledge import (
    EntityTypeDef,
    Fact,
    KEntity,
    KnowledgeGraph,
    KnowledgeSchema,
    PropertyDef,
    RelationTypeDef,
    fact_id_for,
    kentity_id_for,
    kg_key_for,
)
from rag_foundry_domain.testing import make_scope

VERSION = str(uuid.uuid4())
SCOPE = make_scope()
VIEW = str(uuid.uuid4())
ARABIC = "".join(chr(c) for c in (0x0634, 0x0631, 0x0643, 0x0629, 0x20, 0x0627, 0x0644, 0x0646, 0x0648, 0x0631))


def make_schema(extra_props=(), **over):
    args = {
        "schema_id": "demo",
        "version": "1",
        "entity_types": (
            EntityTypeDef(name="Person"),
            EntityTypeDef(
                name="Organization",
                properties=(PropertyDef(name="founded_year", value_kind="year", cardinality="one"), *extra_props),
            ),
        ),
        "relation_types": (
            RelationTypeDef(name="founded", domain="Person", range="Organization", cardinality="many"),
            RelationTypeDef(name="supplies", domain="Organization", range="Organization", cardinality="many"),
            RelationTypeDef(name="leads", domain="Person", range="Organization", cardinality="one"),
        ),
    }
    args.update(over)
    return KnowledgeSchema(**args)


def span(quote="x", start=0, chunk_id=None):
    return GraphSpan(
        chunk_id=chunk_id or str(uuid.uuid4()),
        extraction_view_id=VIEW,
        start=start,
        end=start + len(quote),
        quote=quote,
    )


def ent(name, type, sp=None):
    return KEntity(entity_id=kentity_id_for(name, type), type=type, name=name, mentions=(sp or span(),))


def fact(subject, predicate, value, spans=None, **over):
    kw = {"object_id": value.entity_id} if isinstance(value, KEntity) else {"literal": value}
    key = value.entity_id if isinstance(value, KEntity) else value
    args = {
        "fact_id": fact_id_for(subject.entity_id, predicate, key),
        "subject_id": subject.entity_id,
        "predicate": predicate,
        "evidence": tuple(spans or (span(),)),
        **kw,
    }
    args.update(over)
    return Fact(**args)


def make_kg(entities=(), facts=(), schema=None, **over):
    args = {
        "schema": "rag_foundry.knowledge_graph/1",
        "kg_key": kg_key_for({"a": 1}),
        "tenant_id": SCOPE.tenant_id,
        "project_id": SCOPE.project_id,
        "index_version_id": VERSION,
        "knowledge_schema": schema or make_schema(),
        "generator_json": "{}",
        "entities": tuple(sorted(entities, key=lambda e: e.entity_id)),
        "facts": tuple(sorted(facts, key=lambda f: f.fact_id)),
    }
    args.update(over)
    return KnowledgeGraph(**args)


def basic():
    ana, acme, beta = ent("Ana", "Person"), ent("Acme", "Organization"), ent("Beta", "Organization")
    facts = [
        fact(acme, "founded_year", 1999),
        fact(ana, "founded", acme),
        fact(acme, "supplies", beta),
    ]
    return make_kg([ana, acme, beta], facts), ana, acme, beta


def code(exc_info):
    return exc_info.value.code


def test_new_codes_registered():
    for c in ("kag_schema_invalid", "kag_scope_mismatch", "kag_invalid"):
        assert IndexingError(c).code == c


# --- schema ---
def test_schema_ok_hash_and_predicate():
    s = make_schema()
    assert len(s.schema_sha256()) == 64 and s.schema_sha256() == make_schema().schema_sha256()
    assert s.predicate("Organization", "founded_year").value_kind == "year"
    assert s.predicate("Person", "leads").cardinality == "one"
    assert s.predicate("Person", "founded_year") is None
    assert s.predicate("Nope", "leads") is None


def test_schema_unknown_range_and_domain():
    bad = (RelationTypeDef(name="owns", domain="Person", range="Ghost", cardinality="one"),)
    with pytest.raises(IndexingError) as e:
        make_schema(relation_types=bad)
    assert code(e) == "kag_schema_invalid"
    bad = (RelationTypeDef(name="owns", domain="Ghost", range="Person", cardinality="one"),)
    with pytest.raises(IndexingError) as e:
        make_schema(relation_types=bad)
    assert code(e) == "kag_schema_invalid"


def test_schema_duplicate_names():
    with pytest.raises(IndexingError) as e:  # property and relation share a name on one type
        make_schema(extra_props=(PropertyDef(name="supplies", value_kind="text", cardinality="many"),))
    assert code(e) == "kag_schema_invalid"
    with pytest.raises(IndexingError) as e:  # duplicate property
        make_schema(extra_props=(PropertyDef(name="founded_year", value_kind="int", cardinality="one"),))
    assert code(e) == "kag_schema_invalid"
    with pytest.raises(IndexingError) as e:  # duplicate entity type
        make_schema(entity_types=(EntityTypeDef(name="Person"), EntityTypeDef(name="Person")), relation_types=())
    assert code(e) == "kag_schema_invalid"


def test_schema_name_patterns():
    with pytest.raises(ValidationError):
        PropertyDef(name="Bad", value_kind="text", cardinality="one")
    with pytest.raises(ValidationError):
        EntityTypeDef(name="lower")


def test_schema_canonical_json_cap():
    types = tuple(
        EntityTypeDef(
            name=f"T{i}",
            properties=tuple(PropertyDef(name=f"p{j}", value_kind="text", cardinality="one") for j in range(40)),
        )
        for i in range(60)
    )
    with pytest.raises(IndexingError) as e:
        make_schema(entity_types=types, relation_types=())
    assert code(e) == "kag_schema_invalid"
    make_schema(entity_types=types[:10], relation_types=())  # under the cap is fine


# --- facts and graph validation ---
def test_valid_graph_and_sorted_ids():
    kg, *_ = basic()
    assert len(kg.facts) == 3
    ents = sorted(kg.entities, key=lambda e: e.entity_id, reverse=True)
    with pytest.raises(IndexingError) as e:
        KnowledgeGraph.model_validate({**kg.model_dump(by_alias=True), "entities": tuple(ents)})
    assert code(e) == "kag_invalid"


def test_duplicate_ids_rejected():
    ana = ent("Ana", "Person")
    with pytest.raises(IndexingError) as e:
        make_kg([ana, ana])
    assert code(e) == "kag_invalid"


def test_domain_violation():
    _, ana, acme, beta = basic()
    f = fact(acme, "founded", beta)  # founded domain is Person
    with pytest.raises(IndexingError) as e:
        make_kg([ana, acme, beta], [f])
    assert code(e) == "kag_invalid"


def test_range_violation():
    ana, acme = ent("Ana", "Person"), ent("Acme", "Organization")
    f = fact(ana, "founded", ent("Bob", "Person"))
    with pytest.raises(IndexingError) as e:
        make_kg([ana, acme, ent("Bob", "Person")], [f])
    assert code(e) == "kag_invalid"


def test_unknown_subject_object_and_predicate():
    ana, acme = ent("Ana", "Person"), ent("Acme", "Organization")
    with pytest.raises(IndexingError) as e:
        make_kg([acme], [fact(ana, "founded", acme)])  # subject missing
    assert code(e) == "kag_invalid"
    with pytest.raises(IndexingError) as e:
        make_kg([ana], [fact(ana, "founded", acme)])  # object missing
    assert code(e) == "kag_invalid"
    with pytest.raises(IndexingError) as e:
        make_kg([ana], [fact(ana, "nickname", "A")])  # undeclared predicate
    assert code(e) == "kag_invalid"


@pytest.mark.parametrize("value", ["text", 999, 3000, True])
def test_bad_year_literal(value):
    acme = ent("Acme", "Organization")
    with pytest.raises((IndexingError, ValidationError)):
        make_kg([acme], [fact(acme, "founded_year", value)])


def test_bad_literal_kinds():
    acme = ent("Acme", "Organization")
    s_int = make_schema(extra_props=(PropertyDef(name="staff", value_kind="int", cardinality="one"),))
    s_txt = make_schema(extra_props=(PropertyDef(name="motto", value_kind="text", cardinality="one"),))
    with pytest.raises(IndexingError) as e:
        make_kg([acme], [fact(acme, "staff", "many")], schema=s_int)
    assert code(e) == "kag_invalid"
    with pytest.raises(IndexingError) as e:
        make_kg([acme], [fact(acme, "motto", 5)], schema=s_txt)
    assert code(e) == "kag_invalid"
    make_kg([acme], [fact(acme, "staff", -5)], schema=s_int)


def test_int_literal_bigint_bounds():
    acme = ent("Acme", "Organization")
    s_int = make_schema(extra_props=(PropertyDef(name="staff", value_kind="int", cardinality="one"),))
    make_kg([acme], [fact(acme, "staff", 2**63 - 1)], schema=s_int)
    make_kg([acme], [fact(acme, "staff", -(2**63) + 1)], schema=s_int)
    for big in (2**63, -(2**63), 2**70):
        with pytest.raises(IndexingError) as e:
            fact(acme, "staff", big)
        assert code(e) == "kag_invalid"


def test_object_and_literal_exclusivity():
    ana, acme = ent("Ana", "Person"), ent("Acme", "Organization")
    base = {"fact_id": "fct_" + "0" * 16, "subject_id": ana.entity_id, "predicate": "founded", "evidence": (span(),)}
    with pytest.raises(IndexingError) as e:  # both
        Fact(**base, object_id=acme.entity_id, literal="x")
    assert code(e) == "kag_invalid"
    with pytest.raises(IndexingError) as e:  # neither
        Fact(**base)
    assert code(e) == "kag_invalid"


def test_property_fact_with_object_id_and_relation_fact_with_literal():
    ana, acme = ent("Ana", "Person"), ent("Acme", "Organization")
    prop_with_obj = fact(acme, "founded_year", acme)
    with pytest.raises(IndexingError) as e:
        make_kg([acme], [prop_with_obj])
    assert code(e) == "kag_invalid"
    rel_with_lit = fact(ana, "founded", "Acme")
    with pytest.raises(IndexingError) as e:
        make_kg([ana, acme], [rel_with_lit])
    assert code(e) == "kag_invalid"


def test_entity_type_must_be_declared():
    with pytest.raises(IndexingError) as e:
        make_kg([ent("Zed", "Robot")])
    assert code(e) == "kag_invalid"


# --- json, scope, lookup ---
def test_json_round_trip():
    kg, *_ = basic()
    again = KnowledgeGraph.from_json(kg.to_json())
    assert again == kg and again.to_json() == kg.to_json()


def test_from_json_errors():
    kg, *_ = basic()
    for bad in ("not json", "[]", json.dumps({"schema": "other"})):
        with pytest.raises(IndexingError) as e:
            KnowledgeGraph.from_json(bad)
        assert code(e) == "kag_invalid"
    data = json.loads(kg.to_json())
    data["facts"][0]["subject_id"] = "bad"
    with pytest.raises(IndexingError) as e:
        KnowledgeGraph.from_json(json.dumps(data))
    assert code(e) == "kag_invalid"


def test_scope_checks():
    kg, *_ = basic()
    kg.check_scope(SCOPE, VERSION)
    with pytest.raises(IndexingError) as e:
        kg.check_scope(make_scope(), VERSION)
    assert code(e) == "kag_scope_mismatch"
    with pytest.raises(IndexingError) as e:
        kg.check_scope(SCOPE, str(uuid.uuid4()))
    assert code(e) == "kag_scope_mismatch"


def test_arabic_entity_by_name():
    org = ent(ARABIC, "Organization")
    kg = make_kg([org])
    assert kg.entity_by_name("  " + ARABIC + " ") == (org,)
    assert kg.entity_by_name(ARABIC, "Person") == ()


def test_entity_by_name_case_insensitive():
    kg, ana, *_ = basic()
    assert kg.entity_by_name("ANA", "Person") == (ana,)
    assert kg.entity_by_name("nobody") == ()


def test_facts_about_subject_object_and_predicate():
    kg, ana, acme, beta = basic()
    assert {f.predicate for f in kg.facts_about(acme.entity_id)} == {"founded_year", "supplies"}
    assert [f.predicate for f in kg.facts_about(acme.entity_id, "supplies")] == ["supplies"]
    assert {f.predicate for f in kg.facts_about(acme.entity_id, as_object=True)} == {"founded"}
    assert [f.subject_id for f in kg.facts_about(beta.entity_id, as_object=True)] == [acme.entity_id]


def test_mutual_chunk_fact_index():
    ana, acme = ent("Ana", "Person"), ent("Acme", "Organization")
    s1, s2, s3 = span("aa"), span("bb", 5), span("cc", 9)
    f1 = fact(ana, "founded", acme, [s1, s2])
    f2 = fact(acme, "founded_year", 2001, [s2])
    f3 = fact(ana, "leads", acme, [s3])
    kg = make_kg([ana, acme], [f1, f2, f3])
    assert kg.chunks_for_fact(f1.fact_id) == tuple(sorted({s1.chunk_id, s2.chunk_id}))
    assert kg.chunks_for_fact("fct_" + "f" * 16) == ()
    ids = [f.fact_id for f in kg.facts_for_chunk(s2.chunk_id)]
    assert ids == sorted([f1.fact_id, f2.fact_id])
    for f in kg.facts:
        for c in kg.chunks_for_fact(f.fact_id):
            assert f in kg.facts_for_chunk(c)
    assert kg.facts_for_chunk(str(uuid.uuid4())) == ()


# --- conflicts ---
def test_conflict_reported_not_resolved():
    reef = ent("Blue Reef", "Organization")
    f1, f2 = fact(reef, "founded_year", 2012), fact(reef, "founded_year", 2014)
    kg = make_kg([reef], [f1, f2])
    assert len(kg.facts) == 2  # both kept
    (c,) = kg.conflicts()
    assert c.subject_id == reef.entity_id and c.predicate == "founded_year"
    assert c.fact_ids == tuple(sorted([f1.fact_id, f2.fact_id]))


def test_relation_one_conflict_and_many_is_not():
    ana = ent("Ana", "Person")
    a, b = ent("Acme", "Organization"), ent("Beta", "Organization")
    kg = make_kg(
        [ana, a, b], [fact(ana, "leads", a), fact(ana, "leads", b), fact(ana, "founded", a), fact(ana, "founded", b)]
    )
    assert [c.predicate for c in kg.conflicts()] == ["leads"]


def test_same_year_from_two_spans_is_one_fact():
    acme = ent("Acme Two", "Organization")
    kg = make_kg([acme], [fact(acme, "founded_year", 2000, [span("one"), span("two", 4)])])
    assert len(kg.facts) == 1 and len(kg.facts[0].evidence) == 2
    assert kg.conflicts() == ()
    twin = fact(acme, "founded_year", 2000)
    with pytest.raises(IndexingError) as e:  # a duplicate fact id is not silently merged
        make_kg([acme], [twin, twin])
    assert code(e) == "kag_invalid"


def test_no_conflicts_in_basic_graph():
    kg, *_ = basic()
    assert kg.conflicts() == ()
