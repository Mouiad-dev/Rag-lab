import uuid

import pytest
from pydantic import ValidationError
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.graphs import (
    Entity,
    EntityGraph,
    GraphSpan,
    Relation,
    entity_id_for,
    graph_key_for,
    normalize_name,
    relation_id_for,
)
from rag_foundry_domain.testing import make_scope

NEW_CODES = (
    "graph_scope_mismatch",
    "graph_source_mismatch",
    "graph_invalid",
    "graph_exists",
    "graph_too_large",
    "graph_evidence_missing",
    "graph_key_mismatch",
    "untraced_claim",
    "community_cap_exceeded",
    "global_search_too_many_communities",
)
VERSION = str(uuid.uuid4())
SCOPE = make_scope()


def span(quote="x"):
    return GraphSpan(
        chunk_id=str(uuid.uuid4()), extraction_view_id=str(uuid.uuid4()), start=0, end=len(quote), quote=quote
    )


def entity(name):
    return Entity(entity_id=entity_id_for(name, "org"), name=name, type="org", mentions=(span("ab"),))


def relation(a, label, b):
    sid, tid = entity_id_for(a, "org"), entity_id_for(b, "org")
    return Relation(
        relation_id=relation_id_for(sid, label, tid), source_id=sid, target_id=tid, label=label, evidence=(span(),)
    )


def build(names, rels, **over):
    ents = sorted((entity(n) for n in names), key=lambda e: e.entity_id)
    rs = sorted((relation(*r) for r in rels), key=lambda r: r.relation_id)
    args = {
        "schema": "rag_foundry.entity_graph/1",
        "graph_key": graph_key_for({"a": 1}),
        "tenant_id": SCOPE.tenant_id,
        "project_id": SCOPE.project_id,
        "index_version_id": VERSION,
        "generator_json": "{}",
        "entities": tuple(ents),
        "relations": tuple(rs),
    }
    args.update(over)
    return EntityGraph(**args)


@pytest.mark.parametrize("code", NEW_CODES)
def test_new_codes_registered(code):
    assert IndexingError(code).code == code


def test_ids_stable_and_merge_case_and_space():
    assert entity_id_for("Cedar  Labs", "org") == entity_id_for("cedar labs", "Org")
    assert entity_id_for("a", "t").startswith("ent_") and len(entity_id_for("a", "t")) == 20
    assert relation_id_for("ent_" + "0" * 16, "Leads", "ent_" + "1" * 16).startswith("rel_")
    assert len(graph_key_for({"a": 1})) == 64
    assert normalize_name("e\u0301") == normalize_name("\u00e9")
    arabic_decomposed = "\u0627\u0654"
    assert normalize_name(arabic_decomposed) == normalize_name("\u0623")


def test_validation_errors():
    with pytest.raises(ValidationError):
        GraphSpan(chunk_id=str(uuid.uuid4()), extraction_view_id=str(uuid.uuid4()), start=0, end=3, quote="ab")
    with pytest.raises(ValidationError):
        build(["A"], [("A", "leads", "B")])
    e = entity("A")
    with pytest.raises(ValidationError):
        build(["A"], [], entities=(e, e))


def test_json_round_trip_and_scope():
    g = build(["A", "B"], [("A", "leads", "B")])
    assert EntityGraph.from_json(g.to_json()) == g
    g.check_scope(SCOPE, VERSION)
    with pytest.raises(IndexingError) as exc:
        g.check_scope(make_scope(), VERSION)
    assert exc.value.code == "graph_scope_mismatch"
    with pytest.raises(IndexingError):
        g.check_scope(SCOPE, str(uuid.uuid4()))
    with pytest.raises(IndexingError) as exc:
        EntityGraph.from_json(g.to_json().replace("entity_graph/1", "entity_graph/9"))
    assert exc.value.code == "graph_invalid"


def test_shortest_path_reverse_cutoff_and_tie():
    g = build(["Omar Nasser", "Harbor Authority"], [("Omar Nasser", "leads", "Harbor Authority")])
    ha, on = entity_id_for("Harbor Authority", "org"), entity_id_for("Omar Nasser", "org")
    rid = relation_id_for(on, "leads", ha)
    assert g.shortest_path(ha, on, 3) == ((rid, "reverse"),)
    assert g.shortest_path(on, ha, 3) == ((rid, "forward"),)
    chain = build(["A", "B", "C"], [("A", "x", "B"), ("B", "x", "C")])
    a, c = entity_id_for("A", "org"), entity_id_for("C", "org")
    assert chain.shortest_path(a, c, 1) is None
    assert len(chain.shortest_path(a, c, 2)) == 2
    tie = build(["S", "M1", "M2", "T"], [("S", "x", "M1"), ("S", "y", "M2"), ("M1", "x", "T"), ("M2", "y", "T")])
    s, t = entity_id_for("S", "org"), entity_id_for("T", "org")
    m1, m2 = entity_id_for("M1", "org"), entity_id_for("M2", "org")
    via_m1 = (relation_id_for(s, "x", m1), relation_id_for(m1, "x", t))
    via_m2 = (relation_id_for(s, "y", m2), relation_id_for(m2, "y", t))
    expected = tuple((rid, "forward") for rid in min(via_m1, via_m2, key=lambda p: p[0]))
    assert tie.shortest_path(s, t, 4) == expected  # tie broken by the smaller first relation ID


def test_unsorted_entities_rejected():
    g = build(["A", "B", "C"], [])
    with pytest.raises(ValidationError):
        build(["A", "B", "C"], [], entities=tuple(reversed(g.entities)))


def test_unknown_entity_has_no_path_even_to_itself():
    g = build(["A"], [])
    assert g.shortest_path("ent_" + "f" * 16, "ent_" + "f" * 16, 3) is None
