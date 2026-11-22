"""Shared fixture for the R48 KAG tests (P7.3-06 to P7.3-10).

Chunking rule (relied on by 08b): every filler paragraph is at least as long as the longest fact paragraph and
no two fact paragraphs are adjacent. A RecursiveChunker with overlap 0 and a size at least the longest
paragraph but smaller than any adjacent pair plus "\\n\\n" therefore yields exactly one chunk per paragraph.

Fact paragraphs K0..K8 (``LABELS``). K4/K5 give the Blue Reef founding-year conflict (used by no count or
multi-hop question). K6 and K7 are scripted extraction misses: K6 has a fact quote that is not in the chunk,
K7 has a fact whose subject is not in its quote. The scripted replies also carry one bad item per rejection
reason (except ``too_many_spans``) that must never reach the graph.
"""

import json
from dataclasses import dataclass

from rag_foundry_domain.knowledge import EntityTypeDef, KnowledgeSchema, PropertyDef, RelationTypeDef
from rag_foundry_indexing import KagExtractConfig, KagGraphBuilder
from rag_foundry_indexing.context_expansion import SourceBlockRef, ViewSource

from tests.chunking._scripted_models import ScriptedChat, ScriptedEmbedder
from tests.indexing._inmemory_index import ready_index
from tests.indexing._source_reader import StaticSourceReader

SCHEMA = KnowledgeSchema(
    schema_id="demo",
    version="1",
    entity_types=(
        EntityTypeDef(name="Person"),
        EntityTypeDef(
            name="Organization",
            properties=(PropertyDef(name="founded_year", value_kind="year", cardinality="one"),),
        ),
    ),
    relation_types=(
        RelationTypeDef(name="founded", domain="Person", range="Organization", cardinality="many"),
        RelationTypeDef(name="supplies", domain="Organization", range="Organization", cardinality="many"),
        RelationTypeDef(name="leads", domain="Person", range="Organization", cardinality="one"),
    ),
)

K0 = "Lina Haddad founded Cedar Labs in 2019."
K1 = "Cedar Labs supplies tide sensors to the Harbor Authority."
K2 = "Omar Nasser leads the Harbor Authority."
K3 = "Delta Works, founded in 2010, also supplies the Harbor Authority."
K4 = "Blue Reef was founded in 2012."
K5 = "Blue Reef was founded in 2014."
K6 = "Nour Saleh leads Delta Works."
K7 = "Gulf Marine also supplies the Harbor Authority."
AR_ORG = "\u0634\u0631\u0643\u0629 \u0627\u0644\u0646\u0648\u0631"
K8 = AR_ORG + " \u062a\u0623\u0633\u0633\u062a \u0639\u0627\u0645 2015."

_FILLER = (
    "The weather station logged steady rainfall over the northern valley today.",
    "A small bakery on the corner sold out of cardamom bread before noon today.",
    "The regional library extended its opening hours throughout the winter term.",
    "Volunteers repainted the old footbridge over the quiet river last weekend.",
    "The orchestra rehearsed a new symphony in the restored concert hall tonight.",
    "A freight train carrying grain crossed the plateau just after sunrise today.",
    "The museum catalogued a collection of ceramic tiles from the coastal villages.",
    "Local farmers reported an early harvest of olives across the southern terraces.",
    "The city council approved extra lighting for the cycling paths along the canal.",
)

_FACTS = (K0, K1, K2, K3, K4, K5, K6, K7, K8)
LABELS = {f"K{i}": text for i, text in enumerate(_FACTS)}
PARAGRAPHS = tuple(p for pair in zip(_FACTS, _FILLER, strict=True) for p in pair)
assert len(PARAGRAPHS) == 18
assert min(len(p) for p in _FILLER) >= max(len(p) for p in _FACTS)
TEXT = "\n\n".join(PARAGRAPHS)


def _spans():
    out, pos = [], 0
    for paragraph in PARAGRAPHS:
        out.append((pos, pos + len(paragraph), paragraph))
        pos += len(paragraph) + 2
    return tuple(out)


SPANS = _spans()
PARAGRAPH_SPAN = {text: (a, b) for a, b, text in SPANS}


def span_of(label):
    return PARAGRAPH_SPAN[LABELS[label]]


def _ent(name, type_, quote=None):
    return {"name": name, "type": type_, "quote": quote or name}


def _rel(subject, predicate, obj, quote):
    return {"subject": subject, "predicate": predicate, "object": obj, "quote": quote}


def _val(subject, predicate, value, quote):
    return {"subject": subject, "predicate": predicate, "value": value, "quote": quote}


def _reply(entities=(), facts=()):
    return json.dumps({"entities": list(entities), "facts": list(facts)}, ensure_ascii=False)


KAG_REPLIES = {
    K0: _reply(
        [
            _ent("Lina Haddad", "Person"),
            _ent("Cedar Labs", "Organization"),
            _ent("Cedar", "Product", "Cedar Labs"),  # unknown_type
        ],
        [
            _rel("Lina Haddad", "founded", "Cedar Labs", K0),
            _val("Cedar Labs", "founded_year", 2019, "Cedar Labs in 2019"),
            _rel("Lina Haddad", "invented", "Cedar Labs", K0),  # unknown_predicate
        ],
    ),
    K1: _reply(
        [
            _ent("Cedar Labs", "Organization"),
            _ent("Harbor Authority", "Organization"),
            _ent("Cedar Labs", "Organization", "s"),  # ambiguous_quote
        ],
        [
            _rel("Cedar Labs", "supplies", "Harbor Authority", K1),
            _rel("Cedar Labs", "supplies", "tide sensors", K1),  # unknown_object
        ],
    ),
    K2: _reply(
        [_ent("Omar Nasser", "Person"), _ent("Harbor Authority", "Organization")],
        [
            _rel("Omar Nasser", "leads", "Harbor Authority", K2),
            _rel("Omar Nasser", "leads", "Omar Nasser", K2),  # range_violation
        ],
    ),
    K3: _reply(
        [_ent("Delta Works", "Organization"), _ent("Harbor Authority", "Organization")],
        [
            _rel("Delta Works", "supplies", "Harbor Authority", K3),
            _val("Delta Works", "founded_year", 2010, "Delta Works, founded in 2010"),
            _val("Delta Works", "founded_year", 2011, "Delta Works, founded in 2010"),  # value_not_in_quote
        ],
    ),
    K4: _reply(
        [_ent("Blue Reef", "Organization")],
        [
            _val("Blue Reef", "founded_year", 2012, K4),
            _val("Blue Reef", "founded_year", "twenty twelve", K4),  # bad_literal
        ],
    ),
    K5: _reply(
        [_ent("Blue Reef", "Organization")],
        [
            _val("Blue Reef", "founded_year", 2014, K5),
            _val("Blue", "founded_year", 2014, K5),  # unknown_subject
        ],
    ),
    K6: _reply(
        [_ent("Nour Saleh", "Person"), _ent("Delta Works", "Organization")],
        [_rel("Nour Saleh", "leads", "Delta Works", "Nour Saleh is the head of Delta Works")],  # quote_not_found
    ),
    K7: _reply(
        [_ent("Gulf Marine", "Organization"), _ent("Harbor Authority", "Organization")],
        [_rel("Delta Works", "supplies", "Harbor Authority", K7)],  # name_not_in_quote
    ),
    K8: _reply(
        [_ent(AR_ORG, "Organization")] * 10 + [_ent("Extra", "Organization", K8)],  # 11th: too_many_items
        [_val(AR_ORG, "founded_year", 2015, K8[:-1])],
    ),
}

# The scripted rejections, by reason, with the chunk text that must carry each one.
EXPECTED_REASONS = {
    "unknown_type": K0,
    "unknown_predicate": K0,
    "ambiguous_quote": K1,
    "unknown_object": K1,
    "range_violation": K2,
    "value_not_in_quote": K3,
    "bad_literal": K4,
    "unknown_subject": K5,
    "quote_not_found": K6,
    "name_not_in_quote": K7,
    "too_many_items": K8,
}


def _r38(entities, relations=()):
    rels = [{"source": s, "target": t, "label": label, "quote": q} for s, t, label, q in relations]
    return json.dumps({"entities": list(entities), "relations": rels}, ensure_ascii=False)


R38_REPLIES = {
    K0: _r38(
        [_ent("Lina Haddad", "person"), _ent("Cedar Labs", "organization")],
        [("Lina Haddad", "Cedar Labs", "founded", K0)],
    ),
    K1: _r38(
        [_ent("Cedar Labs", "organization"), _ent("Harbor Authority", "organization")],
        [("Cedar Labs", "Harbor Authority", "supplies sensors to", K1)],
    ),
    K2: _r38(
        [_ent("Omar Nasser", "person"), _ent("Harbor Authority", "organization")],
        [("Omar Nasser", "Harbor Authority", "leads", K2)],
    ),
    K3: _r38(
        [_ent("Delta Works", "organization"), _ent("Harbor Authority", "organization")],
        [("Delta Works", "Harbor Authority", "supplies", K3)],
    ),
    K4: _r38([_ent("Blue Reef", "organization")]),
    K5: _r38([_ent("Blue Reef", "organization")]),
    K6: _r38([_ent("Nour Saleh", "person"), _ent("Delta Works", "organization")]),
    K7: _r38([_ent("Gulf Marine", "organization"), _ent("Harbor Authority", "organization")]),
    K8: _r38([_ent(AR_ORG, "organization")]),
}

# Gold evidence spans are whole-paragraph (start, end) offsets in TEXT. Q2 is a losing example: the true count
# is 3 (K1, K3, K7), but the scripted extraction misses K7, so the graph holds only two supplier facts.
QUESTIONS = (
    {
        "id": "Q0",
        "kind": "lookup",
        "text": "Who founded Cedar Labs?",
        "answer": "Lina Haddad",
        "gold": (span_of("K0"),),
    },
    {
        "id": "Q1",
        "kind": "lookup",
        "text": "In which year was Cedar Labs founded?",
        "answer": 2019,
        "gold": (span_of("K0"),),
    },
    {
        "id": "Q2",
        "kind": "count",
        "text": "How many organizations supply the Harbor Authority?",
        "answer": 3,
        "graph_answer": 2,
        "gold": (span_of("K1"), span_of("K3"), span_of("K7")),
    },
    {
        "id": "Q3",
        "kind": "multi_hop",
        "text": "Who leads the organization that Cedar Labs supplies?",
        "answer": "Omar Nasser",
        "gold": (span_of("K1"), span_of("K2")),
    },
    {
        "id": "Q4",
        "kind": "lookup",
        "text": "Who leads Delta Works?",
        "answer": "Nour Saleh",
        "graph_answer": None,
        "gold": (span_of("K6"),),
    },
)


@dataclass
class KagFixture:
    scope: object
    stranger: object
    index: object
    version: object
    graph: object
    report: object
    chat: object
    source: object
    view_id: str
    chunks: list
    builder: object


def make_chat(replies):
    def respond(request):
        return replies.get(json.loads(request.messages[1].content)["chunk"], _reply()), "stop"

    return ScriptedChat(respond)


async def build_fixture_kg(*, spans=SPANS, replies=None, config=None, schema=None, text=None, chat=None):
    replies = KAG_REPLIES if replies is None else replies
    text = TEXT if text is None else text
    embedder = ScriptedEmbedder(lambda t: (1.0, 0.0, 0.0, 0.0))
    scope, stranger, index, version = await ready_index(spans, embedder)
    chunks = await index.list_chunks(scope, version.id, offset=0, limit=500)
    view_id = chunks[0].extraction_view_id
    view = ViewSource(
        extraction_view_id=view_id, text=text, blocks=(SourceBlockRef(block_id="b00000", start=0, end=len(text)),)
    )
    source = StaticSourceReader({(scope.tenant_id, scope.project_id, view_id): view})
    chat = chat or make_chat(replies)
    builder = KagGraphBuilder(
        config or KagExtractConfig(),
        scope=scope,
        index_version=version,
        index=index,
        source=source,
        chat=chat,
        knowledge_schema=schema or SCHEMA,
    )
    report = await builder.build()
    graph = await index.get_knowledge_graph(scope, version.id, report.kg_key)
    return KagFixture(scope, stranger, index, version, graph, report, chat, source, view_id, chunks, builder)
