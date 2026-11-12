"""Shared relationship fixture for the R38 graph tests (P7.2-05 to P7.2-09).

Chunking rule (relied on by P7.2-06): every filler paragraph is at least as long as the longest fact
paragraph. A RecursiveChunker with overlap 0 and a size at least the longest paragraph but smaller than any
adjacent pair plus "\\n\\n" therefore yields exactly one chunk per paragraph.

R21 gap: the three evidence paragraphs are P0, P7 and P14. A 5-sentence window around the seeds P0 and P14
covers P0-P5 and P9-P15, so the 7-paragraph distance between P0 and P7 leaves P7 uncovered. The 3-hop answer
needs P7, which only the relation path reaches.
"""

import json
from dataclasses import dataclass

from rag_foundry_indexing import EntityRelationExtractConfig, EntityRelationGraphBuilder
from rag_foundry_indexing.context_expansion import SourceBlockRef, ViewSource
from rag_foundry_techniques import RetrievalResult, RetrievedChunk

from tests.chunking._scripted_models import ScriptedChat, ScriptedEmbedder
from tests.indexing._inmemory_index import ready_index
from tests.indexing._source_reader import StaticSourceReader

P0 = "Lina Haddad founded Cedar Labs in 2019."
P7 = "Cedar Labs supplies tide sensors to the Harbor Authority."
P14 = "Omar Nasser leads the Harbor Authority."
P15 = (
    "\u0633\u0627\u0645\u064a \u064a\u0639\u0645\u0644 \u0641\u064a \u0645\u064a\u0646\u0627\u0621 \u0635\u0648\u0631."
)
AR_PERSON = "\u0633\u0627\u0645\u064a"
AR_PORT = "\u0645\u064a\u0646\u0627\u0621 \u0635\u0648\u0631"

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
    "A travelling theatre group staged two comedies in the market square this month.",
    "The observatory published a calendar of meteor showers for the coming season.",
    "Students from three schools planted young cedar trees beside the football field.",
)

PARAGRAPHS = (P0, *_FILLER[:6], P7, *_FILLER[6:], P14, P15)
assert len(PARAGRAPHS) == 16
assert min(len(p) for i, p in enumerate(PARAGRAPHS) if i not in (0, 7, 14, 15)) >= max(
    len(P0), len(P7), len(P14), len(P15)
)
TEXT = "\n\n".join(PARAGRAPHS)


def _spans():
    out, pos = [], 0
    for paragraph in PARAGRAPHS:
        out.append((pos, pos + len(paragraph), paragraph))
        pos += len(paragraph) + 2
    return tuple(out)


SPANS = _spans()
PARAGRAPH_START = {i: SPANS[i][0] for i in range(len(SPANS))}

QUESTION = "How is Lina Haddad connected to Omar Nasser?"
NAMES = ("Lina Haddad", "Omar Nasser")
GOLD_SPANS = ((SPANS[0][0], SPANS[0][1]), (SPANS[7][0], SPANS[7][1]), (SPANS[14][0], SPANS[14][1]))


def _ent(name, type_, quote=None):
    return {"name": name, "type": type_, "quote": quote or name}


def _rel(source, target, label, quote):
    return {"source": source, "target": target, "label": label, "quote": quote}


def _reply(entities=(), relations=()):
    return json.dumps({"entities": list(entities), "relations": list(relations)}, ensure_ascii=False)


REPLIES = {
    P0: _reply(
        [_ent("Lina Haddad", "person"), _ent("Cedar Labs", "organization")],
        [_rel("Lina Haddad", "Cedar Labs", "founded", P0)],
    ),
    P7: _reply(
        [_ent("Cedar Labs", "organization"), _ent("Harbor Authority", "organization")],
        [_rel("Cedar Labs", "Harbor Authority", "supplies sensors to", P7)],
    ),
    P14: _reply(
        [_ent("Omar Nasser", "person"), _ent("Harbor Authority", "organization")],
        [_rel("Omar Nasser", "Harbor Authority", "leads", P14)],
    ),
    P15: _reply(
        [_ent(AR_PERSON, "person"), _ent(AR_PORT, "place")],
        [_rel(AR_PERSON, AR_PORT, "works in", P15[:-1])],
    ),
}


@dataclass
class Fixture:
    scope: object
    stranger: object
    index: object
    version: object
    graph: object
    chat: object
    source: object
    view_id: str
    chunks: list


async def build_fixture_graph():
    embedder = ScriptedEmbedder(lambda text: (1.0, 0.0, 0.0, 0.0))
    scope, stranger, index, version = await ready_index(SPANS, embedder)
    chunks = await index.list_chunks(scope, version.id, offset=0, limit=500)
    view_id = chunks[0].extraction_view_id
    view = ViewSource(
        extraction_view_id=view_id, text=TEXT, blocks=(SourceBlockRef(block_id="b00000", start=0, end=len(TEXT)),)
    )
    source = StaticSourceReader({(scope.tenant_id, scope.project_id, view_id): view})

    def respond(request):
        return REPLIES.get(json.loads(request.messages[1].content)["chunk"], _reply()), "stop"

    chat = ScriptedChat(respond)
    builder = EntityRelationGraphBuilder(
        EntityRelationExtractConfig(), scope=scope, index_version=version, index=index, source=source, chat=chat
    )
    report = await builder.build()
    graph = await index.get_entity_graph(scope, version.id, report.graph_key)
    return Fixture(scope, stranger, index, version, graph, chat, source, view_id, chunks)


def entity_seed(question, chunks, version):
    """R02-labelled seed: the chunks whose text contains a fixture entity name named in the question."""
    named = [n for n in NAMES if n.casefold() in question.casefold()]
    picked = [c for c in chunks if any(n in c.text for n in named)]
    hits = tuple(
        RetrievedChunk(
            chunk_id=c.id,
            extraction_view_id=c.extraction_view_id,
            start=c.start,
            end=c.end,
            block_ids=c.block_ids,
            text=c.text,
            score=1.0 - i / 10,
            rank=i + 1,
        )
        for i, c in enumerate(picked)
    )
    return RetrievalResult(
        technique_id="R02", grounding="corpus", index_version_id=version.id, hits=hits, scoring="fixture-seed"
    )


def evidence_span_recall(result, gold, view_id):
    """Fraction of gold (start, end) spans fully inside one hit of the given view."""
    covered = sum(
        any(h.extraction_view_id == view_id and h.start <= a and b <= h.end for h in result.hits) for a, b in gold
    )
    return covered / len(gold)
