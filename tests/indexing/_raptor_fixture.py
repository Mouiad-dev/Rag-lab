"""Shared R37 summary-tree fixture (P7.3-03, used by 04, 09 and 10).

Thirteen paragraphs in three lexically disjoint topics (harbor, orchard, clinic) and one Arabic clinic
paragraph that also carries the Latin word "clinic". The Latin word is there because LexicalTestEmbedder
shares no tokens across Arabic and English: that word, not meaning, makes the Arabic paragraph group with the
clinic topic. Only a real multilingual embedder (the P7.3-10 live run) can show Arabic/English grouping by
meaning. One chunk per paragraph. The scripted summaries are
hand-written: level-1 summaries name their topic and "budget", the clinic summary omits the vaccine fridge
detail asked by ``Q_DETAIL`` (a flat chunk index finds it, the summary alone does not), and the root summary
uses words that appear in no leaf (so only the tree, not the flat index, can match an overview question).
"""

import json
from dataclasses import dataclass

from rag_foundry_indexing import RaptorBuildConfig, RaptorTreeBuilder
from rag_foundry_indexing.context_expansion import SourceBlockRef, ViewSource

from tests.chunking._scripted_models import ScriptedChat
from tests.indexing._inmemory_index import ready_index
from tests.indexing._lexical_embedder import LexicalTestEmbedder
from tests.indexing._source_reader import StaticSourceReader

AR_CLINIC = " ".join(
    (
        "\u062a\u0641\u062a\u062d",
        "\u0627\u0644\u0639\u064a\u0627\u062f\u0629",
        "clinic",
        "\u0623\u0628\u0648\u0627\u0628\u0647\u0627",
        "\u0644\u0644\u0645\u0631\u0636\u0649",
        "\u0643\u0644",
        "\u0635\u0628\u0627\u062d",
        "\u0641\u064a",
        "\u0627\u0644\u0640",
        "clinic.",
    )
)

PARAGRAPHS = (
    "Harbor crane, harbor quay, harbor berth: the harbor crane lifts containers onto the quay.",
    "Harbor budget, harbor budget: dredging, harbor lights and quay repairs.",
    "Harbor boats, harbor herring, harbor nets: fishing boats leave the harbor quay at dawn.",
    "Harbor pier, harbor tugboats, harbor budget, harbor budget: pier repairs and quay lights.",
    "Orchard apples, orchard pruning, orchard rows: apples ripen on every orchard tree.",
    "Orchard ladders, orchard crates, orchard apples: the orchard budget pays for pruning.",
    "Orchard bees, orchard blossoms, orchard hives: bees pollinate the pear trees each spring.",
    "Orchard frost, orchard heaters, orchard plums: frost threatens the plum trees in April.",
    "Clinic nurse, clinic desk, clinic patients: the clinic opens at eight each morning.",
    "Clinic staff, clinic training, clinic budget: the clinic budget funds nurse courses.",
    "Clinic vaccine fridge, clinic nurse: the fridge stays at four degrees and the clinic logs it.",
    "Clinic hall, clinic doctor, clinic patients: patients wait until the doctor calls a number.",
    AR_CLINIC,
)
LABELS = {
    0: "H1",
    1: "H2",
    2: "H3",
    3: "H4",
    4: "O1",
    5: "O2",
    6: "O3",
    7: "O4",
    8: "C1",
    9: "C2",
    10: "C3",
    11: "C4",
    12: "C5-ar",
}
TOPICS = {
    "harbor": ("H1", "H2", "H3", "H4"),
    "orchard": ("O1", "O2", "O3", "O4"),
    "clinic": ("C1", "C2", "C3", "C4", "C5-ar"),
}
TEXT = "\n\n".join(PARAGRAPHS)


def _spans():
    out, pos = [], 0
    for paragraph in PARAGRAPHS:
        out.append((pos, pos + len(paragraph), paragraph))
        pos += len(paragraph) + 2
    return tuple(out)


SPANS = _spans()
LABEL_OF_START = {SPANS[i][0]: LABELS[i] for i in range(len(SPANS))}
SPAN_OF_LABEL = {LABELS[i]: (SPANS[i][0], SPANS[i][1]) for i in range(len(SPANS))}


def _gold(*labels):
    return tuple(SPAN_OF_LABEL[label] for label in labels)


@dataclass(frozen=True)
class Question:
    text: str
    gold: tuple[tuple[int, int], ...]


Q_BUDGET = Question("What does each budget pay for?", _gold("H2", "H4", "O2", "C2"))
Q_DETAIL = Question("At what temperature is the clinic vaccine fridge kept?", _gold("C3"))
Q_OVERVIEW = Question(
    "Give a municipal overview of shared priorities and finances in the region.", _gold(*LABELS.values())
)
Q_ARABIC = Question(
    "\u0645\u062a\u0649 \u062a\u0641\u062a\u062d \u0627\u0644\u0639\u064a\u0627\u062f\u0629\u061f", _gold("C5-ar")
)

ROOT_SUMMARY = "Municipal overview: shared priorities, finances, region."
assert not {w.strip(":,.").casefold() for w in ROOT_SUMMARY.split()} & {
    w.strip(":,.").casefold() for p in PARAGRAPHS for w in p.split()
}
LEVEL1_SUMMARIES = {
    "harbor": "Harbor topic: the harbor budget and daily harbor operations.",
    "orchard": "Orchard topic: the orchard budget and seasonal orchard work.",
    "clinic": "Clinic topic: the clinic budget and general clinic service.",
}


def summary_responder(request):
    data = json.loads(request.messages[1].content)
    if data["level"] >= 2:
        return json.dumps({"summary": ROOT_SUMMARY}), "stop"
    joined = " ".join(child["text"] for child in data["children"]).casefold()
    topic = max(LEVEL1_SUMMARIES, key=lambda name: joined.count(name))
    return json.dumps({"summary": LEVEL1_SUMMARIES[topic]}), "stop"


@dataclass
class FixtureTree:
    scope: object
    stranger: object
    index: object
    version: object
    store: object
    tree: object
    report: object
    chat: object
    embedder: object
    source: object
    builder: object
    view_id: str
    chunks: list


async def make_fixture_index(*, paragraphs=PARAGRAPHS, embedder=None, ready=True):
    """A ready index (one chunk per paragraph) plus a source reader for its view."""
    embedder = embedder or LexicalTestEmbedder()
    text = "\n\n".join(paragraphs)
    spans, pos = [], 0
    for paragraph in paragraphs:
        spans.append((pos, pos + len(paragraph), paragraph))
        pos += len(paragraph) + 2
    scope, stranger, index, version = await ready_index(tuple(spans), embedder, ready=ready)
    chunks = await index.list_chunks(scope, version.id, offset=0, limit=500)
    view_id = chunks[0].extraction_view_id
    view = ViewSource(
        extraction_view_id=view_id, text=text, blocks=(SourceBlockRef(block_id="b00000", start=0, end=len(text)),)
    )
    source = StaticSourceReader({(scope.tenant_id, scope.project_id, view_id): view})
    return scope, stranger, index, version, source, embedder, chunks, view_id


async def build_fixture_tree(*, responder=None, **config):
    scope, stranger, index, version, source, embedder, chunks, view_id = await make_fixture_index()
    chat = ScriptedChat(responder or summary_responder)
    builder = RaptorTreeBuilder(
        RaptorBuildConfig(**config),
        scope=scope,
        index_version=version,
        index=index,
        source=source,
        chat=chat,
        embedder=embedder,
    )
    report = await builder.build()
    tree = await index.get_summary_tree(scope, version.id, report.tree_key)
    return FixtureTree(
        scope, stranger, index, version, index, tree, report, chat, embedder, source, builder, view_id, chunks
    )
