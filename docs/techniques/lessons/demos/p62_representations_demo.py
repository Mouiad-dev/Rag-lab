# ruff: noqa: UP031
"""P6.2 lesson demo: R23 multi-representation and R24 contextual-prefix retrieval mechanics.

Offline. The chat is a scripted responder (fixed JSON replies keyed by the passage) and the embedder is a dict
ScriptedEmbedder (hand-written 6-dimensional vectors). Neither is a model, so the rows show how the code handles
given inputs, not how a model behaves. Every rank, term, count and cost figure comes from calling the package
code. The output has no date or timing values and is byte-identical between runs.
"""

import asyncio
import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root, for the tests.* helpers

from rag_foundry_indexing import (  # noqa: E402
    ContextualPrefixBuildConfig,
    ContextualPrefixBuilder,
    ContextualRetriever,
    ContextualSearchConfig,
    DenseConfig,
    DenseRetriever,
    MultiRepresentationBuildConfig,
    MultiRepresentationBuilder,
    MultiRepresentationRetriever,
    RepresentationSearchConfig,
)
from rag_foundry_indexing.context_expansion import SourceBlockRef, ViewSource  # noqa: E402
from rag_foundry_providers.fake import fake_embedding_profile  # noqa: E402

from tests.chunking._scripted_models import ScriptedChat, ScriptedEmbedder  # noqa: E402
from tests.indexing._fixture import DISTRACTORS, FACT_EN  # noqa: E402
from tests.indexing._inmemory_index import ready_index  # noqa: E402
from tests.indexing._source_reader import StaticSourceReader  # noqa: E402

OUT = []


def emit(line=""):
    OUT.append(line)


# Axes of the scripted 6-dimensional space: tide, bakery, transport, generic, almanac name, handbook name.
def vec(*values):
    return tuple(float(v) for v in values) + (0.0,) * (6 - len(values))


PARAGRAPHS = (FACT_EN, *DISTRACTORS[:5])
NAMES = ("fact_en", "bakery", "railway", "chess", "bridge", "choir")
NAME_OF = dict(zip(PARAGRAPHS, NAMES, strict=True))
DOC = "\n\n".join(PARAGRAPHS)
SPANS = []
_offset = 0
for _text in PARAGRAPHS:
    SPANS.append((_offset, _offset + len(_text), _text))
    _offset += len(_text) + 2

ORIGINAL_VECTORS = (
    vec(0.2, 0, 0, 0.98),
    vec(0, 0.9, 0, 0.4),
    vec(0, 0, 0.9, 0.4),
    vec(0.3, 0, 0, 0.95),
    vec(0, 0, 0.5, 0.85),
    vec(0, 0.1, 0, 1.0),
)
# R23 replies: chunk -> (summary, question, summary vector, question vector).
R23_SCRIPT = {
    FACT_EN: (
        "Sea level logging at a coastal station.",
        "How often are sea levels recorded?",
        vec(0.7, 0, 0, 0.7),
        vec(0.95, 0, 0, 0.3),
    ),
    PARAGRAPHS[1]: (
        "Bread baking routine.",
        "What does Mill Lane produce?",
        vec(0, 0.95, 0, 0.3),
        vec(0, 0.6, 0, 0.8),
    ),
    PARAGRAPHS[2]: (
        "Mountain railway climb.",
        "How long does the railway climb take?",
        vec(0, 0, 0.95, 0.3),
        vec(0, 0, 0.98, 0.1),
    ),
    PARAGRAPHS[3]: (
        "A club meeting.",
        "When does the chess club meet?",
        vec(0.1, 0, 0, 1.0),
        vec(0.1, 0, 0, 0.99),
    ),
    PARAGRAPHS[4]: (
        "Bridge maintenance.",
        "When is the bridge repainted?",
        vec(0, 0, 0.6, 0.8),
        vec(0, 0, 0.6, 0.8),
    ),
    PARAGRAPHS[5]: (
        "Local activities and routines.",
        "Where does the choir rehearse?",
        vec(0, 0.99, 0, 0.1),
        vec(0, 0.1, 0, 1.0),
    ),
}
# R24 replies: chunk -> (context, vector of context + blank line + chunk).
R24_SCRIPT = {
    FACT_EN: ("From the Harbour Almanac, section on tide gauges.", vec(0.6, 0, 0, 0.3, 0.74, 0)),
    PARAGRAPHS[1]: ("From the Mill Lane Handbook, section on bakeries.", vec(0, 0.6, 0, 0.3, 0, 0.74)),
    PARAGRAPHS[2]: ("From the Mill Lane Handbook, section on railways.", vec(0, 0, 0.6, 0.3, 0, 0.74)),
    PARAGRAPHS[3]: ("From the Mill Lane Handbook, section on clubs.", vec(0.1, 0, 0, 0.7, 0, 0.7)),
    PARAGRAPHS[4]: ("From the Mill Lane Handbook, section on bridge upkeep.", vec(0, 0, 0.5, 0.5, 0, 0.7)),
    PARAGRAPHS[5]: ("From the chess club newsletter, meeting times.", vec(0.1, 0, 0, 0.99, 0, 0.05)),
}
Q1 = "How frequently are the tide readings taken?"
Q2 = "Which place bakes bread before dawn?"
Q3 = "Harbour Almanac: how often are readings taken?"
Q4 = "When does the chess club meet?"
QUERY_VECTORS = {Q1: vec(1), Q2: vec(0, 1), Q3: vec(0.6, 0, 0, 0, 0.8), Q4: vec(0.1, 0, 0, 0.9)}
SYNTHETIC = {}  # synthetic text -> kind
VECTORS = dict(zip(PARAGRAPHS, ORIGINAL_VECTORS, strict=True))
VECTORS.update(QUERY_VECTORS)
for _chunk, (_summary, _question, _sv, _qv) in R23_SCRIPT.items():
    VECTORS[_summary], VECTORS[_question] = _sv, _qv
    SYNTHETIC[_summary], SYNTHETIC[_question] = "summary", "question"
for _chunk, (_context, _cv) in R24_SCRIPT.items():
    VECTORS[_context + "\n\n" + _chunk] = _cv
    SYNTHETIC[_context] = "context_prefix"


def reply(request):
    payload = json.loads(request.messages[-1].content)
    if "passage" in payload:
        summary, question, _, _ = R23_SCRIPT[payload["passage"]]
        return json.dumps({"summary": summary, "questions": [question]}), "stop"
    return json.dumps({"context": R24_SCRIPT[payload["chunk"]][0]}), "stop"


def ranks(result):
    return ", ".join("%s(%d)" % (NAME_OF[hit.text], hit.rank) for hit in result.hits)


def rank_of(result, name):
    for hit in result.hits:
        if NAME_OF[hit.text] == name:
            return str(hit.rank)
    return "not in top %d" % len(result.hits)


def cost(report):
    return sum((Decimal(str(u.estimated_cost_usd)) for u in report.usage), Decimal(0))


class Counts:
    def __init__(self, chat, embedder):
        self.chat, self.embedder = chat, embedder
        self.mark = (len(chat.requests), len(embedder.calls))

    def take(self):
        now = (len(self.chat.requests), len(self.embedder.calls))
        delta = (now[0] - self.mark[0], now[1] - self.mark[1])
        self.mark = now
        return delta


def show(title, question, expected, base, retrieval, note):
    emit("### %s" % title)
    emit()
    emit("Question: `%s`. Right chunk: `%s`." % (question, expected))
    emit()
    emit("| retriever | order (chunk(rank)) | right chunk rank |")
    emit("|---|---|---|")
    emit("| R02 dense on original chunks (k=5) | %s | %s |" % (ranks(base), rank_of(base, expected)))
    emit("| %s | %s | %s |" % (note, ranks(retrieval.result), rank_of(retrieval.result, expected)))
    emit()
    emit("| rank | chunk (original text) | matched by | match kind | synthetic | matched synthetic text |")
    emit("|---|---|---|---|---|---|")
    for hit, match in zip(retrieval.result.hits, retrieval.matches, strict=True):
        emit(
            "| %d | %s | %s | %s | %s | %s |"
            % (
                hit.rank,
                NAME_OF[hit.text],
                hit.score_components[0].term,
                match.kind,
                "yes" if match.synthetic else "no",
                ("`%s`" % match.text) if match.text else "-",
            )
        )
    emit()


async def main():
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    embedder = ScriptedEmbedder(lambda text: VECTORS[text], fake_embedding_profile(dim=6))
    owner, _, index, version = await ready_index(SPANS, embedder)
    chunk_embeds = len(embedder.calls)
    view_id = version.extraction_view_ids[0]
    source = StaticSourceReader(
        {
            (owner.tenant_id, owner.project_id, view_id): ViewSource(
                extraction_view_id=view_id,
                text=DOC,
                blocks=(SourceBlockRef(block_id="b00000", start=0, end=len(DOC)),),
            )
        }
    )
    chat = ScriptedChat(reply)
    counts = Counts(chat, embedder)
    usage_rows = [("original chunk embeddings (index build, one call per chunk)", 0, chunk_embeds, 0, Decimal(0))]
    counts.take()
    dense = DenseRetriever(DenseConfig(k=5), scope=owner, index_version=version, index=index, embedder=embedder)

    r23_build = MultiRepresentationBuilder(
        MultiRepresentationBuildConfig(n_questions=1),
        scope=owner,
        index_version=version,
        index=index,
        chat=chat,
        embedder=embedder,
    )
    r23_report = await r23_build.build()
    build_chat, build_embed = counts.take()
    usage_rows.append(("R23 build (summary + question per chunk)", build_chat, build_embed, 0, cost(r23_report)))
    r24_build = ContextualPrefixBuilder(
        ContextualPrefixBuildConfig(),
        scope=owner,
        index_version=version,
        index=index,
        source=source,
        chat=chat,
        embedder=embedder,
    )
    r24_report = await r24_build.build()
    build_chat, build_embed = counts.take()
    usage_rows.append(("R24 build (one prefix per chunk)", build_chat, build_embed, 0, cost(r24_report)))
    r23 = MultiRepresentationRetriever(
        RepresentationSearchConfig(representation_key=r23_report.representation_key, k=3),
        scope=owner,
        index_version=version,
        index=index,
        embedder=embedder,
    )
    r24 = ContextualRetriever(
        ContextualSearchConfig(representation_key=r24_report.representation_key, k=3),
        scope=owner,
        index_version=version,
        index=index,
        embedder=embedder,
    )

    retrievals = []
    note23 = "R23 (k=3, includes original chunks)"
    note24 = "R24 (k=3, prefixed text only)"
    sections = [
        ("R23 success: a generated question matches", r23, Q1, "fact_en", note23),
        ("R23 losing: a vague summary outranks the right chunk", r23, Q2, "bakery", note23),
        ("R24 success: the prefix adds the missing document name", r24, Q3, "fact_en", note24),
        ("R24 losing: a wrong prefix pulls a distractor up", r24, Q4, "chess", note24),
    ]
    emit("## R23 and R24 mechanics")
    emit()
    for number, (title, retriever, question, expected, note) in enumerate(sections, 1):
        base = await dense.retrieve(question)
        found = await retriever.retrieve_traced(question)
        _, query_embeds = counts.take()
        label = ("R23 success", "R23 losing", "R24 success", "R24 losing")[number - 1]
        usage_rows.append((label + " query (R02 baseline + technique)", 0, 0, query_embeds, Decimal(0)))
        retrievals.append((label, found))
        show("%d. %s" % (number, title), question, expected, base, found, note)
    emit(
        "The losing rows are not bugs: the code ranks by the vectors it is given. In the R23 losing row the scripted "
        "summary of `choir` is deliberately vague; in the R24 losing row the scripted prefix of `choir` names the "
        "wrong topic. A real generator can produce such text, and R23/R24 cannot detect it."
    )
    emit()
    emit("## 5. Synthetic-text check")
    emit()
    emit(
        "For every retrieval above: are all hit texts original chunks, and does any generated string appear inside "
        "any `hit.text`?"
    )
    emit()
    emit("| retrieval | hits | hits equal to an original chunk | generated strings found inside hit texts |")
    emit("|---|---|---|---|")
    for label, found in retrievals:
        hits = found.result.hits
        originals = sum(hit.text in NAME_OF for hit in hits)
        leaked = sum(1 for hit in hits for text in SYNTHETIC if text in hit.text)
        emit("| %s | %d | %d | %d |" % (label, len(hits), originals, leaked))
    emit()
    kinds = list(SYNTHETIC.values())
    emit(
        "Generated strings checked: %d (%d summaries, %d questions, %d prefixes). Matches that came from generated "
        "text are marked `synthetic: yes` in the tables; hits are always the stored original chunk with its source "
        "span." % (len(kinds), kinds.count("summary"), kinds.count("question"), kinds.count("context_prefix"))
    )
    emit()
    emit("## 6. Usage (scripted chat and embedder, fake usage, cost 0)")
    emit()
    emit("| step | build chat calls | build embed calls | query embed calls | cost USD |")
    emit("|---|---|---|---|---|")
    totals = [0, 0, 0, Decimal(0)]
    for name, chat_calls, embed_calls, query_calls, cost_usd in usage_rows:
        emit("| %s | %d | %d | %d | %s |" % (name, chat_calls, embed_calls, query_calls, cost_usd))
        for i, value in enumerate((chat_calls, embed_calls, query_calls, cost_usd)):
            totals[i] += value
    emit("| total | %d | %d | %d | %s |" % tuple(totals))
    emit()
    emit(
        "Cross-check against the doubles: the scripted chat received %d requests; the scripted embedder received %d "
        "calls, of which %d embedded original chunks. Builder reports: R23 %d chat usage rows and %d rejected, R24 "
        "%d chat usage rows and %d rejected. Each build embeds all its rows in one batched call."
        % (
            len(chat.requests),
            len(embedder.calls),
            chunk_embeds,
            sum(u.operation.value == "chat" for u in r23_report.usage),
            len(r23_report.rejected),
            sum(u.operation.value == "chat" for u in r24_report.usage),
            len(r24_report.rejected),
        )
    )
    emit()
    head = [
        "# P6.2 R23 and R24 representations: demo output",
        "",
        "Offline fixture: the first six `_fixture` paragraphs in an in-memory index (chunk names: %s). The chat is a "
        "scripted responder that returns fixed JSON keyed by the passage; it is not a model. The embedder is a "
        "dict-backed ScriptedEmbedder with hand-written 6-dimensional vectors, so similarity is by construction. All "
        "generated text below is synthetic and scripted. Every table value is computed by the package code from "
        "those inputs. This is not a benchmark." % ", ".join(NAMES),
        "",
    ]
    print("\n".join(head + OUT))


asyncio.run(main())
