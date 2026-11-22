"""R48 multi-hop question answered through the existing AnswerService path (scripted chat, offline)."""

import asyncio
import json
import re

import pytest
from _lexical_embedder import LexicalTestEmbedder
from _scripted_chat import ScriptedChatModel
from _templates import prompt_templates
from rag_foundry_answering.service import AnswerConfig, AnswerService, RunPrompts
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_domain.index_testing import InMemoryIndexRepository
from rag_foundry_domain.testing import make_scope
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
from rag_foundry_indexing import IndexService, KagConfig, KagExtractConfig, KagGraphBuilder, KagRetriever
from rag_foundry_indexing.context_expansion import PayloadSourceReader
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.ledger import RecordingChatModel, UsageLedger
from test_extraction_service import ingest, setup

from tests.indexing._kag_fixture import K1, K2, K3, KAG_REPLIES, PARAGRAPHS, SCHEMA, TEXT, make_chat

_EVIDENCE = re.compile(r"<<<EVIDENCE (S\d+) ([0-9a-f]{16})>>>\n(.*?)\n<<<END \1 \2>>>", re.DOTALL)

MULTI_Q = "Who leads the organization that Cedar Labs supplies?"
MISS_Q = "Who leads Delta Works?"
CONFLICT_Q = "When was Blue Reef founded?"
PARTIAL_Q = "Which organizations supply the Harbor Authority?"


def _e(name):
    return {"entity": {"name": name, "type": "Organization"}}


def _step(i, subject, predicate, obj):
    return {"op": "retrieve", "id": f"s{i}", "subject": subject, "predicate": predicate, "object": obj}


def _out(i, *inputs):
    return {"op": "output", "id": f"s{i}", "inputs": list(inputs)}


PLANS = {
    MULTI_Q: {
        "steps": [
            _step(1, _e("Cedar Labs"), "supplies", None),
            _step(2, None, "leads", {"step": "s1"}),
            _out(3, "s2"),
        ]
    },
    MISS_Q: {"steps": [_step(1, None, "leads", _e("Delta Works")), _out(2, "s1")]},
    CONFLICT_Q: {"steps": [_step(1, _e("Blue Reef"), "founded_year", None), _out(2, "s1")]},
    PARTIAL_Q: {"steps": [_step(1, None, "supplies", _e("Harbor Authority")), _out(2, "s1")]},
}


def _plan_responder(request):
    question = json.loads(request.messages[1].content)["question"]
    return json.dumps(PLANS[question]), "stop"


def _cite_all(request):
    """Parse the S markers in the request and cite every evidence block, one sentence each."""
    user = next(message.content for message in request.messages if message.role == "user")
    blocks = _EVIDENCE.findall(user)
    if not blocks:
        return "INSUFFICIENT_EVIDENCE", "stop"
    return " ".join(f"Fact {i} is stated in the source [{m}]." for i, (m, _n, _t) in enumerate(blocks, 1)), "stop"


def _chunk_size():
    # The recursive splitter keeps each paragraph's trailing separator inside its piece (len + 2, none for the last).
    sizes = [len(p) + 2 for p in PARAGRAPHS[:-1]] + [len(PARAGRAPHS[-1])]
    size = max(sizes)
    assert size < min(a + b for a, b in zip(sizes, sizes[1:], strict=False))
    return size


@pytest.fixture
def built(tmp_path):
    async def create():
        scope = make_scope()
        extraction, intake, _spy, store, _docs, views, _scope = setup(tmp_path, scope)
        doc_id, version_id = await ingest(intake, scope, TEXT.encode("utf-8"), name="kag.txt", media="text/plain")
        view = (await extraction.extract(scope, doc_id, version_id)).view
        index = InMemoryIndexRepository(views)
        uow = InMemoryUnitOfWorkFactory(extractions=views, indexes=index)
        anchors = AnchorService(store, InMemoryUnitOfWorkFactory(extractions=views), provider_namespace="test")
        chunker = RecursiveChunker(RecursiveConfig(chunk_size=_chunk_size(), chunk_overlap=0))
        version = await IndexService(store, uow, anchors).build(scope, (view.id,), chunker, LexicalTestEmbedder())
        chunks = await index.list_chunks(scope, version.id, offset=0, limit=500)
        assert [c.text for c in chunks] == list(PARAGRAPHS)  # exactly one chunk per paragraph

        build_ledger = UsageLedger()
        builder = KagGraphBuilder(
            KagExtractConfig(),
            scope=scope,
            index_version=version,
            index=index,
            source=PayloadSourceReader(store, uow),
            chat=RecordingChatModel(make_chat(KAG_REPLIES), build_ledger),
            knowledge_schema=SCHEMA,
        )
        report = await builder.build()  # stores the graph in the in-memory index store
        graph = await index.get_knowledge_graph(scope, version.id, report.kg_key)
        return scope, index, anchors, version, graph, build_ledger

    return asyncio.run(create())


async def _answer(built, question):
    scope, index, anchors, version, graph, _build_ledger = built
    plan_ledger, ledger = UsageLedger(), UsageLedger()
    plan_chat = RecordingChatModel(ScriptedChatModel(_plan_responder), plan_ledger)
    chat = RecordingChatModel(ScriptedChatModel(_cite_all), ledger)
    retriever = KagRetriever(KagConfig(), scope=scope, graph=graph, index=index, index_version=version, chat=plan_chat)
    # Open item: the derived answer (retriever.last_trace.answer, e.g. "Omar Nasser" or a count) is not passed to
    # the answer model; only the supporting original chunks are. A later task may add that.
    result = await AnswerService(anchors, nonce_factory=lambda: "0123456789abcdef").answer(
        scope,
        question,
        prompts=RunPrompts(*prompt_templates()),
        retriever=retriever,
        technique_version=retriever.manifest.version,
        index_version=version,
        chat=chat,
        ledger=ledger,
        config=AnswerConfig(min_score=0.0),
    )
    return result, retriever, plan_ledger, anchors, scope


def test_multi_hop_question_is_answered_with_two_exact_citations(built):
    async def run():
        result, retriever, plan_ledger, anchors, scope = await _answer(built, MULTI_Q)
        build_calls, plan_calls = built[5].totals().calls, plan_ledger.totals().calls
        print(f"answer calls={result.usage.model_calls} plan calls={plan_calls} KG build calls={build_calls}")
        assert result.status == "answered"
        assert result.pins.technique_id == "R48"
        assert result.usage.model_calls == 1  # answer only; plan and build calls are on their own ledgers
        assert plan_calls == 1 and build_calls == len(PARAGRAPHS)
        assert retriever.last_trace.status == "answered"
        assert len(result.citations) == 2
        assert {e.marker for e in result.evidence if e.used} == {"S1", "S2"}
        resolved = await anchors.resolve(scope, tuple(a for c in result.citations for a in c.anchors))
        assert resolved and all(item.status == "exact" for item in resolved)
        exact = " ".join(item.exact for item in resolved)
        for quote in (K1, K2):
            assert quote in exact

    asyncio.run(run())


@pytest.mark.parametrize(("question", "status"), [(MISS_Q, "incomplete"), (CONFLICT_Q, "conflict")])
def test_losing_case_incomplete_or_conflict_abstains_without_answer_model_call(built, question, status):
    result, retriever, plan_ledger, _anchors, _scope = asyncio.run(_answer(built, question))
    assert retriever.last_trace.status == status
    assert plan_ledger.totals().calls == 1
    assert result.status == "abstained"
    assert result.usage.model_calls == 0
    assert result.citations == ()
    assert result.evidence == ()
    assert result.answer is None


def test_partial_case_incomplete_graph_cites_only_the_known_suppliers(built):
    async def run():
        result, retriever, _plan_ledger, anchors, scope = await _answer(built, PARTIAL_Q)
        # The Gulf Marine fact (K7) was rejected at extraction, so the graph knows only Cedar Labs and Delta Works.
        # The KG cannot know it is incomplete: R48 reports "answered" and the answer path answers on two of three
        # suppliers. The answer cannot be complete, and nothing in the response says so (pinned limit).
        assert retriever.last_trace.status == "answered"
        assert result.status == "answered"
        assert len(result.citations) == 2
        resolved = await anchors.resolve(scope, tuple(a for c in result.citations for a in c.anchors))
        assert resolved and all(item.status == "exact" for item in resolved)
        exact = " ".join(item.exact for item in resolved)
        assert K1 in exact and K3 in exact
        assert "Gulf Marine" not in exact

    asyncio.run(run())
