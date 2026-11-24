"""R38 relationship question answered through the existing AnswerService path (scripted chat, offline)."""

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
from rag_foundry_indexing import EntityRelationExtractConfig, EntityRelationGraphBuilder, IndexService
from rag_foundry_indexing.context_expansion import PayloadSourceReader
from rag_foundry_indexing.graph_retrieve import EntityGraphConfig, EntityGraphRetriever
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.ledger import RecordingChatModel, UsageLedger
from test_extraction_service import ingest, setup

from tests.chunking._scripted_models import ScriptedChat
from tests.indexing._graph_fixture import P0, P7, P14, PARAGRAPHS, QUESTION, REPLIES, _reply

_EVIDENCE = re.compile(r"<<<EVIDENCE (S\d+) ([0-9a-f]{16})>>>\n(.*?)\n<<<END \1 \2>>>", re.DOTALL)
_ALLOWED_PARTIAL = {"answered", "abstained"}


def _cite_all(request):
    """Cite every evidence block in the request, one sentence each."""
    user = next(message.content for message in request.messages if message.role == "user")
    blocks = _EVIDENCE.findall(user)
    if not blocks:
        return "INSUFFICIENT_EVIDENCE", "stop"
    return " ".join(
        f"Fact {i} is stated in the source [{marker}]." for i, (marker, _n, _t) in enumerate(blocks, 1)
    ), "stop"


# P15 (short Arabic line) is dropped: P14 plus a blank line plus P15 is shorter than the longest paragraph, so
# no chunk size can keep the two apart. The three-hop evidence (P0, P7, P14) does not involve P15.
PARAS = PARAGRAPHS[:15]
TEXT_USED = "\n\n".join(PARAS)


def _chunk_size():
    # The recursive splitter keeps each paragraph's trailing "\n\n" inside its piece, so a piece is len + 2
    # (the last paragraph has no separator) and two adjacent pieces merge when their sum fits the size.
    sizes = [len(p) + 2 for p in PARAS[:-1]] + [len(PARAS[-1])]
    size = max(sizes)
    assert size < min(a + b for a, b in zip(sizes, sizes[1:], strict=False))
    return size


@pytest.fixture
def built(tmp_path):
    async def create():
        scope = make_scope()
        extraction, intake, _spy, store, _docs, views, _scope = setup(tmp_path, scope)
        doc_id, version_id = await ingest(
            intake, scope, TEXT_USED.encode("utf-8"), name="graph.txt", media="text/plain"
        )
        view = (await extraction.extract(scope, doc_id, version_id)).view
        index = InMemoryIndexRepository(views)
        uow = InMemoryUnitOfWorkFactory(extractions=views, indexes=index)
        anchors = AnchorService(store, InMemoryUnitOfWorkFactory(extractions=views), provider_namespace="test")
        chunker = RecursiveChunker(RecursiveConfig(chunk_size=_chunk_size(), chunk_overlap=0))
        version = await IndexService(store, uow, anchors).build(scope, (view.id,), chunker, LexicalTestEmbedder())
        chunks = await index.list_chunks(scope, version.id, offset=0, limit=500)
        assert [c.text for c in chunks] == list(PARAS)

        def respond(request):
            return REPLIES.get(json.loads(request.messages[1].content)["chunk"], _reply()), "stop"

        build_ledger = UsageLedger()
        build_chat = RecordingChatModel(ScriptedChat(respond), build_ledger)
        builder = EntityRelationGraphBuilder(
            EntityRelationExtractConfig(),
            scope=scope,
            index_version=version,
            index=index,
            source=PayloadSourceReader(store, uow),
            chat=build_chat,
        )
        report = await builder.build()
        graph = await index.get_entity_graph(scope, version.id, report.graph_key)
        return scope, index, anchors, version, graph, build_ledger

    return asyncio.run(create())


async def _answer(built, question):
    scope, index, anchors, version, graph, _build_ledger = built
    ledger = UsageLedger()
    chat = RecordingChatModel(ScriptedChatModel(_cite_all), ledger)
    retriever = EntityGraphRetriever(EntityGraphConfig(), scope=scope, graph=graph, index=index, index_version=version)
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
    return result, anchors, scope


def test_relationship_question_is_answered_with_three_exact_citations(built):
    async def run():
        result, anchors, scope = await _answer(built, QUESTION)
        build_calls = built[5].totals().calls
        print(f"answer model calls={result.usage.model_calls} graph build model calls={build_calls}")
        assert result.status == "answered"
        assert result.pins.technique_id == "R38"
        assert result.usage.model_calls == 1
        assert build_calls == len(PARAS)
        assert len(result.citations) == 3
        resolved = await anchors.resolve(scope, tuple(a for c in result.citations for a in c.anchors))
        assert resolved and all(item.status == "exact" for item in resolved)
        exact = " ".join(item.exact for item in resolved)
        for quote in (P0, P7, P14):
            assert quote in exact

    asyncio.run(run())


def test_losing_case_no_entity_abstains_without_answer_model_call(built):
    result, _anchors, _scope = asyncio.run(_answer(built, "Who repaired the lighthouse?"))
    assert result.status == "abstained"
    assert result.usage.model_calls == 0
    assert result.citations == ()
    assert result.evidence == ()


def test_partial_case_alias_question_cannot_be_complete(built):
    # "Dr. Haddad" matches no entity, so R38 only sees Omar Nasser (neighborhood mode) and returns just the P14
    # chunk. Any answer built from it cannot cover the full Lina Haddad to Omar Nasser path.
    result, _anchors, _scope = asyncio.run(_answer(built, "How is Dr. Haddad connected to Omar Nasser?"))
    assert result.status in _ALLOWED_PARTIAL
    assert result.pins.technique_id == "R38"
    assert len(result.citations) <= 1
