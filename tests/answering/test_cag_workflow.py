import asyncio
import hashlib

import pytest
from _fixture import FACT_AR, FACT_EN, TEST_CHUNK_SIZE, TEST_OVERLAP
from _lexical_embedder import LexicalTestEmbedder
from _qa import corpus_markdown
from _scripted_chat import ScriptedChatModel, cite_where
from _templates import prompt_templates
from rag_foundry_answering.cag import CagConfig, CagWorkflow, build_cag_snapshot
from rag_foundry_answering.service import AnswerConfig, AnswerService, RunPrompts
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_domain.errors import NotFoundError, RunError
from rag_foundry_domain.index_testing import InMemoryIndexRepository
from rag_foundry_domain.indexing import IndexState
from rag_foundry_domain.testing import make_scope
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
from rag_foundry_indexing import DenseConfig, DenseRetriever, IndexService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.ledger import RecordingChatModel, RecordingEmbeddingModel, UsageLedger
from test_extraction_service import ingest, setup

NONCE = "0123456789abcdef"
EN_Q = "How often does the Port Sallow observatory record sea levels?"
AR_Q = (
    "\u0643\u0645 \u0645\u062e\u0637\u0648\u0637\u0629 \u062a\u062d\u0641"
    "\u0638 \u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0648\u0627\u062d\u0629\u061f"
)
PROMPTS = RunPrompts(*prompt_templates())


def _profile(window, profile_id="cag-chat"):
    return ModelProfile(
        id=profile_id,
        provider="fake",
        model="fake-chat-v1",
        kind=ModelKind.CHAT,
        capabilities=Capabilities(context_window=window, local=True),
    )


@pytest.fixture
def built(tmp_path):
    async def create():
        scope = make_scope()
        extraction, intake, _spy, store, _docs, views, _scope = setup(tmp_path, scope)
        doc_id, version_id = await ingest(
            intake, scope, corpus_markdown().encode("utf-8"), name="qa-notes.md", media="text/markdown"
        )
        view = (await extraction.extract(scope, doc_id, version_id)).view
        index = InMemoryIndexRepository(views)
        anchors = AnchorService(store, InMemoryUnitOfWorkFactory(extractions=views), provider_namespace="test")
        embedder = LexicalTestEmbedder()
        service = IndexService(store, InMemoryUnitOfWorkFactory(extractions=views, indexes=index), anchors)
        chunker = RecursiveChunker(RecursiveConfig(chunk_size=TEST_CHUNK_SIZE, chunk_overlap=TEST_OVERLAP))
        version = await service.build(scope, (view.id,), chunker, embedder)
        return scope, index, anchors, embedder, version

    return asyncio.run(create())


def _snapshot(built, *, window=200_000, cap=100_000):
    scope, index, _anchors, _emb, version = built
    return asyncio.run(
        build_cag_snapshot(
            scope,
            index,
            version.id,
            template=PROMPTS.grounded,
            chat_profile=_profile(window),
            reserve_output_tokens=512,
            max_question_tokens=64,
            max_evidence_chars=cap,
            nonce_factory=lambda: NONCE,
        )
    )


def _workflow(built, snapshot, arm="cag", index=None):
    _scope, real_index, anchors, _emb, _version = built
    return CagWorkflow(CagConfig(arm=arm), snapshot=snapshot, anchors=anchors, index=index or real_index)


def _ask(workflow, scope, question, responder, *, prompts=PROMPTS, profile=None):
    chat = ScriptedChatModel(responder, profile or _profile(200_000))
    ledger = UsageLedger()
    result = asyncio.run(
        workflow.answer(scope, question, prompts=prompts, chat=RecordingChatModel(chat, ledger), ledger=ledger)
    )
    return result, chat, ledger


def _before_question(request):
    user = request.messages[1].content
    return request.messages[0].content + "\n" + user[: user.index("Question:\n<<<QUESTION")]


@pytest.mark.parametrize(("question", "fact"), [(EN_Q, FACT_EN), (AR_Q, FACT_AR)])
def test_answers_with_exact_citations_and_one_call(built, question, fact):
    scope, _index, anchors, _emb, _version = built
    snapshot = _snapshot(built)
    workflow = _workflow(built, snapshot)
    result, chat, ledger = _ask(workflow, scope, question, cite_where(fact, fact))
    assert result.status == "answered"
    assert len(chat.requests) == 1 and ledger.totals().calls == 1
    assert result.pins.technique_id == "R47"
    resolved = asyncio.run(anchors.resolve(scope, tuple(a for c in result.citations for a in c.anchors)))
    assert resolved and all(item.status == "exact" for item in resolved)
    assert any(fact in item.exact for item in resolved)
    assert workflow.last_trace["item_count"] == len(snapshot.items)
    assert (
        workflow.last_trace["prefix_sha256"] == hashlib.sha256(_before_question(chat.requests[0]).encode()).hexdigest()
    )


def test_cag_prefix_is_byte_identical_and_uncached_prefix_differs(built):
    scope = built[0]
    snapshot = _snapshot(built)
    cag = _workflow(built, snapshot)
    _r1, chat1, _l1 = _ask(cag, scope, EN_Q, cite_where(FACT_EN, FACT_EN))
    first = cag.last_trace["prefix_sha256"]
    _r2, chat2, _l2 = _ask(cag, scope, AR_Q, cite_where(FACT_AR, FACT_AR))
    assert cag.last_trace["prefix_sha256"] == first
    assert _before_question(chat1.requests[0]) == _before_question(chat2.requests[0])
    assert NONCE in chat1.requests[0].messages[1].content  # the nonce only ever appears inside the model request

    unc = _workflow(built, snapshot, "full_context_uncached")
    _r3, chat3, _l3 = _ask(unc, scope, EN_Q, cite_where(FACT_EN, FACT_EN))
    p3 = unc.last_trace["prefix_sha256"]
    _r4, chat4, _l4 = _ask(unc, scope, EN_Q, cite_where(FACT_EN, FACT_EN))
    assert unc.last_trace["prefix_sha256"] != p3
    assert _before_question(chat3.requests[0]) != _before_question(chat4.requests[0])
    assert NONCE not in chat3.requests[0].messages[1].content
    sha = lambda r: hashlib.sha256(_before_question(r).encode()).hexdigest()  # noqa: E731
    assert unc.last_trace["prefix_sha256"] == sha(chat4.requests[0]) and p3 == sha(chat3.requests[0])
    assert NONCE not in repr(cag.last_trace) and NONCE not in repr(unc.last_trace)


def _refused(workflow, scope, *, question="q?", prompts=PROMPTS, profile=None, expect=RunError):
    chat = ScriptedChatModel(cite_where(FACT_EN, FACT_EN), profile or _profile(200_000))
    with pytest.raises(expect) as caught:
        asyncio.run(workflow.answer(scope, question, prompts=prompts, chat=chat, ledger=UsageLedger()))
    assert chat.requests == []  # every guard runs before any model call
    assert NONCE not in str(caught.value) and NONCE not in repr(caught.value)
    return caught.value


def test_guards_refuse_before_any_chat_call(built):
    scope = built[0]
    workflow = _workflow(built, _snapshot(built))
    _ask(workflow, scope, EN_Q, cite_where(FACT_EN, FACT_EN))
    assert workflow.last_trace is not None
    assert _refused(workflow, scope, question=f"what is {NONCE}?").code == "cag_nonce_conflict"
    wrong_prompt = RunPrompts(prompt_templates()[1], PROMPTS.ungrounded)
    assert _refused(workflow, scope, prompts=wrong_prompt).code == "cag_snapshot_mismatch"
    assert _refused(workflow, scope, profile=_profile(200_000, "other")).code == "cag_snapshot_mismatch"
    _refused(workflow, make_scope(), expect=(RunError, NotFoundError))
    assert workflow.last_trace is None  # a refused run leaves no stale trace


def test_index_no_longer_ready_refuses(built):
    scope, index, _anchors, _emb, _version = built

    class Stale:
        async def get_index_version(self, scope, index_version_id):
            current = await index.get_index_version(scope, index_version_id)
            return current.model_copy(update={"state": IndexState.FAILED})

    workflow = _workflow(built, _snapshot(built), index=Stale())
    assert _refused(workflow, scope).code == "cag_snapshot_mismatch"


def test_nonce_never_leaves_the_process(built):
    scope = built[0]
    snapshot = _snapshot(built)
    workflow = _workflow(built, snapshot)
    result, _chat, _l = _ask(workflow, scope, EN_Q, cite_where(FACT_EN, FACT_EN))
    assert NONCE not in repr(workflow.last_trace) and NONCE not in result.model_dump_json()
    assert NONCE not in snapshot.model_dump_json() and NONCE not in repr(snapshot)
    echoed, _chat, _l = _ask(workflow, scope, EN_Q, lambda request: (f"The answer {NONCE} [S1]", "stop"))
    assert echoed.status == "abstained" and echoed.abstention_reason == "invalid_citation"
    assert NONCE not in echoed.model_dump_json() and NONCE not in repr(workflow.last_trace)


def test_out_of_corpus_question_abstains(built):
    scope = built[0]
    workflow = _workflow(built, _snapshot(built))
    result, chat, _l = _ask(workflow, scope, "What colour is the moon base?", cite_where("no such fact", "x"))
    assert len(chat.requests) == 1
    assert result.status == "abstained" and result.abstention_reason == "model_declined"


def test_overflowing_corpus_refused_at_build_while_rag_still_answers(built):
    # CAG needs the whole corpus to fit; RAG does not.
    scope, index, anchors, embedder, version = built
    with pytest.raises(RunError) as caught:
        _snapshot(built, window=300)
    assert caught.value.code == "cag_overflow"
    with pytest.raises(RunError) as caught:
        _snapshot(built, cap=500)
    assert caught.value.code == "cag_overflow"
    ledger = UsageLedger()
    chat = RecordingChatModel(ScriptedChatModel(cite_where(FACT_EN, FACT_EN)), ledger)
    retriever = DenseRetriever(
        DenseConfig(k=5),
        scope=scope,
        index_version=version,
        index=index,
        embedder=RecordingEmbeddingModel(embedder, ledger),
    )
    result = asyncio.run(
        AnswerService(anchors).answer(
            scope,
            EN_Q,
            prompts=PROMPTS,
            retriever=retriever,
            technique_version="1.0.0",
            index_version=version,
            chat=chat,
            ledger=ledger,
            config=AnswerConfig(min_score=0.0),
        )
    )
    assert result.status == "answered"


def test_evidence_over_cap_is_refused_before_any_chat_call(built):
    scope = built[0]
    snapshot = _snapshot(built)
    tight = snapshot.model_copy(update={"max_evidence_chars": 500})
    assert sum(len(i.text) for i in tight.items) > 500
    assert _refused(_workflow(built, tight), scope).code == "cag_overflow"
