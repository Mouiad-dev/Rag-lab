import asyncio
import hashlib
import json
import re
from types import SimpleNamespace

import pytest
import test_answer_service
from _fixture import FACT_EN, QUERY_EN
from _scripted_chat import ScriptedChatModel, cite_where
from _templates import prompt_templates
from rag_foundry_answering.service import AnswerConfig, AnswerService, RunPrompts
from rag_foundry_chunking._units import LabRegexTokenizer
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.runtime_settings import ANSWER_MAX_EVIDENCE_CHARS_MAX
from rag_foundry_indexing import DenseConfig, DenseRetriever
from rag_foundry_indexing.compression import CompressingRetriever, ExtractiveCompressionConfig, ExtractiveCompressor
from rag_foundry_indexing.packing import ContextPackConfig, ContextPacker
from rag_foundry_providers.ledger import RecordingChatModel, RecordingEmbeddingModel, UsageLedger

built = test_answer_service.built  # reuse the fixture

DEFAULT_REQUEST_SHA256 = "32d7bf0aaf123af476343a095989dd983fda5a13b19584bfa6e5fe84ef46d316"
DEFAULT_RESPONSE_SHA256 = "38b6460ed5f4b127bdb54391e53a3176af4e8f2969117d75a1616fb161d9ddf0"
DEFAULT_CONFIG = AnswerConfig(min_score=-1.0, max_evidence_chars=ANSWER_MAX_EVIDENCE_CHARS_MAX)
PROMPTS = RunPrompts(*prompt_templates())
EVIDENCE_MARKER = re.compile(r"<<<EVIDENCE (S\d+) ")


class StaticRetriever:
    def __init__(self, result):
        self.result = result

    async def retrieve(self, query):
        return self.result


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _tokens(text) -> int:
    return len(LabRegexTokenizer().token_spans(text))


def _dense(built, ledger):
    scope, index, _anchors, embedder, version = built
    return DenseRetriever(
        DenseConfig(k=5),
        scope=scope,
        index_version=version,
        index=index,
        embedder=RecordingEmbeddingModel(embedder, ledger),
    )


def _base_result(built):
    result = asyncio.run(_dense(built, UsageLedger()).retrieve(QUERY_EN))
    assert len(result.hits) == 5
    assert any(FACT_EN in hit.text for hit in result.hits)
    return result


def _answer(built, responder, *, retriever=None, chat=None, config=DEFAULT_CONFIG, **kwargs):
    scope, _index, anchors, _embedder, version = built
    # One ledger for retrieval and chat, as in the pre-step that pinned the default digests.
    ledger = UsageLedger()
    if retriever is None:
        retriever = StaticRetriever(asyncio.run(_dense(built, ledger).retrieve(QUERY_EN)))
    scripted = chat or ScriptedChatModel(responder)
    service = AnswerService(anchors, nonce_factory=lambda: "0123456789abcdef", run_id_factory=lambda: "fixed-run")
    response = asyncio.run(
        service.answer(
            scope,
            QUERY_EN,
            prompts=PROMPTS,
            retriever=retriever,
            technique_version="1.0.0",
            index_version=version,
            chat=RecordingChatModel(scripted, ledger),
            ledger=ledger,
            config=config,
            **kwargs,
        )
    )
    return response, scripted


def _markers_in_request(scripted):
    return EVIDENCE_MARKER.findall("\n".join(m.content for m in scripted.requests[0].messages))


def _window_for_s1(result) -> int:
    fixed = _tokens(PROMPTS.grounded.system_text + "\n" + QUERY_EN)
    return DEFAULT_CONFIG.max_output_tokens + fixed + _tokens(result.hits[0].text)


def _packer(**config):
    return ContextPacker(ContextPackConfig(prompt_overhead_tokens=0, per_item_overhead_tokens=0, **config))


def test_defaults_match_digests_pinned_before_the_change(built):
    response, scripted = _answer(built, cite_where(FACT_EN, FACT_EN))
    messages = [[m.role, m.content] for m in scripted.requests[0].messages]
    request = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()
    projection = {
        "status": response.status,
        "abstention_reason": response.abstention_reason,
        "answer": response.answer,
        "warnings": list(response.warnings),
        "evidence": [(e.marker, e.rank, e.score, e.used) for e in response.evidence],
        "citations": [c.marker for c in response.citations],
        "usage": response.usage.model_dump(mode="json"),
    }
    assert request == DEFAULT_REQUEST_SHA256
    assert _sha(projection) == DEFAULT_RESPONSE_SHA256


def test_lost_in_middle_order_in_the_prompt(built):
    packer = _packer(order="lost_in_middle", context_window_tokens=100000)
    response, scripted = _answer(built, cite_where(FACT_EN, FACT_EN), packer=packer)
    assert _markers_in_request(scripted) == ["S1", "S3", "S5", "S4", "S2"]
    assert all(e.used for e in response.evidence)
    assert "evidence_packed_dropped" not in response.warnings
    assert response.status == "answered"


def test_budget_keeps_only_s1(built):
    window = _window_for_s1(_base_result(built))
    response, scripted = _answer(
        built,
        lambda request: ("Port Sallow records sea levels every 17 minutes [S1]", "stop"),
        packer=_packer(context_window_tokens=window),
    )
    assert _markers_in_request(scripted) == ["S1"]
    assert [(e.marker, e.used) for e in response.evidence] == [
        ("S1", True),
        ("S2", False),
        ("S3", False),
        ("S4", False),
        ("S5", False),
    ]
    assert "evidence_packed_dropped" in response.warnings
    assert response.status == "answered"


def test_empty_pack_abstains_before_the_chat_call(built):
    window = _window_for_s1(_base_result(built)) - 1
    response, scripted = _answer(built, lambda request: ("ZEBRA", "stop"), packer=_packer(context_window_tokens=window))
    assert response.status == "abstained"
    assert response.abstention_reason == "no_evidence"
    assert "evidence_packing_empty" in response.warnings
    assert not scripted.requests
    assert not any(e.used for e in response.evidence)


def test_no_context_window_raises_before_the_chat_call(built):
    chat = ScriptedChatModel(lambda request: ("ZEBRA", "stop"))
    chat.profile = SimpleNamespace(
        id="fake-chat",
        provider="fake",
        model="fake-chat-v1",
        capabilities=SimpleNamespace(context_window=None),
    )
    with pytest.raises(IndexingError, match="packing_no_context_window"):
        _answer(built, None, chat=chat, packer=ContextPacker(ContextPackConfig()))
    assert not chat.requests


def _claim_warnings(built, text, check):
    dense = _dense(built, UsageLedger())
    response, _ = _answer(
        built, cite_where(FACT_EN, text), retriever=dense, config=AnswerConfig(min_score=0.0), citation_check=check
    )
    assert response.status == "answered"
    return [w for w in response.warnings if w.startswith("claim_")]


def test_citation_warnings(built):
    mixed = FACT_EN + " Tea is sold there"
    assert _claim_warnings(built, mixed, "warn") == ["claim_uncited", "claim_unsupported_lexical"]
    assert _claim_warnings(built, mixed, "off") == []
    assert _claim_warnings(built, "Port Sallow records sea levels every 15 minutes", "warn") == [
        "claim_unsupported_lexical"
    ]
    assert _claim_warnings(built, "Port Sallow records sea levels every 17 minutes", "warn") == []


def test_r26_sub_span_citations_resolve_exact(built):
    scope, _index, anchors, _embedder, _version = built
    retriever = CompressingRetriever(_dense(built, UsageLedger()), ExtractiveCompressor(ExtractiveCompressionConfig()))
    response, _ = _answer(built, cite_where(FACT_EN, FACT_EN), retriever=retriever, config=AnswerConfig(min_score=0.0))
    assert response.status == "answered"
    resolved = asyncio.run(anchors.resolve(scope, tuple(a for c in response.citations for a in c.anchors)))
    assert resolved and all(item.status == "exact" for item in resolved)
    assert any(FACT_EN in item.exact for item in resolved)
