"""One bounded retrieval-to-answer run."""

import secrets
import uuid
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_contracts.runs import (
    AbstentionReason,
    Citation,
    Evidence,
    RunPins,
    RunResponse,
    RunStatus,
    RunUsage,
)
from rag_foundry_domain.errors import RunError
from rag_foundry_domain.indexing import IndexVersion
from rag_foundry_domain.runtime_settings import (
    ANSWER_K_DEFAULT,
    ANSWER_K_MAX,
    ANSWER_K_MIN,
    ANSWER_MAX_EVIDENCE_CHARS_MAX,
    ANSWER_MAX_EVIDENCE_CHARS_MIN,
    ANSWER_MAX_OUTPUT_TOKENS_MAX,
    ANSWER_MAX_OUTPUT_TOKENS_MIN,
    ANSWER_TEMPERATURE_MAX,
    ANSWER_TEMPERATURE_MIN,
    AnswerSettings,
)
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.ledger import UsageLedger
from rag_foundry_providers.ports import ChatModel
from rag_foundry_techniques.kinds import Retriever

from rag_foundry_answering.citations import check_answer, normalize_nonce_markers, parse_markers
from rag_foundry_answering.gate import apply_gate
from rag_foundry_answering.prompt import (
    GROUNDED_TEMPLATE_ID,
    UNGROUNDED_TEMPLATE_ID,
    build_grounded_request,
    build_ungrounded_request,
    nonce_conflicts,
)


class AnswerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(default=ANSWER_K_DEFAULT, ge=ANSWER_K_MIN, le=ANSWER_K_MAX)  # fixed: ADR-0014 D3
    min_score: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    max_output_tokens: int = Field(
        default=AnswerSettings().max_output_tokens, ge=ANSWER_MAX_OUTPUT_TOKENS_MIN, le=ANSWER_MAX_OUTPUT_TOKENS_MAX
    )  # fixed: ADR-0014 D3
    temperature: float | None = Field(
        default=AnswerSettings().temperature, ge=ANSWER_TEMPERATURE_MIN, le=ANSWER_TEMPERATURE_MAX
    )  # fixed: ADR-0014 D3
    max_evidence_chars: int = Field(
        default=AnswerSettings().max_evidence_chars, ge=ANSWER_MAX_EVIDENCE_CHARS_MIN, le=ANSWER_MAX_EVIDENCE_CHARS_MAX
    )  # fixed: ADR-0014 D3


class AnswerService:
    def __init__(
        self,
        anchors: AnchorService,
        *,
        nonce_factory: Callable[[], str] = lambda: secrets.token_hex(8),
        run_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        self._anchors = anchors
        self._nonce_factory = nonce_factory
        self._run_id_factory = run_id_factory

    async def answer(
        self,
        scope,
        question: str,
        *,
        retriever: Retriever,
        technique_version: str,
        index_version: IndexVersion | None,
        chat: ChatModel,
        ledger: UsageLedger,
        config: AnswerConfig,
        settings_sha256: str | None = None,
    ) -> RunResponse:
        if index_version is not None and config.min_score is None:
            raise RunError("abstention_threshold_unset")
        result = await retriever.retrieve(question)
        if result.grounding != "none" and index_version is None:
            raise RunError("index_version_required")
        texts = tuple(hit.text for hit in result.hits)
        nonce = ""
        for _ in range(8):
            nonce = self._nonce_factory()
            if not nonce_conflicts(nonce, question, texts):
                break
        else:
            raise RuntimeError("could not generate a non-conflicting nonce")

        profile = chat.profile
        pins = RunPins(
            technique_id=result.technique_id,
            technique_version=technique_version,
            index_version_id=index_version.id if index_version else None,
            embedding_key=index_version.embedding_key if index_version else None,
            chat_profile_id=profile.id,
            chat_provider=profile.provider,
            chat_model=profile.model,
            prompt_template=UNGROUNDED_TEMPLATE_ID if result.grounding == "none" else GROUNDED_TEMPLATE_ID,
            min_score=config.min_score if result.grounding != "none" else None,
            k=config.k if result.grounding != "none" else None,
            settings_sha256=settings_sha256,
        )

        def response(status, *, reason=None, answer=None, citations=(), evidence=(), warnings=()):
            usage = ledger.totals()
            return RunResponse(
                run_id=self._run_id_factory(),
                status=status,
                abstention_reason=reason,
                answer=answer,
                citations=tuple(citations),
                evidence=tuple(evidence),
                pins=pins,
                usage=RunUsage(
                    model_calls=usage.calls,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    estimated_cost_usd=usage.cost_usd,
                ),
                warnings=tuple(warnings),
            )

        if result.grounding == "none":
            request = build_ungrounded_request(
                question,
                nonce=nonce,
                max_output_tokens=config.max_output_tokens,
                temperature=config.temperature,
            )
            completion = await chat.complete(request)
            text = completion.text.strip()
            if not text:
                return response(RunStatus.ABSTAINED, reason=AbstentionReason.MODEL_DECLINED)
            warnings = []
            if completion.finish_reason == "length":
                warnings.append("truncated_output")
            if parse_markers(completion.text):
                warnings.append("unexpected_markers")
            return response(RunStatus.UNGROUNDED, answer=text, warnings=warnings)

        gate = apply_gate(result.hits, min_score=config.min_score, max_evidence_chars=config.max_evidence_chars)
        evidence: list[Evidence] = []
        anchors_by_marker = {}
        for hit in result.hits:
            marker = f"S{hit.rank}"
            hit_anchors = await self._anchors.anchors_for_span(scope, hit.extraction_view_id, hit.start, hit.end)
            anchors_by_marker[marker] = hit_anchors
            evidence.append(
                Evidence(
                    marker=marker,
                    rank=hit.rank,
                    score=hit.score,
                    used=hit in gate.used,
                    anchors=hit_anchors,
                )
            )
        if gate.outcome != "pass":
            reason = AbstentionReason.NO_EVIDENCE if gate.outcome == "no_evidence" else AbstentionReason.LOW_SCORE
            return response(RunStatus.ABSTAINED, reason=reason, evidence=evidence)

        request = build_grounded_request(
            question,
            tuple((f"S{hit.rank}", hit.text) for hit in gate.used),
            nonce=nonce,
            max_output_tokens=config.max_output_tokens,
            temperature=config.temperature,
        )
        completion = await chat.complete(request)
        text = normalize_nonce_markers(completion.text, nonce)
        check = check_answer(text, completion.finish_reason, {f"S{hit.rank}" for hit in gate.used})
        reason_by_outcome = {
            "declined": AbstentionReason.MODEL_DECLINED,
            "truncated": AbstentionReason.TRUNCATED_OUTPUT,
            "uncited": AbstentionReason.UNCITED_ANSWER,
            "invalid": AbstentionReason.INVALID_CITATION,
        }
        warnings = ["evidence_truncated"] if gate.truncated else []
        if check.outcome == "ok" and nonce in text:
            # The nonce outside an accepted marker form must never reach a response (ADR-0013 amendment).
            return response(
                RunStatus.ABSTAINED, reason=AbstentionReason.INVALID_CITATION, evidence=evidence, warnings=warnings
            )
        if check.outcome != "ok":
            return response(
                RunStatus.ABSTAINED, reason=reason_by_outcome[check.outcome], evidence=evidence, warnings=warnings
            )

        citations = []
        for marker in check.cited:
            resolved = await self._anchors.resolve(scope, anchors_by_marker[marker])
            if any(item.status != "exact" for item in resolved):
                return response(
                    RunStatus.ABSTAINED,
                    reason=AbstentionReason.CITATION_UNRESOLVED,
                    evidence=evidence,
                    warnings=warnings,
                )
            citations.append(Citation(marker=marker, anchors=anchors_by_marker[marker]))
        return response(
            RunStatus.ANSWERED,
            answer=text.strip(),
            citations=citations,
            evidence=evidence,
            warnings=warnings,
        )
