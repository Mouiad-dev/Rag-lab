"""Run request and response contract models, version 1.1."""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, field_validator, model_validator

from rag_foundry_contracts.source import CONTRACT_VERSION, SourceAnchor, _Model, check_contract_version

MAX_QUESTION_CHARS = 2000
MAX_RUN_K = 20

Marker = Annotated[str, Field(pattern=r"^S[1-9][0-9]?$")]
TechniqueId = Annotated[str, Field(pattern=r"^[A-Z][0-9]{2}$")]
PromptTemplate = Annotated[str, Field(pattern=r"^[a-z0-9-]+-v[0-9]+$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
WarningCode = Annotated[str, Field(pattern=r"^[a-z_]{1,64}$")]


class RunStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    UNGROUNDED = "ungrounded"


class AbstentionReason(StrEnum):
    NO_EVIDENCE = "no_evidence"
    LOW_SCORE = "low_score"
    MODEL_DECLINED = "model_declined"
    TRUNCATED_OUTPUT = "truncated_output"
    UNCITED_ANSWER = "uncited_answer"
    INVALID_CITATION = "invalid_citation"
    CITATION_UNRESOLVED = "citation_unresolved"


class RunRequest(_Model):
    contract_version: str = CONTRACT_VERSION
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    technique_id: TechniqueId = "R02"
    index_version_id: str | None = Field(default=None, min_length=1, max_length=64)
    k: int = Field(default=5, ge=1, le=MAX_RUN_K)
    min_score: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)

    @field_validator("contract_version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        return check_contract_version(value)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be whitespace-only")
        return value


class Evidence(_Model):
    marker: Marker
    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)
    used: bool
    anchors: tuple[SourceAnchor, ...] = Field(min_length=1)


class Citation(_Model):
    marker: Marker
    anchors: tuple[SourceAnchor, ...] = Field(min_length=1)


class RunPins(_Model):
    technique_id: TechniqueId
    technique_version: str
    index_version_id: str | None
    embedding_key: str | None
    chat_profile_id: str
    chat_provider: str
    chat_model: str
    prompt_template: PromptTemplate
    min_score: float | None = Field(allow_inf_nan=False)
    k: int | None
    settings_sha256: Sha256 | None = None
    prompt_sha256: Sha256 | None = None


class RunUsage(_Model):
    model_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: Decimal | None


class RunResponse(_Model):
    contract_version: str = CONTRACT_VERSION
    run_id: str = Field(min_length=1, max_length=64)
    status: RunStatus
    abstention_reason: AbstentionReason | None = None
    answer: str | None = None
    citations: tuple[Citation, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    pins: RunPins
    usage: RunUsage
    warnings: tuple[WarningCode, ...] = ()

    @field_validator("contract_version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        return check_contract_version(value)

    @model_validator(mode="after")
    def _response_consistent(self) -> Self:
        markers = [item.marker for item in self.evidence]
        ranks = [item.rank for item in self.evidence]
        if len(markers) != len(set(markers)):
            raise ValueError("evidence markers must be unique")
        if len(ranks) != len(set(ranks)):
            raise ValueError("evidence ranks must be unique")
        if any(item.marker != f"S{item.rank}" for item in self.evidence):
            raise ValueError("evidence marker must match its rank")

        if self.status is RunStatus.ANSWERED:
            if self.abstention_reason is not None:
                raise ValueError("answered response cannot have an abstention reason")
            if self.answer is None or not self.answer.strip():
                raise ValueError("answered response requires a non-blank answer")
            if not self.citations:
                raise ValueError("answered response requires citations")
            used = {item.marker for item in self.evidence if item.used}
            citation_markers = [item.marker for item in self.citations]
            if any(marker not in used for marker in citation_markers):
                raise ValueError("citations must reference used evidence")
            if len(citation_markers) != len(set(citation_markers)):
                raise ValueError("citation markers must be unique")
        elif self.status is RunStatus.ABSTAINED:
            if self.abstention_reason is None:
                raise ValueError("abstained response requires a reason")
            if self.answer is not None or self.citations:
                raise ValueError("abstained response cannot contain answer or citations")
        else:
            if self.abstention_reason is not None:
                raise ValueError("ungrounded response cannot have an abstention reason")
            if self.answer is None or not self.answer.strip():
                raise ValueError("ungrounded response requires a non-blank answer")
            if self.citations or self.evidence:
                raise ValueError("ungrounded response cannot contain citations or evidence")
            if self.pins.index_version_id is not None:
                raise ValueError("ungrounded response cannot pin an index version")
        return self
