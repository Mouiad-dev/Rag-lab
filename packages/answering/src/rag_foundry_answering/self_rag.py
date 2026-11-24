"""R31 short-form Self-RAG inference.

This adaptation computes the retrieval threshold from normalized probabilities,
not the upstream implementation's raw-logprob ratio. Ollama exposes at most 20
alternatives; reflection tokens outside that set use the official -100 logprob
fallback. Retrieved passage text is untrusted and is stripped of reflection
markers before it is placed after the fixed prompt. The raw Self-RAG prompt has
no structural separation between instruction and passage, so passage text can
only be neutralised, not isolated: lines starting with "###" lose their leading
hashes so they cannot fake instruction or response turns.

The wide retrieval-decision distribution is requested in a separate one-token
call. Under the pinned greedy sampler its position 0 matches position 0 of the
following generation; this costs one additional model call.

Documented deviation from the official procedure (owner decision 2026-11-24):
with `citation_guard` on (default) the best-scored candidate is still selected
unchanged, but it is cited only when its own generated tokens show `[Relevant]`
at position 0 and a `[Fully supported]` or `[Partially supported]` token.
Otherwise the answer text is returned uncited with grounding "ungrounded".
"""

import math
import re
from collections.abc import Sequence
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain.errors import RunError
from rag_foundry_providers.capabilities import Capability
from rag_foundry_providers.messages import RawCompletionRequest
from rag_foundry_providers.ports import RawCompletionModel
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest
from rag_foundry_techniques.kinds import RetrievalResult, RetrievedChunk

REFLECTION_TOKENS = (
    "[No Retrieval]",
    "[Retrieval]",
    "[Continue to Use Evidence]",
    "[Irrelevant]",
    "[Relevant]",
    "<paragraph>",
    "</paragraph>",
    "[Utility:1]",
    "[Utility:2]",
    "[Utility:3]",
    "[Utility:4]",
    "[Utility:5]",
    "[Fully supported]",
    "[Partially supported]",
    "[No support / Contradictory]",
)
RETRIEVAL = ("[Retrieval]", "[No Retrieval]")
RELEVANCE = ("[Relevant]", "[Irrelevant]")
SUPPORT = ("[Fully supported]", "[Partially supported]", "[No support / Contradictory]")
UTILITY = ("[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]")
_UTILITY_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)
_FROZEN = ConfigDict(frozen=True, extra="forbid")


class SelfRagConfig(BaseModel):
    model_config = _FROZEN
    threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    w_rel: float = Field(default=1.0, ge=0.0)
    w_sup: float = Field(default=1.0, ge=0.0)
    w_use: float = Field(default=1.0, ge=0.0)
    use_seqscore: bool = True
    citation_guard: bool = True
    k: int = Field(default=5, ge=1, le=50)
    max_new_tokens: int = Field(default=100, ge=1)
    decision_top_logprobs: int = Field(default=20, ge=1, le=65536)
    evidence_top_logprobs: int = Field(default=20, ge=1, le=65536)
    supported_languages: tuple[str, ...] = ("en",)
    token_strings: tuple[str, ...] = REFLECTION_TOKENS
    token_ids: dict[str, int] = Field(
        default_factory=lambda: {
            "[No Retrieval]": 32000,
            "[Retrieval]": 32001,
            "[Continue to Use Evidence]": 32002,
            "[Irrelevant]": 32003,
            "[Relevant]": 32004,
            "<paragraph>": 32005,
            "</paragraph>": 32006,
            "[Utility:1]": 32007,
            "[Utility:2]": 32008,
            "[Utility:3]": 32009,
            "[Utility:4]": 32010,
            "[Utility:5]": 32011,
            "[Fully supported]": 32012,
            "[Partially supported]": 32013,
            "[No support / Contradictory]": 32014,
        }
    )


class SelfRagCitation(BaseModel):
    model_config = _FROZEN
    chunk_id: str
    extraction_view_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    block_ids: tuple[str, ...]


class SelfRagAnswer(BaseModel):
    model_config = _FROZEN
    status: Literal["answered", "abstained"]
    answer: str
    grounding: Literal["corpus", "ungrounded"]
    citations: tuple[SelfRagCitation, ...] = ()
    abstention_reason: str | None = None


class SelfRagWorkflow:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R31",
        kind=PluginKind.WORKFLOW,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=SelfRagConfig,
    )

    def __init__(self, config: SelfRagConfig) -> None:
        self.config = config
        self.last_trace: dict | None = None

    async def answer(self, question: str, *, retriever, raw_model: RawCompletionModel) -> SelfRagAnswer:
        self.last_trace = None
        if raw_model.profile.declared(Capability.RAW_LOGPROBS) is not True or set(self.config.token_strings) != set(
            REFLECTION_TOKENS
        ):
            raise RunError("selfrag_tokens_missing")
        if not question.strip():
            raise ValueError("question must not be empty")
        language_outside = any("\u0600" <= char <= "\u06ff" for char in question)
        trace: dict = {
            "language_outside_model": language_outside,
            "retrieval_decision": None,
            "retrieval_fallbacks": 0,
            "candidates": [],
            "calls": [],
            "scored_positions": 0,
            "positions_missing_group_tokens": 0,
            "passage_reflection_tokens_removed": 0,
            "passage_prompt_markers_removed": 0,
        }
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        decision = await self._complete(
            raw_model, prompt, trace, top_logprobs=self.config.decision_top_logprobs, max_output_tokens=1
        )
        first = decision.tokens[0] if decision.tokens else None
        first_map, match_mode = _reflection_top_map(first, self.config)
        trace["match_mode"] = match_mode
        missing_decision = all(token not in first_map for token in RETRIEVAL)
        trace["scored_positions"] += 1
        generation = None
        if missing_decision:
            trace["positions_missing_group_tokens"] += 1
            trace["retrieval_fallbacks"] += 2
            # Upstream tests the full generation, not the one-token decision call.
            generation = await self._complete(raw_model, prompt, trace)
            do_retrieve = any(_generated_reflection(token, RETRIEVAL[:1], self.config) for token in generation.tokens)
            p_ret = None
            rule = "undetermined_topk_generated_token"
        else:
            retrieval_lps = [_lp(first_map, token, trace, "retrieval") for token in RETRIEVAL]
            weights = [math.exp(lp) for lp in retrieval_lps]
            p_ret = weights[0] / sum(weights)
            do_retrieve = p_ret > self.config.threshold
            rule = "probability_threshold"
        trace["retrieval_decision"] = {
            "p_retrieval": p_ret,
            "p_no_retrieval": None if p_ret is None else 1.0 - p_ret,
            "threshold": self.config.threshold,
            "retrieve": do_retrieve,
            "rule": rule,
            "generated_text": decision.text,
            "logprobs": _token_rows(decision.tokens[:1]),
            "k": self.config.decision_top_logprobs,
            "top_logprobs": self.config.decision_top_logprobs,
            "alternatives_returned": len(first.top) if first is not None else 0,
            "match_mode": match_mode,
        }
        if not do_retrieve:
            if generation is None:
                generation = await self._complete(raw_model, prompt, trace)
            trace["answer_grounding"] = "ungrounded"
            self.last_trace = trace
            return SelfRagAnswer(status="answered", answer=_strip_reflections(generation.text), grounding="ungrounded")

        retrieved: RetrievalResult = await retriever.retrieve(question)
        scored: list[tuple[float, RetrievedChunk, str, dict, object]] = []
        if not retrieved.hits:
            trace["answer_grounding"] = "none"
            self.last_trace = trace
            return SelfRagAnswer(status="abstained", answer="", grounding="ungrounded", abstention_reason="no_passages")
        for hit in retrieved.hits[: self.config.k]:
            clean_text, removed, markers = _clean_passage(hit.text)
            trace["passage_reflection_tokens_removed"] += removed
            trace["passage_prompt_markers_removed"] += markers
            prompt = f"### Instruction:\n{question}\n\n### Response:\n[Retrieval]<paragraph>\n{clean_text}</paragraph>"
            candidate = await self._complete(raw_model, prompt, trace, top_logprobs=self.config.evidence_top_logprobs)
            part, valid = _score_candidate(candidate, trace, self.config)
            row = {
                "chunk_id": hit.chunk_id,
                "token_probabilities": _token_rows(candidate.tokens),
                "chosen": False,
                "match_mode": _reflection_top_map(candidate.tokens[0], self.config)[1],
                **part,
            }
            if not valid:
                row.update(total=None, chosen=False, error="selfrag_output_invalid")
                trace["candidates"].append(row)
                continue
            score = part["total"]
            scored.append((score, hit, candidate.text, row, candidate))
            trace["candidates"].append(row)
        if not scored:
            trace["answer_grounding"] = "none"
            self.last_trace = trace
            return SelfRagAnswer(
                status="abstained", answer="", grounding="ungrounded", abstention_reason="all_candidates_invalid"
            )
        scored.sort(key=lambda item: (-item[0], item[1].rank))
        best_score, hit, text, best_row, best_candidate = scored[0]
        best_row["chosen"] = True
        trace["chosen_chunk_id"] = hit.chunk_id
        trace["chosen_score"] = best_score
        reason = _citation_guard_reason(best_candidate, self.config) if self.config.citation_guard else None
        trace["citation_guard"] = {
            "enabled": self.config.citation_guard,
            "cited": reason is None,
            "reason": reason,
        }
        if reason is not None:
            trace["answer_grounding"] = "ungrounded"
            self.last_trace = trace
            return SelfRagAnswer(status="answered", answer=_strip_reflections(text), grounding="ungrounded")
        citation = SelfRagCitation(
            chunk_id=hit.chunk_id,
            extraction_view_id=hit.extraction_view_id,
            start=hit.start,
            end=hit.end,
            block_ids=hit.block_ids,
        )
        trace["answer_grounding"] = "corpus"
        self.last_trace = trace
        return SelfRagAnswer(
            status="answered", answer=_strip_reflections(text), grounding="corpus", citations=(citation,)
        )

    async def _complete(
        self,
        model: RawCompletionModel,
        prompt: str,
        trace: dict,
        *,
        top_logprobs: int = 20,
        max_output_tokens: int | None = None,
    ):
        result = await model.complete_raw(
            RawCompletionRequest(
                prompt=prompt,
                max_output_tokens=max_output_tokens or self.config.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                top_logprobs=top_logprobs,
            )
        )
        if not result.tokens or all(not token.top for token in result.tokens):
            raise RunError("selfrag_logprobs_unavailable")
        trace["calls"].append(
            {"usage": result.usage.model_dump(), "finish_reason": result.finish_reason, "top_logprobs": top_logprobs}
        )
        return result


def _top_map(token) -> dict[str, float]:
    return {} if token is None else {name: lp for name, lp in token.top}


def _reflection_top_map(token, config: SelfRagConfig) -> tuple[dict[str, float], str]:
    if token is None:
        return {}, "text"
    if token.token_id is not None or token.top_ids is not None:
        if token.top_ids is None:
            return {}, "id"
        by_id = {value: key for key, value in config.token_ids.items()}
        matched = {
            by_id[token_id]: pair[1]
            for token_id, pair in zip(token.top_ids, token.top, strict=True)
            if token_id in by_id
        }
        return matched, "id"
    return _top_map(token), "text"


def _generated_reflection(token, names: Sequence[str], config: SelfRagConfig) -> str | None:
    if token.token_id is not None:  # id wins: a text lookalike with another id never matches
        by_id = {value: key for key, value in config.token_ids.items()}
        name = by_id.get(token.token_id)
        return name if name in names else None
    return token.token if token.token in names else None


def _citation_guard_reason(candidate, config: SelfRagConfig) -> str | None:
    """Return why the chosen candidate may not be cited, judged from its generated tokens."""
    tokens = candidate.tokens
    irrelevant = not tokens or _generated_reflection(tokens[0], RELEVANCE[:1], config) is None
    support = next((name for name in (_generated_reflection(t, SUPPORT, config) for t in tokens) if name), None)
    if irrelevant:
        return "irrelevant"
    if support is None:
        return "no_support_token"
    if support == SUPPORT[2]:
        return "not_supported"
    return None


def _lp(top: dict[str, float], token: str, trace: dict, group: str) -> float:
    if token in top:
        return top[token]
    trace[f"{group}_fallbacks"] = trace.get(f"{group}_fallbacks", 0) + 1
    return -100.0


def _group_probs(top: dict[str, float], tokens: Sequence[str], trace: dict, group: str) -> list[float]:
    if not any(token in top for token in tokens):
        trace["positions_missing_group_tokens"] += 1
    lps = [_lp(top, token, trace, group) for token in tokens]
    values = [math.exp(value) for value in lps]
    denom = sum(values)
    return [value / denom for value in values]


def _score_candidate(result, trace: dict, config: SelfRagConfig) -> tuple[dict, bool]:
    tops = [_reflection_top_map(token, config)[0] for token in result.tokens]
    # Official code reads position 0's top-k whatever token was generated.
    trace["scored_positions"] += 1
    if not any(token in tops[0] for token in RELEVANCE):
        trace["positions_missing_group_tokens"] += 1
        return ({"relevance": None, "support": None, "utility": None, "sequence": None}, False)
    rel = _group_probs(tops[0], RELEVANCE, trace, "relevance")
    relevance = rel[0]
    support_at = next(
        (
            top
            for generated, top in zip(result.tokens, tops, strict=True)
            if _generated_reflection(generated, SUPPORT, config)
        ),
        None,
    )
    utility_at = next(
        (
            top
            for generated, top in zip(result.tokens, tops, strict=True)
            if _generated_reflection(generated, UTILITY, config)
        ),
        None,
    )
    support_score = utility_score = 0.0
    support_probs = utility_probs = None
    if support_at is not None:
        trace["scored_positions"] += 1
        support_probs = _group_probs(support_at, SUPPORT, trace, "support")
        support_score = support_probs[0] + 0.5 * support_probs[1]
    if utility_at is not None:
        trace["scored_positions"] += 1
        utility_probs = _group_probs(utility_at, UTILITY, trace, "utility")
        utility_score = sum(p * value for p, value in zip(utility_probs, _UTILITY_VALUES, strict=True))
    seqscore = math.exp(sum(token.logprob for token in result.tokens) / len(result.tokens))
    total = selfrag_score(relevance, support_score, utility_score, seqscore, config)
    return (
        {
            "relevance": relevance,
            "support": support_score,
            "utility": utility_score,
            "sequence": seqscore if config.use_seqscore else None,
            "support_state": "scored" if support_at is not None else "absent",
            "utility_state": "scored" if utility_at is not None else "absent",
            "total": total,
        },
        True,
    )


def selfrag_score(
    relevance: float, support: float, utility: float, sequence: float, config: SelfRagConfig | None = None
) -> float:
    cfg = config or SelfRagConfig()
    return cfg.w_rel * relevance + cfg.w_sup * support + cfg.w_use * utility + (sequence if cfg.use_seqscore else 0.0)


def _token_rows(tokens) -> list[dict]:
    return [
        {"token": t.token, "logprob": t.logprob, "top": [{"token": k, "logprob": v} for k, v in t.top]} for t in tokens
    ]


def _remove_reflections(text: str) -> tuple[str, int]:
    pattern = re.compile("|".join(re.escape(token) for token in sorted(REFLECTION_TOKENS, key=len, reverse=True)))
    found = list(pattern.finditer(text))
    return pattern.sub("", text), len(found)


def _clean_passage(text: str) -> tuple[str, int, int]:
    text, removed = _remove_reflections(text)
    text, markers = re.subn(r"(?m)^([ \t]*)#{3,}", r"\1", text)
    return text, removed, markers


def _strip_reflections(text: str) -> str:
    return _remove_reflections(text)[0].strip()
