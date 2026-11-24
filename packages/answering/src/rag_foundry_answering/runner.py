"""Local, bounded composition point for one answer run."""

import asyncio
import logging
from collections.abc import Callable
from decimal import Decimal

from rag_foundry_contracts.runs import RunRequest, RunResponse
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError, RunError
from rag_foundry_domain.indexing import IndexState
from rag_foundry_domain.prompts import CachedPromptTemplateStore, PromptTemplateCache, active_template
from rag_foundry_domain.runtime_settings import RunSettings, RuntimeSettings
from rag_foundry_domain.settings_store import RuntimeSettingsService
from rag_foundry_domain.uow import UnitOfWorkFactory
from rag_foundry_domain.uow_adapters import per_call_indexes
from rag_foundry_indexing import DenseConfig, DenseRetriever, NoRetrieval, NoRetrievalConfig
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.adapters.ollama import ollama_chat_profile, ollama_embedding_profile
from rag_foundry_providers.budget import BudgetedChatModel, BudgetedEmbeddingModel, RunBudget
from rag_foundry_providers.config import Endpoint
from rag_foundry_providers.factories import AdapterRegistry, default_adapter_registry
from rag_foundry_providers.ledger import RecordingChatModel, RecordingEmbeddingModel, UsageLedger
from rag_foundry_providers.ports import ChatModel, EmbeddingModel
from rag_foundry_providers.resilience import ResilientChatModel, ResilientEmbeddingModel, RetryPolicy

from rag_foundry_answering.gate import default_min_score
from rag_foundry_answering.prompt import validate_template
from rag_foundry_answering.service import AnswerConfig, AnswerService, RunPrompts


def retry_policy(run: RunSettings) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=run.max_attempts,
        attempt_timeout_s=run.attempt_timeout_s,
        retry_timeouts_for_local=run.retry_timeouts_for_local,
    )


ModelFactory = Callable[[UsageLedger, RunBudget, RetryPolicy, int], tuple[ChatModel, EmbeddingModel]]

_RETRIEVERS = {"R01": NoRetrieval, "R02": DenseRetriever}
_LOG = logging.getLogger(__name__)


def ollama_model_factory(base_url: str, *, adapters: AdapterRegistry | None = None) -> ModelFactory:
    """Build the local Resilient(Budgeted(Recording(adapter))) model chain."""
    endpoint = Endpoint(id="ollama", adapter="ollama", base_url=base_url, local=True)
    registry = adapters or default_adapter_registry()

    def factory(
        ledger: UsageLedger, budget: RunBudget, retry: RetryPolicy, max_output_tokens: int
    ) -> tuple[ChatModel, EmbeddingModel]:
        chat = registry.build(ollama_chat_profile(), endpoint)
        embedder = registry.build(ollama_embedding_profile(), endpoint)
        return (
            ResilientChatModel(
                BudgetedChatModel(
                    RecordingChatModel(chat, ledger), budget, default_max_output_tokens=max_output_tokens
                ),
                retry,
            ),
            ResilientEmbeddingModel(BudgetedEmbeddingModel(RecordingEmbeddingModel(embedder, ledger), budget), retry),
        )

    return factory


class LocalRunner:
    def __init__(
        self,
        uow: UnitOfWorkFactory,
        anchors: AnchorService,
        *,
        model_factory: ModelFactory,
        base: RuntimeSettings,
        file_keys: frozenset[str] = frozenset(),
        prompt_cache: PromptTemplateCache | None = None,
        service: AnswerService | None = None,
    ) -> None:
        self._uow = uow
        self._anchors = anchors
        self._model_factory = model_factory
        self._base, self._file_keys = base, frozenset(file_keys)
        self._cache = prompt_cache or PromptTemplateCache()
        self._service = service or AnswerService(anchors)
        self._max_concurrent = base.run.max_concurrent
        self._active = 0

    async def run(self, scope: Scope, request: RunRequest) -> RunResponse:
        """A disconnect does not cancel a run; its slot is held through completion or deadline (cancel at P8.4)."""
        if self._active >= self._max_concurrent:
            raise RunError("too_many_runs")
        self._active += 1
        try:
            retriever_cls = _RETRIEVERS.get(request.technique_id)
            if retriever_cls is None:
                raise RunError("unsupported_technique")
            version = None
            if request.technique_id == "R02" and request.index_version_id is None:
                raise RunError("index_version_required")
            if request.technique_id != "R02" and request.index_version_id is not None:
                raise RunError("index_version_not_allowed")
            async with self._uow(read_only=True) as uow:
                if request.technique_id == "R02":
                    version = await uow.indexes.get_index_version(scope, request.index_version_id)
                    if version.state is not IndexState.READY:
                        raise IndexingError("not_ready")
                snap = await RuntimeSettingsService(
                    self._base, file_keys=self._file_keys, repo=uow.settings
                ).run_snapshot(scope)
                prompts = CachedPromptTemplateStore(uow.prompts, self._cache)
                run_prompts = RunPrompts(
                    await active_template(prompts, "grounded"),
                    await active_template(prompts, "ungrounded"),
                )
            validate_template(run_prompts.grounded, "grounded")
            validate_template(run_prompts.ungrounded, "ungrounded")
            run_retry = retry_policy(snap.run)
            ledger = UsageLedger()
            budget = RunBudget(
                Decimal("0"), max_calls=2 * snap.run.max_attempts
            )  # fixed: ADR-0014 D3, two model calls per attempt
            chat, embedder = self._model_factory(ledger, budget, run_retry, snap.answer.max_output_tokens)
            if chat.profile.capabilities.local is not True or embedder.profile.capabilities.local is not True:
                raise RunError("non_local_model")
            min_score = request.min_score
            if version is not None and min_score is None:
                dim = embedder.profile.capabilities.embedding_dim
                min_score = (
                    default_min_score(embedder.profile.provider, embedder.profile.model, dim, snap.answer.min_scores)
                    if dim
                    else None
                )
                if min_score is None:
                    raise RunError("abstention_threshold_unset")
            if version is None:
                retriever = NoRetrieval(NoRetrievalConfig())
            else:
                retriever = DenseRetriever(
                    DenseConfig(k=request.k),
                    scope=scope,
                    index_version=version,
                    index=per_call_indexes(self._uow),
                    embedder=embedder,
                )
            try:
                async with asyncio.timeout(snap.run.deadline_s):
                    response = await self._service.answer(
                        scope,
                        request.question,
                        retriever=retriever,
                        technique_version=retriever_cls.manifest.version,
                        index_version=version,
                        chat=chat,
                        ledger=ledger,
                        config=AnswerConfig(
                            k=request.k,
                            min_score=min_score,
                            max_output_tokens=snap.answer.max_output_tokens,
                            temperature=snap.answer.temperature,
                            max_evidence_chars=snap.answer.max_evidence_chars,
                        ),
                        prompts=run_prompts,
                        settings_sha256=snap.sha256(),
                    )
            except TimeoutError:
                # Only the run deadline maps to run_timeout; other timeouts keep their own type.
                raise RunError("run_timeout") from None
            _LOG.info(
                "run_id=%s status=%s reason=%s model_calls=%d",
                response.run_id,
                response.status,
                response.abstention_reason,
                response.usage.model_calls,
            )
            _LOG.info("run_id=%s settings=%s", response.run_id, snap.effective_json())
            return response
        except Exception as exc:
            # Codes and type names only; never question, answer or exception text.
            _LOG.info("run status=error error=%s code=%s", type(exc).__name__, getattr(exc, "code", None))
            raise
        finally:
            self._active -= 1
