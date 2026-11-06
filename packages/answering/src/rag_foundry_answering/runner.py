"""Local, bounded composition point for one answer run."""

import asyncio
import logging
from collections.abc import Callable
from decimal import Decimal

from rag_foundry_contracts.runs import RunRequest, RunResponse
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError, RunError
from rag_foundry_domain.indexing import IndexState
from rag_foundry_domain.ports import IndexRepository
from rag_foundry_indexing import DenseConfig, DenseRetriever, NoRetrieval, NoRetrievalConfig
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.adapters.ollama import (
    OllamaChatModel,
    OllamaEmbeddingModel,
    ollama_chat_profile,
    ollama_embedding_profile,
)
from rag_foundry_providers.budget import BudgetedChatModel, BudgetedEmbeddingModel, RunBudget
from rag_foundry_providers.ledger import RecordingChatModel, RecordingEmbeddingModel, UsageLedger
from rag_foundry_providers.ports import ChatModel, EmbeddingModel
from rag_foundry_providers.resilience import ResilientChatModel, ResilientEmbeddingModel, RetryPolicy

from rag_foundry_answering.gate import default_min_score
from rag_foundry_answering.service import AnswerConfig, AnswerService

RUN_RETRY = RetryPolicy(max_attempts=2, attempt_timeout_s=120.0, retry_timeouts_for_local=False)
# Provisional until the live run measures them (ADR-0013 D7).
DEFAULT_MAX_CONCURRENT = 1
DEFAULT_RUN_TIMEOUT_S = 300.0
ModelFactory = Callable[[UsageLedger, RunBudget], tuple[ChatModel, EmbeddingModel]]

_RETRIEVERS = {"R01": NoRetrieval, "R02": DenseRetriever}
_LOG = logging.getLogger(__name__)


def ollama_model_factory(base_url: str, *, retry: RetryPolicy = RUN_RETRY) -> ModelFactory:
    """Build the local Resilient(Budgeted(Recording(adapter))) model chain."""

    def factory(ledger: UsageLedger, budget: RunBudget) -> tuple[ChatModel, EmbeddingModel]:
        chat = OllamaChatModel(ollama_chat_profile(), api_base=base_url)
        embedder = OllamaEmbeddingModel(ollama_embedding_profile(), api_base=base_url)
        return (
            ResilientChatModel(
                BudgetedChatModel(RecordingChatModel(chat, ledger), budget, default_max_output_tokens=512), retry
            ),
            ResilientEmbeddingModel(BudgetedEmbeddingModel(RecordingEmbeddingModel(embedder, ledger), budget), retry),
        )

    return factory


class LocalRunner:
    def __init__(
        self,
        index: IndexRepository,
        anchors: AnchorService,
        *,
        model_factory: ModelFactory,
        service: AnswerService | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        run_timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
        max_attempts: int = RUN_RETRY.max_attempts,
    ) -> None:
        self._index = index
        self._anchors = anchors
        self._model_factory = model_factory
        self._service = service or AnswerService(anchors)
        self._max_concurrent = max_concurrent
        self._run_timeout_s = run_timeout_s
        self._max_attempts = max_attempts
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
            if request.technique_id == "R02":
                if request.index_version_id is None:
                    raise RunError("index_version_required")
                version = await self._index.get_index_version(scope, request.index_version_id)
                if version.state is not IndexState.READY:
                    raise IndexingError("not_ready")
            elif request.index_version_id is not None:
                raise RunError("index_version_not_allowed")

            ledger = UsageLedger()
            budget = RunBudget(Decimal("0"), max_calls=2 * self._max_attempts)
            chat, embedder = self._model_factory(ledger, budget)
            if chat.profile.capabilities.local is not True or embedder.profile.capabilities.local is not True:
                raise RunError("non_local_model")
            min_score = request.min_score
            if version is not None and min_score is None:
                dim = embedder.profile.capabilities.embedding_dim
                min_score = default_min_score(embedder.profile.provider, embedder.profile.model, dim) if dim else None
                if min_score is None:
                    raise RunError("abstention_threshold_unset")
            if version is None:
                retriever = NoRetrieval(NoRetrievalConfig())
            else:
                retriever = DenseRetriever(
                    DenseConfig(k=request.k), scope=scope, index_version=version, index=self._index, embedder=embedder
                )
            try:
                async with asyncio.timeout(self._run_timeout_s):
                    response = await self._service.answer(
                        scope,
                        request.question,
                        retriever=retriever,
                        technique_version=retriever_cls.manifest.version,
                        index_version=version,
                        chat=chat,
                        ledger=ledger,
                        config=AnswerConfig(k=request.k, min_score=min_score, temperature=0.0),
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
            return response
        except Exception as exc:
            # Codes and type names only; never question, answer or exception text.
            _LOG.info("run status=error error=%s code=%s", type(exc).__name__, getattr(exc, "code", None))
            raise
        finally:
            self._active -= 1
