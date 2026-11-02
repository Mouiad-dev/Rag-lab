"""Provider-agnostic model layer: ports, profiles, registry and usage records (ADR-0001)."""

from rag_foundry_providers import contract_checks
from rag_foundry_providers.capabilities import Capabilities, Capability, ModelKind, ModelProfile
from rag_foundry_providers.errors import (
    CapabilityProblem,
    DuplicateProfileError,
    MissingCapabilityError,
    ProfileKindError,
    ProviderError,
    ProviderUnavailableError,
    UnknownProfileError,
)
from rag_foundry_providers.fake import (
    FakeChatModel,
    FakeEmbeddingModel,
    FakeReranker,
    StepClock,
    fake_chat_profile,
    fake_embedding_profile,
    fake_reranker_profile,
)
from rag_foundry_providers.ledger import RecordingChatModel, UsageLedger, UsageSink, UsageTotals
from rag_foundry_providers.messages import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResult,
    EmbeddingResult,
    RerankItem,
    RerankResult,
    ToolCall,
    ToolSpec,
)
from rag_foundry_providers.ports import ChatModel, EmbeddingModel, Reranker, check_chat_request
from rag_foundry_providers.registry import ProviderRegistry, StageRequirement
from rag_foundry_providers.usage import Operation, TokenSource, Usage, estimate_cost, measure_latency_ms

__all__ = [
    "Capabilities",
    "Capability",
    "CapabilityProblem",
    "ChatChunk",
    "ChatMessage",
    "ChatModel",
    "ChatRequest",
    "ChatResult",
    "DuplicateProfileError",
    "EmbeddingModel",
    "EmbeddingResult",
    "FakeChatModel",
    "FakeEmbeddingModel",
    "FakeReranker",
    "MissingCapabilityError",
    "ModelKind",
    "ModelProfile",
    "Operation",
    "ProfileKindError",
    "ProviderError",
    "ProviderRegistry",
    "ProviderUnavailableError",
    "RecordingChatModel",
    "RerankItem",
    "RerankResult",
    "Reranker",
    "StageRequirement",
    "StepClock",
    "TokenSource",
    "ToolCall",
    "ToolSpec",
    "UnknownProfileError",
    "Usage",
    "UsageLedger",
    "UsageSink",
    "UsageTotals",
    "check_chat_request",
    "contract_checks",
    "estimate_cost",
    "fake_chat_profile",
    "fake_embedding_profile",
    "fake_reranker_profile",
    "measure_latency_ms",
]
