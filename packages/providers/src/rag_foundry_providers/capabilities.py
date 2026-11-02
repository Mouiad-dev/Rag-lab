"""Model profiles and honest tri-state capability flags."""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"


class ModelKind(StrEnum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANKER = "reranker"


class Capability(StrEnum):
    CHAT = "chat"
    STREAMING = "streaming"
    TOOL_CALLING = "tool_calling"
    STRUCTURED_OUTPUT = "structured_output"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    TOKEN_EMBEDDINGS = "token_embeddings"
    MULTIVECTOR = "multivector"
    RERANKING = "reranking"
    LOCAL = "local"


_CHAT_ONLY = ("streaming", "tool_calling", "structured_output", "vision")

PositiveInt = Annotated[int, Field(gt=0)]


class Capabilities(BaseModel):
    """Declared capabilities. None means not declared and never counts as supported."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    streaming: bool | None = None
    tool_calling: bool | None = None
    structured_output: bool | None = None
    vision: bool | None = None
    token_embeddings: bool | None = None
    multivector: bool | None = None
    local: bool = False
    context_window: PositiveInt | None = None
    max_output_tokens: PositiveInt | None = None
    embedding_dim: PositiveInt | None = None


class ModelProfile(BaseModel):
    """A model profile. Holds no secrets; extra fields such as api_key are rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=_ID_PATTERN)
    provider: str = Field(pattern=_ID_PATTERN)
    model: str = Field(min_length=1)
    kind: ModelKind
    capabilities: Capabilities = Field(default_factory=Capabilities)
    cost_per_mtok_in: Decimal | None = Field(default=None, ge=0)
    cost_per_mtok_out: Decimal | None = Field(default=None, ge=0)
    cost_per_mtok_cache_read: Decimal | None = Field(default=None, ge=0)
    cost_per_mtok_cache_write: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_kind(self) -> Self:
        has_dim = self.capabilities.embedding_dim is not None
        is_embedding = self.kind is ModelKind.EMBEDDING
        if is_embedding and not has_dim:
            raise ValueError(f"profile '{self.id}': embedding profiles require capabilities.embedding_dim")
        if has_dim and not is_embedding:
            raise ValueError(f"profile '{self.id}': embedding_dim is only allowed on embedding profiles")
        if self.kind is not ModelKind.CHAT:
            bad = [f for f in _CHAT_ONLY if getattr(self.capabilities, f) is True]
            if bad:
                raise ValueError(f"profile '{self.id}': chat-only flags {bad} cannot be true on kind '{self.kind}'")
        return self

    def declared(self, cap: Capability) -> bool | None:
        """True or False when known, None when not declared."""
        cap = Capability(cap)  # accept config strings such as "chat"
        if cap is Capability.CHAT:
            return self.kind is ModelKind.CHAT
        if cap is Capability.EMBEDDINGS:
            return self.kind is ModelKind.EMBEDDING
        if cap is Capability.RERANKING:
            return self.kind is ModelKind.RERANKER
        if cap is Capability.LOCAL:
            return self.capabilities.local
        return getattr(self.capabilities, cap.value)
