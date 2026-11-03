"""Request and result types for the model ports. Framework-neutral, frozen, no vendor types."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_foundry_providers.usage import Usage


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChatMessage(_Frozen):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ToolSpec(_Frozen):
    name: str = Field(min_length=1)
    description: str
    parameters_schema: dict[str, Any]


class ToolCall(_Frozen):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class ChatRequest(_Frozen):
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0)
    tools: tuple[ToolSpec, ...] = ()
    response_schema: dict[str, Any] | None = None


class ChatResult(_Frozen):
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    structured: dict[str, Any] | None = None
    finish_reason: Literal["stop", "length", "tool_calls"]
    usage: Usage


class ChatChunk(_Frozen):
    delta: str
    usage: Usage | None = None


class EmbeddingResult(_Frozen):
    vectors: tuple[tuple[float, ...], ...]
    usage: Usage

    @model_validator(mode="after")
    def _equal_lengths(self) -> Self:
        if len({len(v) for v in self.vectors}) > 1:
            raise ValueError("all embedding vectors must have equal length")
        return self


class RerankItem(_Frozen):
    index: int = Field(ge=0)
    score: float


class RerankResult(_Frozen):
    items: tuple[RerankItem, ...]
    usage: Usage
