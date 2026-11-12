"""Per-call chat timing: total latency, time to first token (streams only) and cold/warm marking."""

import re
import time
from collections.abc import AsyncIterator, Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_providers.capabilities import ModelProfile
from rag_foundry_providers.messages import ChatChunk, ChatRequest, ChatResult

from .errors import EvaluationError


class CallTiming(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: str = Field(pattern=r"^[a-z_]{1,32}$")
    profile_id: str
    mode: Literal["complete", "stream"]
    total_ms: float = Field(ge=0)
    ttft_ms: float | None = None
    cold: bool
    ok: bool


class TimingRecorder:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._records: list[CallTiming] = []
        self._seen: set[tuple[str, str]] = set()

    def wrap(self, model, stage: str) -> "TimingChatModel":
        if not re.fullmatch(r"[a-z_]{1,32}", stage):
            raise EvaluationError("invalid_stage", stage)
        return TimingChatModel(model, stage, self)

    def records(self) -> tuple[CallTiming, ...]:
        return tuple(self._records)

    def _start(self, stage: str, profile_id: str) -> bool:
        key = (stage, profile_id)
        cold = key not in self._seen
        self._seen.add(key)
        return cold

    def _add(self, **fields) -> None:
        self._records.append(CallTiming(**fields))


class TimingChatModel:
    def __init__(self, model, stage: str, recorder: TimingRecorder) -> None:
        self._model = model
        self._stage = stage
        self._recorder = recorder

    @property
    def profile(self) -> ModelProfile:
        return self._model.profile

    def _record(self, mode, t0, end, first, cold, ok) -> None:
        self._recorder._add(
            stage=self._stage,
            profile_id=self.profile.id,
            mode=mode,
            total_ms=round((end - t0) * 1000, 3),
            ttft_ms=None if first is None else round((first - t0) * 1000, 3),
            cold=cold,
            ok=ok,
        )

    async def complete(self, request: ChatRequest) -> ChatResult:
        cold = self._recorder._start(self._stage, self.profile.id)
        clock = self._recorder._clock
        t0 = clock()
        try:
            result = await self._model.complete(request)
        except BaseException:
            self._record("complete", t0, clock(), None, cold, False)
            raise
        self._record("complete", t0, clock(), None, cold, True)
        return result

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        cold = self._recorder._start(self._stage, self.profile.id)
        clock = self._recorder._clock
        t0 = clock()
        first: float | None = None
        saw_usage = False
        try:
            async for chunk in self._model.stream(request):
                if first is None and chunk.delta != "":
                    first = clock()
                if chunk.usage is not None:
                    saw_usage = True
                yield chunk
        finally:
            # ok means a usage-carrying chunk was seen (spec rule); an error raised after
            # that chunk still records ok=True, and the error propagates to the caller.
            self._record("stream", t0, clock(), first, cold, saw_usage)
