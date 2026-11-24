"""Provider registry: profiles, adapter bindings and pre-job capability validation.

No config loading, no default profiles, no global state, and no fallback to another profile.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from rag_foundry_providers.capabilities import Capability, ModelKind, ModelProfile
from rag_foundry_providers.errors import (
    CapabilityProblem,
    DuplicateProfileError,
    MissingCapabilityError,
    ProfileKindError,
    ProviderError,
    ProviderUnavailableError,
    UnknownProfileError,
)
from rag_foundry_providers.ports import ChatModel, EmbeddingModel, Reranker

_PORTS: dict[ModelKind, type] = {
    ModelKind.CHAT: ChatModel,
    ModelKind.EMBEDDING: EmbeddingModel,
    ModelKind.RERANKER: Reranker,
}


@dataclass(frozen=True)
class StageRequirement:
    """Capabilities one pipeline stage needs from one model profile."""

    profile_id: str
    required: frozenset[Capability]

    def __post_init__(self) -> None:
        # Accept config strings such as "streaming", like ModelProfile.declared() does.
        object.__setattr__(self, "required", frozenset(Capability(c) for c in self.required))


class ProviderRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, ModelProfile] = {}
        self._bound: dict[str, object] = {}

    def add_profile(self, profile: ModelProfile) -> None:
        if profile.id in self._profiles:
            raise DuplicateProfileError(profile.id)
        self._profiles[profile.id] = profile

    def profile(self, pid: str) -> ModelProfile:
        try:
            return self._profiles[pid]
        except KeyError:
            raise UnknownProfileError(pid) from None

    def profile_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def bind(self, pid: str, model: object) -> None:
        profile = self.profile(pid)
        if pid in self._bound:
            raise DuplicateProfileError(pid)
        if getattr(model, "profile", None) != profile:
            raise ProviderError(f"adapter profile does not match the registered declaration for '{pid}'")
        if not isinstance(model, _PORTS[profile.kind]):
            raise ProfileKindError(pid, expected=profile.kind.value, actual=type(model).__name__)
        self._bound[pid] = model

    def _get(self, pid: str, kind: ModelKind) -> object:
        profile = self.profile(pid)
        if profile.kind is not kind:
            raise ProfileKindError(pid, expected=kind.value, actual=profile.kind.value)
        try:
            return self._bound[pid]
        except KeyError:
            raise ProviderUnavailableError(pid) from None

    def chat(self, pid: str) -> ChatModel:
        model = self._get(pid, ModelKind.CHAT)
        assert isinstance(model, ChatModel)
        return model

    def embedding(self, pid: str) -> EmbeddingModel:
        model = self._get(pid, ModelKind.EMBEDDING)
        assert isinstance(model, EmbeddingModel)
        return model

    def reranker(self, pid: str) -> Reranker:
        model = self._get(pid, ModelKind.RERANKER)
        assert isinstance(model, Reranker)
        return model

    def validate_requirements(self, reqs: Mapping[str, StageRequirement]) -> None:
        """Check every stage before a job starts; raise one error listing all gaps."""
        for stage in sorted(reqs):
            self.profile(reqs[stage].profile_id)
        problems: list[CapabilityProblem] = []
        for stage in sorted(reqs):
            req = reqs[stage]
            profile = self._profiles[req.profile_id]
            for cap in sorted(req.required, key=lambda c: c.value):
                declared = profile.declared(cap)
                if declared is not True:
                    problems.append(
                        CapabilityProblem(stage=stage, profile_id=profile.id, capability=cap.value, declared=declared)
                    )
        if problems:
            raise MissingCapabilityError(tuple(problems))
