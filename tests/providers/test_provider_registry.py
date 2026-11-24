"""ProviderRegistry bindings and pre-job capability validation."""

from collections.abc import AsyncIterator, Sequence
from typing import Literal

import pytest
from rag_foundry_providers import (
    Capabilities,
    Capability,
    ChatChunk,
    ChatRequest,
    ChatResult,
    DuplicateProfileError,
    EmbeddingResult,
    MissingCapabilityError,
    ModelKind,
    ModelProfile,
    ProfileKindError,
    ProviderError,
    ProviderUnavailableError,
    UnknownProfileError,
)
from rag_foundry_providers.registry import ProviderRegistry, StageRequirement


def chat_profile(pid: str = "chat", **caps: bool | None) -> ModelProfile:
    return ModelProfile(id=pid, provider="fake", model="m", kind=ModelKind.CHAT, capabilities=Capabilities(**caps))


def embed_profile(pid: str = "emb") -> ModelProfile:
    return ModelProfile(
        id=pid, provider="fake", model="m", kind=ModelKind.EMBEDDING, capabilities=Capabilities(embedding_dim=4)
    )


class StubChat:
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    async def complete(self, request: ChatRequest) -> ChatResult:
        raise NotImplementedError

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        raise NotImplementedError


class StubEmbed:
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    async def embed(
        self, texts: Sequence[str], *, input_type: Literal["document", "query"] = "document"
    ) -> EmbeddingResult:
        raise NotImplementedError


def make_registry() -> ProviderRegistry:
    reg = ProviderRegistry()
    reg.add_profile(chat_profile("chat"))
    reg.add_profile(embed_profile("emb"))
    return reg


def test_add_get_and_sorted_ids() -> None:
    reg = ProviderRegistry()
    reg.add_profile(chat_profile("b"))
    reg.add_profile(chat_profile("a"))
    assert reg.profile_ids() == ("a", "b")
    assert reg.profile("a").id == "a"


def test_duplicate_and_unknown_profile() -> None:
    reg = make_registry()
    with pytest.raises(DuplicateProfileError):
        reg.add_profile(chat_profile("chat"))
    with pytest.raises(UnknownProfileError):
        reg.profile("nope")
    with pytest.raises(UnknownProfileError):
        reg.bind("nope", StubChat(chat_profile("nope")))
    with pytest.raises(UnknownProfileError):
        reg.chat("nope")


def test_bind_and_retrieve() -> None:
    reg = make_registry()
    chat, emb = StubChat(reg.profile("chat")), StubEmbed(reg.profile("emb"))
    reg.bind("chat", chat)
    reg.bind("emb", emb)
    assert reg.chat("chat") is chat
    assert reg.embedding("emb") is emb


def test_duplicate_bind() -> None:
    reg = make_registry()
    reg.bind("chat", StubChat(reg.profile("chat")))
    with pytest.raises(DuplicateProfileError):
        reg.bind("chat", StubChat(reg.profile("chat")))


def test_bind_profile_mismatch() -> None:
    reg = make_registry()
    other = chat_profile("chat", streaming=True)
    with pytest.raises(ProviderError) as exc:
        reg.bind("chat", StubChat(other))
    assert not isinstance(exc.value, (ProfileKindError, DuplicateProfileError, UnknownProfileError))


def test_bind_kind_mismatch() -> None:
    reg = make_registry()
    with pytest.raises(ProfileKindError):
        reg.bind("emb", StubChat(reg.profile("emb")))


def test_wrong_kind_accessor() -> None:
    reg = make_registry()
    with pytest.raises(ProfileKindError):
        reg.embedding("chat")
    with pytest.raises(ProfileKindError):
        reg.chat("emb")
    with pytest.raises(ProfileKindError):
        reg.reranker("chat")


def test_unbound_is_unavailable_with_id() -> None:
    reg = make_registry()
    for call, pid in ((reg.chat, "chat"), (reg.embedding, "emb")):
        with pytest.raises(ProviderUnavailableError, match=pid):
            call(pid)
    reg.add_profile(ModelProfile(id="rr", provider="fake", model="m", kind=ModelKind.RERANKER))
    with pytest.raises(ProviderUnavailableError, match="rr"):
        reg.reranker("rr")


def test_tool_calling_none_and_false_rejected() -> None:
    reg = ProviderRegistry()
    reg.add_profile(chat_profile("undeclared"))
    reg.add_profile(chat_profile("off", tool_calling=False))
    for pid, declared in (("undeclared", None), ("off", False)):
        reqs = {"agent": StageRequirement(pid, frozenset({Capability.TOOL_CALLING}))}
        with pytest.raises(MissingCapabilityError) as exc:
            reg.validate_requirements(reqs)
        (problem,) = exc.value.problems
        assert problem.declared is declared
        assert (problem.stage, problem.profile_id, problem.capability) == ("agent", pid, "tool_calling")


def test_all_problems_in_one_stable_error() -> None:
    reg = ProviderRegistry()
    reg.add_profile(chat_profile("c"))
    reg.add_profile(chat_profile("d", vision=False))
    reqs = {
        "z_stage": StageRequirement("c", frozenset({Capability.VISION, Capability.TOOL_CALLING})),
        "a_stage": StageRequirement("d", frozenset({Capability.VISION})),
    }
    with pytest.raises(MissingCapabilityError) as exc:
        reg.validate_requirements(reqs)
    got = [(p.stage, p.capability) for p in exc.value.problems]
    assert got == [("a_stage", "vision"), ("z_stage", "tool_calling"), ("z_stage", "vision")]
    assert "a_stage" in str(exc.value) and "z_stage" in str(exc.value)


def test_satisfied_returns_none_without_adapters() -> None:
    reg = ProviderRegistry()
    reg.add_profile(chat_profile("c", streaming=True, tool_calling=True))
    reqs = {"s": StageRequirement("c", frozenset({Capability.CHAT, Capability.STREAMING, Capability.TOOL_CALLING}))}
    assert reg.validate_requirements(reqs) is None
    assert reg.validate_requirements({}) is None


def test_local_required_on_non_local_rejected() -> None:
    reg = ProviderRegistry()
    reg.add_profile(chat_profile("remote"))
    reg.add_profile(chat_profile("local", local=True))
    ok = {"s": StageRequirement("local", frozenset({Capability.LOCAL}))}
    assert reg.validate_requirements(ok) is None
    with pytest.raises(MissingCapabilityError) as exc:
        reg.validate_requirements({"s": StageRequirement("remote", frozenset({Capability.LOCAL}))})
    assert exc.value.problems[0].capability == "local"
    assert exc.value.problems[0].declared is False


def test_unknown_profile_in_requirements_raises_first() -> None:
    reg = ProviderRegistry()
    reg.add_profile(chat_profile("c"))
    reqs = {
        "a": StageRequirement("c", frozenset({Capability.VISION})),
        "b": StageRequirement("ghost", frozenset({Capability.CHAT})),
    }
    with pytest.raises(UnknownProfileError):
        reg.validate_requirements(reqs)


def test_no_error_is_a_lookup_error() -> None:
    reg = make_registry()
    errors: list[Exception] = []
    for action in (
        lambda: reg.profile("x"),
        lambda: reg.add_profile(chat_profile("chat")),
        lambda: reg.chat("chat"),
        lambda: reg.embedding("chat"),
        lambda: reg.validate_requirements({"s": StageRequirement("chat", frozenset({Capability.VISION}))}),
    ):
        with pytest.raises(ProviderError) as exc:
            action()
        errors.append(exc.value)
    assert errors
    assert not any(isinstance(e, (KeyError, LookupError)) for e in errors)


def test_stage_requirement_accepts_capability_strings():
    req = StageRequirement(profile_id="p", required=frozenset({"chat", "streaming"}))
    assert req.required == frozenset({Capability.CHAT, Capability.STREAMING})
