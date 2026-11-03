from decimal import Decimal

import pytest
from pydantic import ValidationError
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
from rag_foundry_providers.usage import Operation, TokenSource, Usage, estimate_cost, measure_latency_ms


def chat(**caps) -> ModelProfile:
    return ModelProfile(
        id="c1",
        provider="ollama",
        model="m",
        kind="chat",
        capabilities=Capabilities(**caps),
        cost_per_mtok_in=Decimal("3"),
        cost_per_mtok_out=Decimal("15"),
    )


def embed() -> ModelProfile:
    return ModelProfile(
        id="e1", provider="ollama", model="bge-m3", kind="embedding", capabilities=Capabilities(embedding_dim=1024)
    )


def rerank() -> ModelProfile:
    return ModelProfile(id="r1", provider="local", model="ce", kind="reranker")


def test_valid_profiles():
    assert chat(streaming=True).kind is ModelKind.CHAT
    assert embed().capabilities.embedding_dim == 1024
    assert rerank().kind is ModelKind.RERANKER


def test_profile_is_frozen():
    with pytest.raises(ValidationError):
        chat().id = "x"


@pytest.mark.parametrize("bad_id", ["", "Upper", "-x", "a b"])
def test_bad_ids(bad_id):
    with pytest.raises(ValidationError):
        ModelProfile(id=bad_id, provider="p", model="m", kind="chat")
    with pytest.raises(ValidationError):
        ModelProfile(id="ok", provider=bad_id, model="m", kind="chat")


def test_empty_model_rejected():
    with pytest.raises(ValidationError):
        ModelProfile(id="a", provider="p", model="", kind="chat")


def test_embedding_requires_dim():
    with pytest.raises(ValidationError, match="embedding_dim"):
        ModelProfile(id="e", provider="p", model="m", kind="embedding")


def test_dim_only_on_embedding():
    with pytest.raises(ValidationError, match="embedding_dim"):
        ModelProfile(id="c", provider="p", model="m", kind="chat", capabilities=Capabilities(embedding_dim=8))


@pytest.mark.parametrize("flag", ["streaming", "tool_calling", "structured_output", "vision"])
def test_chat_only_flags_rejected_on_non_chat(flag):
    with pytest.raises(ValidationError, match="chat-only"):
        ModelProfile(id="r", provider="p", model="m", kind="reranker", capabilities=Capabilities(**{flag: True}))
    # False and None are allowed
    ModelProfile(id="r", provider="p", model="m", kind="reranker", capabilities=Capabilities(**{flag: False}))


def test_api_key_rejected():
    with pytest.raises(ValidationError):
        ModelProfile(id="c", provider="p", model="m", kind="chat", api_key="secret")
    with pytest.raises(ValidationError):
        Capabilities(api_key="secret")


def test_capability_numeric_and_strict_validation():
    with pytest.raises(ValidationError):
        Capabilities(context_window=0)
    with pytest.raises(ValidationError):
        Capabilities(streaming="yes")
    with pytest.raises(ValidationError):
        ModelProfile(id="c", provider="p", model="m", kind="chat", cost_per_mtok_in=Decimal("-1"))


def test_capability_enum_has_ten_members():
    assert len(Capability) == 10


def test_declared_table():
    p = chat(streaming=True, tool_calling=False, local=True)
    assert p.declared(Capability.CHAT) is True
    assert p.declared(Capability.EMBEDDINGS) is False
    assert p.declared(Capability.RERANKING) is False
    assert p.declared(Capability.STREAMING) is True
    assert p.declared(Capability.TOOL_CALLING) is False
    assert p.declared(Capability.STRUCTURED_OUTPUT) is None
    assert p.declared(Capability.VISION) is None
    assert p.declared(Capability.TOKEN_EMBEDDINGS) is None
    assert p.declared(Capability.MULTIVECTOR) is None
    assert p.declared(Capability.LOCAL) is True
    e = embed()
    assert e.declared(Capability.CHAT) is False
    assert e.declared(Capability.EMBEDDINGS) is True
    assert e.declared(Capability.LOCAL) is False
    assert rerank().declared(Capability.RERANKING) is True
    for c in Capability:  # every capability answers, also when given as a plain string
        assert p.declared(c) == p.declared(c.value)


def test_usage_rejects_negative_cost_and_empty_capability_error():
    with pytest.raises(ValidationError):
        Usage(
            profile_id="p",
            provider="x",
            model="m",
            operation=Operation.CHAT,
            input_tokens=0,
            output_tokens=0,
            token_source=TokenSource.ESTIMATED,
            latency_ms=0.0,
            estimated_cost_usd=Decimal(-1),
        )
    with pytest.raises(ValueError):
        MissingCapabilityError(())


def test_estimate_cost_rejects_negative_tokens():
    with pytest.raises(ValueError):
        estimate_cost(chat(), -5, 0)


def test_estimate_cost_known_value():
    assert estimate_cost(chat(), 2_000_000, 1_000_000) == Decimal("21")
    assert estimate_cost(chat(), 1000, 0) == Decimal("0.003")


def test_estimate_cost_unknown_rate_is_none():
    unknown = ModelProfile(id="k", provider="p", model="m", kind="chat")
    assert estimate_cost(unknown, 10, 0) is None
    assert estimate_cost(unknown, 0, 10) is None
    only_in = ModelProfile(id="k", provider="p", model="m", kind="chat", cost_per_mtok_in=Decimal("1"))
    assert estimate_cost(only_in, 1_000_000, 0) == Decimal("1")
    assert estimate_cost(only_in, 1_000_000, 1) is None


def test_estimate_cost_zero_tokens():
    unknown = ModelProfile(id="k", provider="p", model="m", kind="chat")
    assert estimate_cost(unknown, 0, 0) == Decimal(0)
    assert estimate_cost(chat(), 0, 0) == Decimal(0)


def usage(**over) -> Usage:
    base = dict(
        profile_id="c1",
        provider="ollama",
        model="m",
        operation=Operation.CHAT,
        input_tokens=1,
        output_tokens=2,
        token_source=TokenSource.ESTIMATED,
        latency_ms=1.5,
    )
    return Usage(**{**base, **over})


def test_usage_valid_and_frozen():
    u = usage()
    assert u.estimated_cost_usd is None
    with pytest.raises(ValidationError):
        u.input_tokens = 5


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "latency_ms"])
def test_usage_rejects_negatives(field):
    with pytest.raises(ValidationError):
        usage(**{field: -1})


def test_token_source_closed_enum():
    assert {s.value for s in TokenSource} == {"provider", "estimated"}
    assert usage(token_source="provider").token_source is TokenSource.PROVIDER
    assert usage(token_source="estimated").token_source is TokenSource.ESTIMATED
    with pytest.raises(ValidationError):
        usage(token_source="guess")


def test_token_source_is_required():
    fields = dict(
        profile_id="c1",
        provider="p",
        model="m",
        operation="chat",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
    )
    with pytest.raises(ValidationError, match="token_source"):
        Usage(**fields)


def test_token_source_survives_serialization():
    dumped = usage(token_source="provider").model_dump(mode="json")
    assert dumped["token_source"] == "provider"
    assert Usage.model_validate(dumped).token_source is TokenSource.PROVIDER


def test_measure_latency_ms():
    assert measure_latency_ms(lambda: 10.25, 10.0) == pytest.approx(250.0)


def test_errors_name_ids_and_are_not_lookup_errors():
    problems = (
        CapabilityProblem("generator", "x", "tool_calling", False),
        CapabilityProblem(None, "y", "vision", None),
    )
    errors = [
        UnknownProfileError("p1"),
        DuplicateProfileError("p2"),
        ProfileKindError("p3", "chat", "embedding"),
        ProviderUnavailableError("p4"),
        MissingCapabilityError(problems),
    ]
    for err, pid in zip(errors, ["p1", "p2", "p3", "p4", "x"], strict=True):
        assert isinstance(err, ProviderError)
        assert not isinstance(err, KeyError | LookupError)
        assert pid in str(err)
        assert pid == getattr(err, "profile_id", pid)
    msg = str(errors[-1])
    assert "stage 'generator': profile 'x' lacks 'tool_calling' (declared false)" in msg
    assert "profile 'y' lacks 'vision' (not declared)" in msg
    assert errors[-1].problems == problems
    kind = errors[2]
    assert (kind.expected, kind.actual) == ("chat", "embedding")
