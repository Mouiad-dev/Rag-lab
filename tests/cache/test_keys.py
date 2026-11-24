import pytest
from rag_foundry_cache.keys import (
    GenerationParams,
    ResponseKeyInputs,
    exact_key,
    partition_key,
    source_version_ids,
)
from rag_foundry_contracts.runs import Citation, Evidence, RunPins, RunResponse, RunUsage
from rag_foundry_contracts.source import Precision, SourceAnchor, SourceRef
from rag_foundry_domain.documents import Scope
from rag_foundry_domain.errors import CacheError

SCOPE = Scope(tenant_id="7f3c2b1e-5d4a-4c8b-9e6f-1a2b3c4d5e6f", project_id="2d9e8f7a-6b5c-4a3d-8e2f-0c1b2a3d4e5f")
OTHER_SCOPE = Scope(tenant_id="11111111-2222-4333-8444-555555555555", project_id=SCOPE.project_id)
QUESTION = "What is the SECRET-QUESTION limit?"


def generation(**over):
    data = {
        "max_output_tokens": 512,
        "temperature": 0.0,
        "k": 5,
        "min_score": None,
        "max_evidence_chars": 8000,
        "packer": "none",
        "citation_check": "warn",
    }
    data.update(over)
    return GenerationParams(**data)


def inputs(**over):
    data = {
        "authz_revision": "authz-1",
        "history": (("user", "hello"), ("assistant", "hi")),
        "technique_id": "R02",
        "technique_version": "1",
        "retriever_config_sha256": "a" * 64,
        "index_version_id": "iv-1",
        "embedding_key": "emb-1",
        "prompt_template": "grounded-v1",
        "prompt_sha256": "b" * 64,
        "chat_profile_id": "local",
        "chat_provider": "ollama",
        "chat_model": "m1",
        "generation": generation(),
        "settings_sha256": None,
        "mode": "source_only",
        "freshness": "static_snapshot",
        "tools": (),
        "is_action_run": False,
        "output_language": "auto",
        "contract_version": "1.1",
    }
    data.update(over)
    return ResponseKeyInputs(**data)


CHANGES = [
    ("authz_revision", "authz-2"),
    ("history", (("user", "hello"),)),
    ("history", (("user", "hello"), ("assistant", "hi!"))),
    ("history", (("assistant", "hello"), ("assistant", "hi"))),
    ("technique_id", "R03"),
    ("technique_version", "2"),
    ("retriever_config_sha256", "c" * 64),
    ("index_version_id", "iv-2"),
    ("index_version_id", None),
    ("embedding_key", "emb-2"),
    ("embedding_key", None),
    ("prompt_template", "grounded-v2"),
    ("prompt_sha256", "d" * 64),
    ("chat_profile_id", "other"),
    ("chat_provider", "anthropic"),
    ("chat_model", "m2"),
    ("settings_sha256", "e" * 64),
    ("mode", "enriched"),
    ("freshness", "current"),
    ("tools", ("search",)),
    ("is_action_run", True),
    ("output_language", "ar"),
    ("contract_version", "1.2"),
]
GENERATION_CHANGES = [
    ("max_output_tokens", 513),
    ("temperature", 0.5),
    ("temperature", None),
    ("k", 6),
    ("min_score", 0.1),
    ("max_evidence_chars", 8001),
    ("packer", "f" * 64),
    ("citation_check", "off"),
]


def test_changes_cover_every_field():
    assert {name for name, _ in CHANGES} == set(ResponseKeyInputs.model_fields) - {"generation"}
    assert {name for name, _ in GENERATION_CHANGES} == set(GenerationParams.model_fields)


def test_key_is_deterministic_sha256_hex():
    key = exact_key(SCOPE, QUESTION, inputs())
    assert key == exact_key(SCOPE, QUESTION, inputs())
    assert len(key) == 64 and set(key) <= set("0123456789abcdef")
    assert key != partition_key(SCOPE, inputs())


@pytest.mark.parametrize(("name", "value"), CHANGES)
def test_every_input_changes_both_keys(name, value):
    base, changed = inputs(), inputs(**{name: value})
    assert exact_key(SCOPE, QUESTION, base) != exact_key(SCOPE, QUESTION, changed)
    assert partition_key(SCOPE, base) != partition_key(SCOPE, changed)


@pytest.mark.parametrize(("name", "value"), GENERATION_CHANGES)
def test_every_generation_param_changes_keys(name, value):
    base, changed = inputs(), inputs(generation=generation(**{name: value}))
    assert exact_key(SCOPE, QUESTION, base) != exact_key(SCOPE, QUESTION, changed)
    assert partition_key(SCOPE, base) != partition_key(SCOPE, changed)


def test_scope_changes_keys():
    assert exact_key(SCOPE, QUESTION, inputs()) != exact_key(OTHER_SCOPE, QUESTION, inputs())
    assert partition_key(SCOPE, inputs()) != partition_key(OTHER_SCOPE, inputs())
    project = Scope(tenant_id=SCOPE.tenant_id, project_id=OTHER_SCOPE.tenant_id)
    assert exact_key(SCOPE, QUESTION, inputs()) != exact_key(project, QUESTION, inputs())


def test_partition_ignores_only_the_question():
    assert partition_key(SCOPE, inputs()) == partition_key(SCOPE, inputs())
    assert exact_key(SCOPE, "one", inputs()) != exact_key(SCOPE, "two", inputs())


@pytest.mark.parametrize(
    "other",
    [QUESTION.lower(), QUESTION + " ", " " + QUESTION, QUESTION.replace(" ", "  "), QUESTION + "?", QUESTION[:-1]],
)
def test_no_case_whitespace_or_punctuation_folding(other):
    assert exact_key(SCOPE, QUESTION, inputs()) != exact_key(SCOPE, other, inputs())


def test_nfc_only_normalization():
    composed, decomposed = "café", "café"
    assert exact_key(SCOPE, composed, inputs()) == exact_key(SCOPE, decomposed, inputs())
    one = exact_key(SCOPE, "q", inputs(chat_model=composed))
    assert one == exact_key(SCOPE, "q", inputs(chat_model=decomposed))


def test_arabic_diacritics_differ():
    plain = "ما هي السياسة"
    marked = "مَا هي السياسة"
    assert exact_key(SCOPE, plain, inputs()) != exact_key(SCOPE, marked, inputs())


@pytest.mark.parametrize("question", ["", " ", "\n\t "])
def test_blank_question_is_rejected_without_content(question):
    with pytest.raises(CacheError) as info:
        exact_key(SCOPE, question, inputs())
    assert info.value.code == "cache_key_invalid"


def test_question_and_history_never_appear_in_key_or_error():
    key = exact_key(SCOPE, QUESTION, inputs())
    assert "SECRET" not in key and QUESTION not in key and "hello" not in key
    with pytest.raises(CacheError) as info:
        exact_key(SCOPE, QUESTION, "not inputs")  # type: ignore[arg-type]
    assert QUESTION not in str(info.value) and QUESTION not in repr(info.value)


def test_inputs_have_no_defaults_and_reject_extras():
    for name, field in ResponseKeyInputs.model_fields.items():
        assert field.is_required(), name
    for name, field in GenerationParams.model_fields.items():
        assert field.is_required(), name
    assert len(GenerationParams.model_fields) == 7
    data = {name: getattr(inputs(), name) for name in ResponseKeyInputs.model_fields}
    with pytest.raises(ValueError):
        ResponseKeyInputs(**data, extra_field="x")
    del data["tools"]
    with pytest.raises(ValueError):
        ResponseKeyInputs(**data)


@pytest.mark.parametrize(
    "bad",
    [
        {"authz_revision": ""},
        {"authz_revision": "  "},
        {"history": (("system", "x"),)},
        {"history": (("user", " "),)},
        {"mode": "other"},
        {"index_version_id": ""},
        {"tools": ("",)},
    ],
)
def test_inputs_reject_invalid(bad):
    with pytest.raises(ValueError):
        inputs(**bad)


@pytest.mark.parametrize("bad", [{"temperature": float("nan")}, {"min_score": float("inf")}, {"k": 0}, {"packer": ""}])
def test_generation_rejects_invalid(bad):
    with pytest.raises(ValueError):
        generation(**bad)


def anchor(version_id, anchor_id="an-1"):
    return SourceAnchor(
        anchor_id=anchor_id,
        source=SourceRef(
            provider_namespace="ns",
            document_id="doc-" + version_id,
            document_version_id=version_id,
            extraction_view_id="view-1",
        ),
        block_id="blk",
        precision=Precision.PAGE_ONLY,
        page_index=0,
    )


def test_source_version_ids_uses_document_version_id():
    evidence = (
        Evidence(marker="S1", rank=1, score=0.9, used=True, anchors=(anchor("dv-b"), anchor("dv-a", "an-2"))),
        Evidence(marker="S2", rank=2, score=0.8, used=True, anchors=(anchor("dv-c", "an-3"),)),
        Evidence(marker="S3", rank=3, score=0.1, used=False, anchors=(anchor("dv-d", "an-4"),)),
    )
    response = RunResponse(
        run_id="run-1",
        status="answered",
        answer="x [S1]",
        citations=(Citation(marker="S1", anchors=(anchor("dv-b"),)), Citation(marker="S2", anchors=(anchor("dv-c"),))),
        evidence=evidence,
        pins=RunPins(
            technique_id="R02",
            technique_version="1",
            index_version_id="iv-1",
            embedding_key=None,
            chat_profile_id="local",
            chat_provider="ollama",
            chat_model="m",
            prompt_template="grounded-v1",
            min_score=None,
            k=5,
        ),
        usage=RunUsage(model_calls=1, input_tokens=1, output_tokens=1, estimated_cost_usd=None),
    )
    ids = source_version_ids(response)
    assert ids == ("dv-a", "dv-b", "dv-c", "dv-d")
    assert not set(ids) & {"doc-dv-a", "an-1", "view-1"}


def test_integer_and_float_temperature_give_the_same_key():
    as_int = inputs(generation=generation(temperature=0))
    as_float = inputs(generation=generation(temperature=0.0))
    assert exact_key(SCOPE, QUESTION, as_int) == exact_key(SCOPE, QUESTION, as_float)
    assert partition_key(SCOPE, as_int) == partition_key(SCOPE, as_float)
