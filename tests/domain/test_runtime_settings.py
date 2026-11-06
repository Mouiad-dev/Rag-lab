from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_chunking.recursive import RecursiveConfig
from rag_foundry_domain.errors import SettingsError
from rag_foundry_domain.runtime_settings import (
    ALLOWED_KEYS,
    AnswerSettings,
    LayerOverride,
    RunSettings,
    RuntimeSettings,
    load_runtime_settings_file,
    resolve_indexing,
    resolve_run,
)

ROOT = Path(__file__).resolve().parents[2]


def test_defaults_match_inventory_literals():
    settings = RuntimeSettings()
    assert (settings.answer.min_scores, settings.answer.max_output_tokens, settings.answer.temperature) == (
        {"ollama/bge-m3/1024": 0.45},
        512,
        0.0,
    )
    assert (settings.answer.max_evidence_chars, settings.run.max_attempts, settings.run.attempt_timeout_s) == (
        12000,
        2,
        120.0,
    )
    assert (settings.run.retry_timeouts_for_local, settings.run.deadline_s, settings.run.max_concurrent) == (
        False,
        300.0,
        1,
    )
    assert settings.indexing.embed_batch_size == 32
    assert (settings.indexing.c05_chunk_size, settings.indexing.c05_chunk_overlap) == (800, 100)


def test_chunk_defaults_match_recursive_config():
    assert (RuntimeSettings().indexing.c05_chunk_size, RuntimeSettings().indexing.c05_chunk_overlap) == (
        RecursiveConfig().chunk_size,
        RecursiveConfig().chunk_overlap,
    )


def test_layer_order_project_over_tenant_global_file_default():
    base = RuntimeSettings(answer=AnswerSettings(temperature=0.1))
    result = resolve_run(
        base,
        [
            LayerOverride("project", "answer.temperature", 0.4),
            LayerOverride("global", "answer.temperature", 0.2),
            LayerOverride("tenant", "answer.temperature", 0.3),
        ],
        file_keys=frozenset({"answer.temperature"}),
    )
    assert result.answer.temperature == 0.4
    assert result.sources["answer.temperature"] == "project"
    assert resolve_run(base, [], file_keys=frozenset({"answer.temperature"})).sources["answer.temperature"] == "file"


def test_indexing_layer_order_project_over_tenant_global():
    result = resolve_indexing(
        RuntimeSettings(),
        [
            LayerOverride("global", "indexing.c05_chunk_size", 900),
            LayerOverride("tenant", "indexing.c05_chunk_size", 1000),
            LayerOverride("project", "indexing.c05_chunk_size", 1100),
        ],
    )
    assert result.c05_chunk_size == 1100


def test_reset_falls_back_to_base_value():
    result = resolve_run(
        RuntimeSettings(),
        [LayerOverride("global", "answer.temperature", 0.8), LayerOverride("project", "answer.temperature", None)],
    )
    assert result.answer.temperature == 0.8


def test_bad_indexing_pair_does_not_break_run_resolution():
    overrides = [LayerOverride("global", "indexing.c05_chunk_overlap", 800)]
    resolve_run(RuntimeSettings(), overrides)
    with pytest.raises(SettingsError) as exc:
        resolve_indexing(RuntimeSettings(), overrides)
    assert exc.value.code == "invalid_value"


def test_bad_answer_does_not_break_indexing_resolution():
    overrides = [LayerOverride("global", "answer.temperature", 3.0)]
    resolve_indexing(RuntimeSettings(), overrides)
    with pytest.raises(SettingsError) as exc:
        resolve_run(RuntimeSettings(), overrides)
    assert exc.value.code == "invalid_value"


def test_invalid_value_error_is_value_free_and_has_no_cause_or_context():
    with pytest.raises(SettingsError) as exc:
        resolve_run(RuntimeSettings(), [LayerOverride("global", "answer.temperature", 1234.98765)])
    assert exc.value.code == "invalid_value"
    assert "1234.98765" not in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None


def test_unknown_scope_and_value_errors_are_value_free():
    with pytest.raises(SettingsError) as unknown:
        resolve_run(RuntimeSettings(), [LayerOverride("global", "answer.secret", "SENTINEL-984321")])
    assert unknown.value.code == "unknown_key" and "SENTINEL-984321" not in str(unknown.value)
    with pytest.raises(SettingsError) as scope:
        resolve_run(RuntimeSettings(), [LayerOverride("project", "run.max_attempts", "SCOPE-SENTINEL")])
    assert scope.value.code == "scope_not_allowed"
    assert "SCOPE-SENTINEL" not in str(scope.value)


def test_min_scores_override_replaces_the_whole_value():
    base = RuntimeSettings(answer=AnswerSettings(min_scores={"a/b/1": 0.2}))
    result = resolve_run(base, [LayerOverride("global", "answer.min_scores", {"c/d/2": 0.3})])
    assert result.answer.min_scores == {"c/d/2": 0.3}


def test_schema_version_rejects_bool():
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"schema_version": True})


def test_strict_integer_and_nan_are_rejected():
    with pytest.raises(ValidationError):
        RunSettings(max_attempts="5")
    with pytest.raises(ValidationError):
        AnswerSettings(temperature=float("nan"))


def test_snapshot_hash_stable_across_dict_order_and_changes_with_temperature():
    first = RuntimeSettings(answer=AnswerSettings(min_scores={"a/b/1": 0.2, "c/d/2": 0.3}))
    second = RuntimeSettings(answer=AnswerSettings(min_scores={"c/d/2": 0.3, "a/b/1": 0.2}))
    assert resolve_run(first, []).sha256() == resolve_run(second, []).sha256()
    changed = resolve_run(first, [LayerOverride("global", "answer.temperature", 0.5)])
    assert changed.sha256() != resolve_run(first, []).sha256()


def test_example_file_loads_to_defaults():
    settings, keys = load_runtime_settings_file(ROOT / "config" / "runtime.example.json")
    assert settings == RuntimeSettings()
    assert keys == ALLOWED_KEYS


def test_file_loader_rejects_duplicate_key(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text('{"answer":{"temperature":0.1,"temperature":0.2}}')
    with pytest.raises(SettingsError) as exc:
        load_runtime_settings_file(path)
    assert exc.value.code == "invalid_file"


def test_file_loader_rejects_oversize_file(tmp_path):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(SettingsError) as exc:
        load_runtime_settings_file(path)
    assert exc.value.code == "invalid_file"


def test_file_loader_rejects_non_object(tmp_path):
    path = tmp_path / "array.json"
    path.write_text("[]")
    with pytest.raises(SettingsError) as exc:
        load_runtime_settings_file(path)
    assert exc.value.code == "invalid_file"


def test_file_loader_rejects_deeply_nested_invalid_json(tmp_path):
    path = tmp_path / "recursive.json"
    path.write_text("[" * 40000)
    with pytest.raises(SettingsError) as exc:
        load_runtime_settings_file(path)
    assert exc.value.code == "invalid_file"


def test_file_loader_errors_are_value_free_and_have_no_cause_or_context(tmp_path):
    path = tmp_path / "bad-value.json"
    path.write_text('{"answer":{"temperature":1234.98765}}')
    with pytest.raises(SettingsError) as exc:
        load_runtime_settings_file(path)
    assert exc.value.code == "invalid_file"
    assert "1234.98765" not in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None


def test_file_loader_rejects_unknown_key(tmp_path):
    path = tmp_path / "unknown.json"
    path.write_text('{"answer":{"mystery":1}}')
    with pytest.raises(SettingsError) as exc:
        load_runtime_settings_file(path)
    assert exc.value.code == "invalid_file"
