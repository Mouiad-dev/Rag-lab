import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_core.settings import Settings
from rag_foundry_providers._urls import is_loopback, validate_base_url
from rag_foundry_providers.config import (
    ProvidersConfig,
    ProvidersConfigError,
    load_providers_config,
    parse_providers_config,
)
from rag_foundry_providers.errors import ProviderError

EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "providers.example.json"
SENTINEL = "ZZ-sentinel-9f3a"


def base() -> dict:
    return copy.deepcopy(json.loads(EXAMPLE.read_text(encoding="utf-8")))


def locations(exc: ProvidersConfigError) -> list[str]:
    return [loc for loc, _ in exc.errors]


def bad(data: dict) -> ProvidersConfigError:
    with pytest.raises(ProvidersConfigError) as info:
        parse_providers_config(data)
    return info.value


def test_example_loads():
    cfg = load_providers_config(EXAMPLE)
    assert isinstance(cfg, ProvidersConfig)
    assert cfg.stages["embedder"] == "local-embed"
    assert {e.adapter for e in cfg.endpoints} == {"fake", "ollama", "anthropic", "openai_compatible"}
    haiku = next(p for p in cfg.profiles if p.id == "anthropic-haiku")
    assert str(haiku.cost_per_mtok_in) == "0.10"


def test_error_is_provider_error():
    assert issubclass(ProvidersConfigError, ProviderError)


def test_config_is_frozen():
    cfg = load_providers_config(EXAMPLE)
    with pytest.raises(ValidationError):
        cfg.note = "x"


# unknown keys at every level
def test_unknown_key_root():
    d = base()
    d["extra"] = 1
    assert "<unknown key>" in locations(bad(d))


def test_unknown_key_endpoint():
    d = base()
    d["endpoints"][0]["api_key"] = "x"
    assert "endpoints.0.<unknown key>" in locations(bad(d))


def test_unknown_key_profile():
    d = base()
    d["profiles"][0]["api_key"] = "x"
    assert "profiles.0.<unknown key>" in locations(bad(d))


def test_unknown_key_capabilities():
    d = base()
    d["profiles"][0]["capabilities"]["bogus"] = True
    assert "profiles.0.capabilities.<unknown key>" in locations(bad(d))


def test_unknown_stage_rejected():
    d = base()
    d["stages"]["summarizer"] = "local-chat"
    assert any(loc.startswith("stages.<unknown key>") for loc in locations(bad(d)))


@pytest.mark.parametrize(
    "stage", ["generator", "embedder", "reranker", "query_transform", "judge", "chunk_llm", "scout"]
)
def test_every_adr_stage_is_accepted(stage):
    d = base()
    d["stages"] = {stage: "local-chat"}
    parse_providers_config(d)


def test_schema_version_must_be_1():
    d = base()
    d["schema_version"] = 2
    assert "schema_version" in locations(bad(d))


# endpoint rules
def test_fake_takes_no_url_or_key():
    d = base()
    d["endpoints"][0]["base_url"] = "http://127.0.0.1:1"
    assert "endpoints.0" in locations(bad(d))
    d = base()
    d["endpoints"][0]["api_key_env"] = "SOME_KEY"
    assert "endpoints.0" in locations(bad(d))


def test_ollama_needs_base_url():
    d = base()
    del d["endpoints"][1]["base_url"]
    assert "endpoints.1" in locations(bad(d))


def test_openai_compatible_needs_base_url():
    d = base()
    d["endpoints"].append({"id": "x", "adapter": "openai_compatible", "local": True})
    assert "endpoints.6" in locations(bad(d))


def test_anthropic_needs_key_env():
    d = base()
    del d["endpoints"][2]["api_key_env"]
    assert "endpoints.2" in locations(bad(d))


@pytest.mark.parametrize("value", ["sk-ant-abc123", "lowercase", "1ABC", "HAS SPACE", "A" * 65, ""])
def test_key_env_must_be_a_name(value):
    d = base()
    d["endpoints"][2]["api_key_env"] = value
    assert "endpoints.2.api_key_env" in locations(bad(d))


def test_key_over_http_non_loopback_rejected():
    d = base()
    d["endpoints"][2]["base_url"] = "http://api.example.com"
    assert "endpoints.2" in locations(bad(d))


def test_key_over_http_loopback_and_https_allowed():
    d = base()
    d["endpoints"][2]["base_url"] = "http://127.0.0.1:9"
    parse_providers_config(d)
    d["endpoints"][2]["base_url"] = "https://api.example.com"
    parse_providers_config(d)


def test_bad_base_url_in_config():
    d = base()
    d["endpoints"][1]["base_url"] = "http://u:p@host"
    assert "endpoints.1.base_url" in locations(bad(d))


def test_local_must_be_a_bool():
    d = base()
    d["endpoints"][0]["local"] = "yes"
    assert "endpoints.0.local" in locations(bad(d))


# cross-checks
def test_duplicate_endpoint_and_profile_ids():
    d = base()
    d["endpoints"].append(dict(d["endpoints"][1]))
    d["profiles"].append(dict(d["profiles"][1]))
    locs = locations(bad(d))
    assert "endpoints.6.id" in locs
    assert "profiles.5.id" in locs


def test_profile_provider_must_exist():
    d = base()
    d["profiles"][1]["provider"] = "nowhere"
    assert "profiles.1.provider" in locations(bad(d))


def test_local_flag_must_match_endpoint():
    d = base()
    d["profiles"][1]["capabilities"]["local"] = False
    assert "profiles.1.capabilities.local" in locations(bad(d))


def test_stage_target_must_exist():
    d = base()
    d["stages"]["judge"] = "ghost"
    assert "stages.judge" in locations(bad(d))


@pytest.mark.parametrize("flag", ["tool_calling", "structured_output", "vision"])
@pytest.mark.parametrize("index", [0, 1, 3])
def test_unimplemented_capability_true_rejected(flag, index):
    d = base()
    d["profiles"][index]["capabilities"][flag] = True
    assert f"profiles.{index}.capabilities.{flag}" in locations(bad(d))


def test_unimplemented_capability_false_allowed():
    d = base()
    d["profiles"][1]["capabilities"]["tool_calling"] = False
    parse_providers_config(d)


def test_cross_check_problems_reported_together():
    d = base()
    d["profiles"][1]["provider"] = "nowhere"
    d["stages"]["judge"] = "ghost"
    d["profiles"][3]["capabilities"]["vision"] = True
    assert len(bad(d).errors) == 3


def test_embedding_profile_rules_reused():
    d = base()
    del d["profiles"][2]["capabilities"]["embedding_dim"]
    assert any(loc.startswith("profiles.2") for loc in locations(bad(d)))


# rates as strings
@pytest.mark.parametrize(
    "field", ["cost_per_mtok_in", "cost_per_mtok_out", "cost_per_mtok_cache_read", "cost_per_mtok_cache_write"]
)
@pytest.mark.parametrize("number", [0.1, 1])
def test_numeric_rate_rejected(field, number):
    d = base()
    d["profiles"][3][field] = number
    assert f"profiles.3.{field}" in locations(bad(d))


def test_negative_rate_rejected():
    d = base()
    d["profiles"][3]["cost_per_mtok_in"] = "-1"
    assert "profiles.3.cost_per_mtok_in" in locations(bad(d))


# loader
def write(tmp_path, content: bytes) -> Path:
    p = tmp_path / "c.json"
    p.write_bytes(content)
    return p


def test_loader_rejects_duplicate_json_keys(tmp_path):
    with pytest.raises(ProvidersConfigError) as info:
        load_providers_config(write(tmp_path, b'{"schema_version": 1, "schema_version": 1}'))
    assert "duplicate" in str(info.value)


def test_loader_rejects_nested_duplicate_json_keys(tmp_path):
    text = json.dumps(base()).replace('"local": true}', '"local": true, "local": true}', 1)
    with pytest.raises(ProvidersConfigError):
        load_providers_config(write(tmp_path, text.encode()))


def test_loader_rejects_big_file(tmp_path):
    with pytest.raises(ProvidersConfigError) as info:
        load_providers_config(write(tmp_path, b" " * (1024 * 1024 + 1)))
    assert "1 MiB" in str(info.value)


def test_loader_accepts_file_at_limit(tmp_path):
    raw = json.dumps(base()).encode()
    padded = raw + b" " * (1024 * 1024 - len(raw))
    load_providers_config(write(tmp_path, padded))


@pytest.mark.parametrize("content", [b"[]", b"1", b'"x"', b"null"])
def test_loader_rejects_non_object_root(tmp_path, content):
    with pytest.raises(ProvidersConfigError) as info:
        load_providers_config(write(tmp_path, content))
    assert "object" in str(info.value)


def test_loader_rejects_bad_json_and_bad_utf8_and_missing(tmp_path):
    for content in (b"{not json", b"\xff\xfe\x00"):
        with pytest.raises(ProvidersConfigError):
            load_providers_config(write(tmp_path, content))
    with pytest.raises(ProvidersConfigError):
        load_providers_config(tmp_path / "missing.json")


# secrets never echoed
def _string_paths(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _string_paths(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _string_paths(v, (*path, i))
    elif isinstance(node, str):
        yield path


def _set(d, path, value):
    for part in path[:-1]:
        d = d[part]
    d[path[-1]] = value


def test_sentinel_never_echoed_in_any_string_field():
    paths = list(_string_paths(base()))
    assert len(paths) > 15
    failures = 0
    for path in paths:
        for variant in (SENTINEL, f"http://u:{SENTINEL}@host", f"{SENTINEL} {SENTINEL}\n"):
            d = base()
            _set(d, path, variant)
            try:
                parse_providers_config(d)
            except ProvidersConfigError as exc:
                failures += 1
                assert SENTINEL not in str(exc)
                assert SENTINEL not in repr(exc)
                assert SENTINEL not in repr(exc.errors)
                assert exc.__cause__ is None
                assert exc.__context__ is None
    assert failures > 0


def test_sentinel_in_numeric_rate_and_loader_errors(tmp_path):
    d = base()
    d["profiles"][3]["cost_per_mtok_in"] = SENTINEL
    exc = bad(d)
    assert SENTINEL not in str(exc)
    assert exc.__cause__ is None and exc.__context__ is None
    p = write(tmp_path, f'{{"x": "{SENTINEL}"'.encode())
    with pytest.raises(ProvidersConfigError) as info:
        load_providers_config(p)
    assert SENTINEL not in str(info.value)
    assert info.value.__cause__ is None and info.value.__context__ is None


def test_sentinel_in_key_env_value_not_echoed():
    d = base()
    d["endpoints"][2]["api_key_env"] = "sk-ant-" + SENTINEL
    exc = bad(d)
    assert SENTINEL not in str(exc)


# URL helpers
BAD_URLS = [
    "ftp://x",
    "127.0.0.1:11434",
    "http://u:p@host:11434",
    "http://host/?a=1",
    "http://host/#frag",
    "http://",
    " http://host:11434",
    "http://host:11434\n",
    "http://ho\tst:11434",
    "http://host:99999",
    "http://host:abc",
    "",
    "http://@host",
    "http://host\x00",
    "http://host:11434/?",
    "http://host:11434/#",
    "//host",
    "https:///path",
]


@pytest.mark.parametrize("url", BAD_URLS)
def test_url_parity_with_settings(url):
    settings_rejects = False
    try:
        Settings(ollama_base_url=url)
    except ValidationError:
        settings_rejects = True
    if settings_rejects:
        with pytest.raises(ValueError):
            validate_base_url(url, "base_url")
    # Settings is the reference: where it rejects, we must too. Where it accepts, we must too.
    else:
        assert validate_base_url(url, "base_url") == Settings(ollama_base_url=url).ollama_base_url


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "https://host:11434/", "http://[::1]:80", "http://h/v1/"])
def test_url_good_parity(url):
    assert validate_base_url(url, "base_url") == Settings(ollama_base_url=url).ollama_base_url


def test_url_error_names_field_not_value():
    with pytest.raises(ValueError) as info:
        validate_base_url(f"http://u:{SENTINEL}@host", "endpoints.0.base_url")
    assert "endpoints.0.base_url" in str(info.value)
    assert SENTINEL not in str(info.value)


def test_trailing_slash_stripped_once():
    assert validate_base_url("http://host//", "f") == "http://host/"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:1", True),
        ("http://127.5.6.7", True),
        ("http://[::1]:1", True),
        ("http://localhost:1", True),
        ("http://LOCALHOST", True),
        ("http://10.0.0.1", False),
        ("http://example.com", False),
        ("http://127.0.0.1.example.com", False),
        ("http://0.0.0.0", False),
    ],
)
def test_is_loopback(url, expected):
    assert is_loopback(url) is expected


def test_unknown_key_names_are_redacted_everywhere():
    secret = "sk-ant-SECRET"
    for path in ((), ("endpoints", 0), ("profiles", 0), ("profiles", 0, "capabilities"), ("stages",)):
        d = base()
        _set(d, (*path, secret), 1) if path else d.__setitem__(secret, 1)
        exc = bad(d)
        assert secret not in str(exc)
        assert secret not in repr(exc.errors)
        assert any("<unknown key>" in loc for loc in locations(exc))


@pytest.mark.parametrize(
    "url", ["HTTP://evil.com/v1", "Http://evil.com", "http://localhost.evil.com", "http://127.0.0.1.nip.io"]
)
def test_key_over_non_https_non_loopback_rejected(url):
    d = base()
    d["endpoints"][2]["base_url"] = url
    assert "endpoints.2" in locations(bad(d))


@pytest.mark.parametrize("url", ["http://[::1]", "HTTP://localhost:1234/v1", "HTTPS://api.example.com"])
def test_key_with_loopback_or_https_allowed(url):
    d = base()
    d["endpoints"][2]["base_url"] = url
    parse_providers_config(d)


@pytest.mark.parametrize(
    "value", [" 1", "1 ", "1e3", "-0", "-1", "+1", ".5", "1.", "0x10", "1_0", "", "NaN", "Infinity", "١"]
)
def test_rate_format_rejected(value):
    d = base()
    d["profiles"][3]["cost_per_mtok_out"] = value
    assert "profiles.3.cost_per_mtok_out" in locations(bad(d))


@pytest.mark.parametrize("value", ["0", "0.10", "10", "2.00"])
def test_rate_format_accepted(value):
    d = base()
    d["profiles"][3]["cost_per_mtok_out"] = value
    parse_providers_config(d)


def test_openai_compatible_endpoints_in_example():
    cfg = load_providers_config(EXAMPLE)
    by_id = {e.id: e for e in cfg.endpoints}
    assert by_id["vllm"].base_url == "http://127.0.0.1:8001/v1"
    assert by_id["lmstudio"].base_url == "http://localhost:1234/v1"
    assert by_id["moonshot"].base_url == "https://api.moonshot.ai/v1"
    assert by_id["moonshot"].api_key_env == "MOONSHOT_API_KEY"
    kimi = next(p for p in cfg.profiles if p.id == "kimi-k3-1-pro")
    assert kimi.capabilities.context_window == 1048576
    assert set(cfg.stages.values()) <= {"local-chat", "local-embed"}


# openai_compatible endpoint settings (P2.4-05 review)
MOONSHOT = 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_tokens_field", "max_new_tokens"),
        ("max_tokens_field", SENTINEL),
        ("supports_temperature", "false"),
        ("supports_temperature", 0),
        ("default_max_tokens", 0),
        ("default_max_tokens", -1),
        ("default_max_tokens", "4096"),
        ("default_max_tokens", 4096.0),
    ],
)
def test_openai_compatible_setting_bad_value_rejected(field, value):
    d = base()
    d["endpoints"][MOONSHOT][field] = value
    exc = bad(d)
    assert f"endpoints.{MOONSHOT}.{field}" in locations(exc)
    assert SENTINEL not in str(exc)
    assert SENTINEL not in repr(exc.errors)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize(
    ("field", "value"),
    [("max_tokens_field", "max_tokens"), ("supports_temperature", False), ("default_max_tokens", 4096)],
)
def test_openai_compatible_settings_rejected_on_other_adapters(index, field, value):
    d = base()
    d["endpoints"][index][field] = value
    assert any(loc.startswith(f"endpoints.{index}") for loc in locations(bad(d)))
