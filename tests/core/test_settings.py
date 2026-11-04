import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_core.settings import Settings, load_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("RAG_FOUNDRY_"):
            monkeypatch.delenv(key)


def test_defaults():
    s = load_settings()
    assert s.environment == "dev"
    assert s.log_level == "INFO"
    assert s.api_host == "127.0.0.1"
    assert s.api_port == 8000
    assert s.worker_poll_seconds == 5.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAG_FOUNDRY_API_PORT", "9001")
    assert load_settings().api_port == 9001


def test_unknown_init_key_rejected():
    with pytest.raises(ValidationError):
        Settings(not_a_field=1)


def test_unknown_process_env_key_rejected(monkeypatch):
    monkeypatch.setenv("RAG_FOUNDRY_NOT_A_FIELD", "secret-value")
    with pytest.raises(ValueError) as exc:
        load_settings()
    assert "secret-value" not in str(exc.value)
    assert "RAG_FOUNDRY_NOT_A_FIELD" in str(exc.value)


def test_unknown_env_file_key_rejected(tmp_path):
    f = tmp_path / ".env"
    f.write_text("RAG_FOUNDRY_NOT_A_FIELD=1\n")
    with pytest.raises(ValidationError):
        load_settings(str(f))


def test_ollama_base_url_default():
    assert load_settings().ollama_base_url == "http://127.0.0.1:11434"


def test_ollama_base_url_env_override(monkeypatch):
    monkeypatch.setenv("RAG_FOUNDRY_OLLAMA_BASE_URL", "http://ollama:11434")
    assert load_settings().ollama_base_url == "http://ollama:11434"


def test_ollama_base_url_trailing_slash_stripped():
    assert Settings(ollama_base_url="https://host:11434/").ollama_base_url == "https://host:11434"


@pytest.mark.parametrize(
    "url",
    [
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
    ],
)
def test_ollama_base_url_rejected(url):
    with pytest.raises(ValidationError):
        Settings(ollama_base_url=url)


def test_ollama_base_url_user_info_not_echoed(monkeypatch):
    monkeypatch.setenv("RAG_FOUNDRY_OLLAMA_BASE_URL", "http://u:hunter2-secret@host:11434")
    with pytest.raises(ValidationError) as exc:
        load_settings()
    assert "hunter2-secret" not in str(exc.value)


@pytest.mark.parametrize("kwargs", [{"api_port": 0}, {"environment": "staging"}])
def test_invalid_values_rejected(kwargs):
    with pytest.raises(ValidationError):
        Settings(**kwargs)


def test_object_store_defaults():
    s = load_settings()
    assert s.object_store_root == Path("var/objects")
    assert s.max_upload_bytes == 100 * 1024 * 1024


def test_object_store_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_FOUNDRY_OBJECT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("RAG_FOUNDRY_MAX_UPLOAD_BYTES", "1024")
    s = load_settings()
    assert s.object_store_root == tmp_path
    assert s.max_upload_bytes == 1024


@pytest.mark.parametrize("value", [0, -1, 1024**3 + 1])
def test_max_upload_bytes_out_of_bounds(value):
    with pytest.raises(ValidationError):
        Settings(max_upload_bytes=value)


@pytest.mark.parametrize("value", [1, 1024**3])
def test_max_upload_bytes_bounds_accepted(value):
    assert Settings(max_upload_bytes=value).max_upload_bytes == value


def test_frozen():
    s = load_settings()
    with pytest.raises(ValidationError):
        s.api_port = 1


def test_database_url_defaults_to_none_and_is_secret():
    assert Settings().database_url is None
    s = Settings(database_url="postgresql+psycopg://u:pw-secret@host:5432/db")
    assert "pw-secret" not in repr(s) and "pw-secret" not in str(s)


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://u:pw-secret@h/db",
        "mysql://u:pw-secret@h/db",
        "postgresql+psycopg://u:pw-secret@h:99999999/db",
        "postgresql+psycopg://u:pw secret@h/db",
    ],
)
def test_database_url_rejects_bad_values_without_echo(url):
    with pytest.raises(ValidationError) as exc:
        Settings(database_url=url)
    assert "pw-secret" not in str(exc.value) and "pw secret" not in str(exc.value)
