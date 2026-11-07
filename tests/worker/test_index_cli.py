import asyncio
import json
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from rag_foundry_domain.errors import SettingsError
from rag_foundry_domain.runtime_settings import RuntimeSettings
from rag_foundry_domain.settings_store import RuntimeSettingsService
from rag_foundry_worker import index_cli
from rag_foundry_worker.index_cli import main


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os

    for key in list(os.environ):
        if key.upper().startswith("RAG_FOUNDRY_"):
            monkeypatch.delenv(key)


def test_prod_is_refused(monkeypatch, capsys):
    monkeypatch.setenv("RAG_FOUNDRY_ENVIRONMENT", "prod")
    monkeypatch.setenv("RAG_FOUNDRY_DATABASE_URL", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setattr(index_cli, "_run", lambda *_: pytest.fail("database path reached"))
    assert main(["inspect", "--index-version-id", "x"]) == 2
    assert "no auth context" in capsys.readouterr().err


def test_missing_database_url(monkeypatch, capsys):
    assert main(["inspect", "--index-version-id", "x"]) == 2
    assert "RAG_FOUNDRY_DATABASE_URL is not set" in capsys.readouterr().err


def test_unknown_setting_name(monkeypatch, capsys):
    monkeypatch.setenv("RAG_FOUNDRY_UNKNOWN_SETTING", "secret")
    assert main(["inspect", "--index-version-id", "x"]) == 2
    assert capsys.readouterr().err.strip() == "error: invalid RAG_FOUNDRY_* settings"


def test_invalid_runtime_settings_file_exits_two(monkeypatch, tmp_path, capsys):
    config = tmp_path / "runtime.json"
    config.write_text("{", encoding="utf-8")
    monkeypatch.setenv("RAG_FOUNDRY_RUNTIME_CONFIG_PATH", str(config))
    monkeypatch.setenv("RAG_FOUNDRY_DATABASE_URL", "postgresql+psycopg://u:p@localhost/db")
    assert main(["inspect", "--index-version-id", "x"]) == 2
    assert capsys.readouterr().err.strip() == "error: invalid runtime settings"


def test_unavailable_settings_store_exits_failed(monkeypatch, capsys):
    monkeypatch.setenv("RAG_FOUNDRY_DATABASE_URL", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setattr(index_cli, "_run", lambda *_: (_ for _ in ()).throw(SettingsError("store_unavailable")))
    assert main(["inspect", "--index-version-id", "x"]) == 1
    assert capsys.readouterr().err.strip() == "error: settings unavailable"


def test_inspection_line_omits_text_by_default_and_is_ascii():
    row = SimpleNamespace(
        ordinal=0,
        extraction_view_id="view",
        start=1,
        end=3,
        split_level=0,
        block_ids=("b00000",),
        text_sha256="abc",
        anchors=(),
        text="\u0645\u0631\u062d\u0628\u0627",
    )
    line = index_cli.inspection_line(row)
    assert "text" not in json.loads(line)
    assert line.isascii()
    shown = index_cli.inspection_line(row, True)
    assert '"text"' in shown and "\\u" in shown


def test_bad_answer_override_does_not_break_index_settings():
    class Repository:
        async def current(self, _scope):
            return (SimpleNamespace(level="global", key="answer.max_output_tokens", value=0),)

    async def resolve():
        service = RuntimeSettingsService(RuntimeSettings(), repo=Repository())
        indexing = await service.indexing(None)
        run = await service.run_settings(None)
        assert indexing.embed_batch_size == 32
        assert run.max_attempts == 2
        with pytest.raises(SettingsError, match="invalid_value"):
            await service.run_snapshot(None)

    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        runner.run(resolve())


def test_build_uses_run_settings_when_answer_override_is_invalid(monkeypatch):
    calls = []

    class Repository:
        async def current(self, _scope):
            return (SimpleNamespace(level="global", key="answer.max_output_tokens", value=0),)

    class Maker:
        uow = object()

    @asynccontextmanager
    async def open_db(_url):
        yield Maker()

    async def _done():
        return None

    class Service:
        def __init__(self, *_args, **kwargs):
            calls.append(("service", kwargs.get("batch_size")))

        async def build(self, *_args):
            calls.append(("build",))
            return SimpleNamespace(
                id="v",
                state=SimpleNamespace(value="ready"),
                chunk_count=1,
                unanchored_chunk_count=0,
                embedding_model="m",
                embedding_dim=1,
                embedding_key="k",
            )

    monkeypatch.setattr(index_cli, "open_database", open_db)
    monkeypatch.setattr(index_cli, "per_call_settings", lambda _factory: Repository())
    monkeypatch.setattr(index_cli, "LocalObjectStore", lambda *_: object())
    monkeypatch.setattr(index_cli, "AnchorService", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(index_cli, "IndexService", Service)
    monkeypatch.setattr(index_cli, "RecursiveChunker", lambda *_: object())
    monkeypatch.setattr(index_cli, "default_adapter_registry", lambda: SimpleNamespace(build=lambda *_: object()))
    monkeypatch.setattr(
        index_cli,
        "ResilientEmbeddingModel",
        lambda model, policy: (calls.append(("retry", policy.max_attempts)), model)[1],
    )

    settings = SimpleNamespace(
        database_url="db",
        object_store_root="root",
        runtime_config_path=None,
        provider_namespace="p",
        ollama_base_url="http://localhost",
    )
    args = SimpleNamespace(command="build", chunk_size=None, chunk_overlap=None, view_id=["view"])
    asyncio.run(index_cli._index(settings, args))
    assert ("build",) in calls
    assert ("retry", 2) in calls


def test_inspect_ignores_invalid_indexing_override(monkeypatch):
    class Repository:
        async def current(self, _scope):
            return (SimpleNamespace(level="global", key="indexing.c05_chunk_size", value=1),)

    class Maker:
        uow = object()

    @asynccontextmanager
    async def open_db(_url):
        yield Maker()

    async def _done():
        return None

    class Service:
        def __init__(self, *_args, **_kwargs):
            pass

        async def inspect_chunks(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(index_cli, "open_database", open_db)
    monkeypatch.setattr(index_cli, "per_call_settings", lambda _factory: Repository())
    monkeypatch.setattr(index_cli, "LocalObjectStore", lambda *_: object())
    monkeypatch.setattr(index_cli, "AnchorService", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(index_cli, "IndexService", Service)
    settings = SimpleNamespace(
        database_url="db", object_store_root="root", runtime_config_path=None, provider_namespace="p"
    )
    args = SimpleNamespace(command="inspect", index_version_id="v", offset=0, limit=20, show_text=False)
    assert asyncio.run(index_cli._index(settings, args)) == []
