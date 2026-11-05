import uuid
from types import SimpleNamespace

import pytest
from rag_foundry_domain import IntakeError
from rag_foundry_domain.errors import ExtractionError
from rag_foundry_worker import extract_cli
from rag_foundry_worker.extract_cli import main

SECRET = "SECRET-PW-DO-NOT-PRINT"
DB_URL = f"postgresql+psycopg://rag_foundry:{SECRET}@127.0.0.1:1/rag_foundry"
ARGS = ["--document-id", "doc-x", "--version-id", "ver-y"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    import os

    for k in list(os.environ):
        if k.upper().startswith("RAG_FOUNDRY_"):
            monkeypatch.delenv(k)


def test_prod_is_refused_without_touching_the_database(monkeypatch, capsys):
    monkeypatch.setenv("RAG_FOUNDRY_ENVIRONMENT", "prod")
    monkeypatch.setenv("RAG_FOUNDRY_DATABASE_URL", DB_URL)

    def boom(*_args):  # the database must not be reached in prod
        raise AssertionError("extraction must not run in prod")

    monkeypatch.setattr(extract_cli, "_extract_one", boom)
    assert main(ARGS) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "no auth context" in captured.err
    assert SECRET not in captured.err


def test_missing_database_url_exits_2_with_one_line(capsys):
    assert main(ARGS) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "RAG_FOUNDRY_DATABASE_URL is not set" in captured.err
    assert "Traceback" not in captured.err


def test_invalid_database_url_does_not_echo_it(monkeypatch, capsys):
    monkeypatch.setenv("RAG_FOUNDRY_DATABASE_URL", f"mysql://rag_foundry:{SECRET}@127.0.0.1/rag")
    assert main(ARGS) == 2
    captured = capsys.readouterr()
    assert captured.err.count("\n") == 1
    assert SECRET not in captured.out + captured.err


def test_unknown_rag_foundry_env_name_exits_2(monkeypatch, capsys):
    monkeypatch.setenv("RAG_FOUNDRY_NOT_A_SETTING", "1")
    assert main(ARGS) == 2
    assert "RAG_FOUNDRY_" in capsys.readouterr().err


@pytest.mark.parametrize("missing", [["--document-id", "x"], ["--version-id", "y"], []])
def test_missing_arguments_exit_2(missing, capsys):
    with pytest.raises(SystemExit) as info:
        main(missing)
    assert info.value.code == 2
    assert "required" in capsys.readouterr().err


def test_domain_error_prints_code_only_without_traceback(monkeypatch, capsys):
    monkeypatch.setenv("RAG_FOUNDRY_DATABASE_URL", DB_URL)

    def fail(*_args):
        raise ExtractionError("no_text_layer", "no text layer at /private/path")

    monkeypatch.setattr(extract_cli, "_extract_one", fail)
    assert main(ARGS) == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "error: no_text_layer"
    assert "/private/path" not in captured.err


def test_not_found_is_reported_by_code(monkeypatch, capsys):
    monkeypatch.setenv("RAG_FOUNDRY_DATABASE_URL", DB_URL)

    def fail(*_args):
        raise IntakeError("not_found")

    monkeypatch.setattr(extract_cli, "_extract_one", fail)
    assert main(ARGS) == 1
    assert capsys.readouterr().err.strip() == "error: not_found"


def test_unexpected_error_prints_class_name_only(monkeypatch, capsys):
    monkeypatch.setenv("RAG_FOUNDRY_DATABASE_URL", DB_URL)

    def fail(*_args):
        raise ConnectionError(f"could not reach {DB_URL}")

    monkeypatch.setattr(extract_cli, "_extract_one", fail)
    assert main(ARGS) == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "error: unexpected ConnectionError"
    assert SECRET not in captured.err


def test_success_prints_summary_fields_only(monkeypatch, capsys):
    monkeypatch.setenv("RAG_FOUNDRY_DATABASE_URL", DB_URL)
    view = SimpleNamespace(
        id=uuid.UUID("12345678-1234-4234-8234-123456789abc"),
        char_count=42,
        block_count=2,
        page_count=None,
        warnings=("page_without_text:1",),
        canonical_text="SHOULD-NEVER-PRINT",
    )
    monkeypatch.setattr(
        extract_cli,
        "_extract_one",
        lambda *_args: SimpleNamespace(view=view, created=True),
    )
    assert main(ARGS) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "view_id=12345678-1234-4234-8234-123456789abc",
        "created=true",
        "char_count=42",
        "block_count=2",
        "page_count=none",
        "warnings=page_without_text:1",
    ]
    assert "SHOULD-NEVER-PRINT" not in out
