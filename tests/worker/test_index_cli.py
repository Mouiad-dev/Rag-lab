import json
from types import SimpleNamespace

import pytest
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
