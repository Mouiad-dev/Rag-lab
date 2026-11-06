import json

import pytest
from rag_foundry_contracts.runs import RunResponse
from rag_foundry_worker import ask_cli
from rag_foundry_worker.ask_cli import format_response, main


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os

    for key in list(os.environ):
        if key.upper().startswith("RAG_FOUNDRY_"):
            monkeypatch.delenv(key)


def test_prod_is_refused(monkeypatch, capsys):
    monkeypatch.setenv("RAG_FOUNDRY_ENVIRONMENT", "prod")
    monkeypatch.setenv("RAG_FOUNDRY_DATABASE_URL", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setattr(ask_cli, "_run", lambda *_: pytest.fail("database path reached"))
    assert main(["--question", "Q"]) == 2
    assert "no auth context" in capsys.readouterr().err


def test_missing_database_url(capsys):
    assert main(["--question", "Q"]) == 2
    assert "RAG_FOUNDRY_DATABASE_URL is not set" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--question", "a", "--question-file", "b"], []])
def test_both_or_neither_question_option_exits_two(args, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "required" in error or "not allowed" in error


def test_invalid_run_request_exits_two(monkeypatch, capsys):
    monkeypatch.setenv("RAG_FOUNDRY_DATABASE_URL", "postgresql+psycopg://u:p@localhost/db")
    assert main(["--question", "", "--k", "0"]) == 2
    assert capsys.readouterr().err.strip() == "error: invalid run request"


def test_unknown_setting_name(monkeypatch, capsys):
    monkeypatch.setenv("RAG_FOUNDRY_NOT_A_SETTING", "secret")
    assert main(["--question", "Q"]) == 2
    assert capsys.readouterr().err.strip() == "error: invalid RAG_FOUNDRY_* settings"


def test_format_response_is_ascii_and_round_trips():
    response = RunResponse.model_validate(
        {
            "run_id": "12345678-1234-4234-8234-123456789abc",
            "status": "abstained",
            "abstention_reason": "no_evidence",
            "answer": None,
            "citations": [],
            "evidence": [],
            "pins": {
                "technique_id": "R02",
                "technique_version": "1.0",
                "index_version_id": None,
                "embedding_key": None,
                "chat_profile_id": "p",
                "chat_provider": "ollama",
                "chat_model": "m",
                "prompt_template": "grounded-answer-v2",
                "min_score": None,
                "k": None,
            },
            "usage": {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": "0"},
            "warnings": [],
        }
    )
    rendered = format_response(response)
    assert rendered.isascii()
    assert RunResponse.model_validate(json.loads(rendered)) == response
