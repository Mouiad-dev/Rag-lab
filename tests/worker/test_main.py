import logging

import pytest
from pydantic import ValidationError
from rag_foundry_worker.main import main


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    import os

    for k in list(os.environ):
        if k.upper().startswith("RAG_FOUNDRY_"):
            monkeypatch.delenv(k)


def test_once_returns_zero_and_logs(caplog):
    with caplog.at_level(logging.INFO, logger="rag_foundry_worker"):
        assert main(["--once"]) == 0
    assert "worker started env=dev" in caplog.text


def test_invalid_settings_propagate(monkeypatch):
    monkeypatch.setenv("RAG_FOUNDRY_API_PORT", "not-a-port")
    with pytest.raises(ValidationError):
        main(["--once"])
