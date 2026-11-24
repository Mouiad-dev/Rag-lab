import logging
import os
import time

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


def test_once_with_heartbeat_file_creates_it(tmp_path):
    hb = tmp_path / "hb"
    assert main(["--once", "--heartbeat-file", str(hb)]) == 0
    assert hb.exists()


def test_check_heartbeat_fresh(tmp_path, caplog):
    hb = tmp_path / "hb"
    hb.touch()
    with caplog.at_level(logging.INFO, logger="rag_foundry_worker"):
        assert main(["--check-heartbeat", str(hb)]) == 0
    assert "worker started" not in caplog.text


def test_check_heartbeat_missing(tmp_path):
    assert main(["--check-heartbeat", str(tmp_path / "nope")]) == 1


def test_check_heartbeat_stale(tmp_path):
    hb = tmp_path / "hb"
    hb.touch()
    old = time.time() - 3600
    os.utime(hb, (old, old))
    assert main(["--check-heartbeat", str(hb)]) == 1


def test_check_heartbeat_exclusive_with_once(tmp_path):
    with pytest.raises(SystemExit):
        main(["--check-heartbeat", str(tmp_path / "hb"), "--once"])


def test_check_heartbeat_exclusive_with_heartbeat_file(tmp_path):
    with pytest.raises(SystemExit):
        main(["--check-heartbeat", str(tmp_path / "hb"), "--heartbeat-file", str(tmp_path / "x")])


def test_check_heartbeat_future_mtime_is_stale(tmp_path):
    hb = tmp_path / "hb"
    hb.touch()
    future = time.time() + 3600
    os.utime(hb, (future, future))
    assert main(["--check-heartbeat", str(hb)]) == 1


def test_loop_writes_heartbeat(tmp_path, monkeypatch):
    hb = tmp_path / "hb"
    hb.touch()
    old = time.time() - 3600
    os.utime(hb, (old, old))
    calls = []

    def fake_sleep(_seconds):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", fake_sleep)
    assert main(["--heartbeat-file", str(hb)]) == 0
    assert len(calls) == 2
    assert time.time() - hb.stat().st_mtime < 60


def test_invalid_settings_propagate(monkeypatch):
    monkeypatch.setenv("RAG_FOUNDRY_API_PORT", "not-a-port")
    with pytest.raises(ValidationError):
        main(["--once"])
