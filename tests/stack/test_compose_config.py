"""Checks on the resolved compose model. Uses `docker compose config` only (no pulls)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rag_foundry_core.settings import Settings

ROOT = Path(__file__).resolve().parents[2]


def _docker_compose_ok() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(["docker", "compose", "version"], capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not _docker_compose_ok(), reason="docker compose not available")


def _config(*profile: str) -> dict:
    cmd = ["docker", "compose", "--env-file", ".env.example"]
    for p in profile:
        cmd += ["--profile", p]
    cmd += ["config", "--format", "json"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
    return json.loads(r.stdout)


@pytest.fixture(scope="module")
def full() -> dict:
    return _config("ollama")


def test_ports_bind_loopback_only(full):
    for name, svc in full["services"].items():
        for port in svc.get("ports", []):
            assert port.get("host_ip") == "127.0.0.1", f"{name}: {port}"


def test_no_bind_mounts_or_docker_socket(full):
    for name, svc in full["services"].items():
        for vol in svc.get("volumes", []):
            assert vol.get("type") != "bind", f"{name}: {vol}"
            assert "docker.sock" not in str(vol.get("source", "")), f"{name}: {vol}"


def test_every_service_has_healthcheck(full):
    for name, svc in full["services"].items():
        assert svc.get("healthcheck"), name


def test_images_have_explicit_non_latest_tags(full):
    for name, svc in full["services"].items():
        image = svc["image"]
        tag = image.rsplit("/", 1)[-1].partition(":")[2]
        assert tag and tag != "latest", f"{name}: {image}"


def test_ollama_only_with_profile(full):
    assert "ollama" not in _config()["services"]
    assert "ollama" in full["services"]
    for name in ("api", "worker"):
        assert "ollama" not in full["services"][name].get("depends_on", {})


def test_app_environment_keys_are_settings_fields(full):
    for name in ("api", "worker"):
        for key in full["services"][name]["environment"]:
            assert key.startswith("RAG_FOUNDRY_"), f"{name}: {key}"
            assert key[len("RAG_FOUNDRY_") :].lower() in Settings.model_fields, f"{name}: {key}"
