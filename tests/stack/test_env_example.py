"""Static checks on .env.example: known keys only, placeholder values only."""

import re
from pathlib import Path

from rag_foundry_core.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "RAG_FOUNDRY_"
ALLOWLIST = {"POSTGRES_USER", "POSTGRES_DB", "POSTGRES_PASSWORD", "API_HOST_PORT"}
SECRET_KEY_RE = re.compile(r"PASSWORD|SECRET|TOKEN|KEY")
SECRET_SHAPES = [re.compile(p) for p in (r"sk-", r"ghp_", r"AKIA", r"[A-Za-z0-9+/]{32,}={0,2}")]


def _entries() -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip("\"'")
    return out


def test_has_entries():
    assert _entries()


def test_app_keys_are_valid_settings_fields():
    for key, value in _entries().items():
        if key.startswith(PREFIX):
            field = key[len(PREFIX) :].lower()
            assert field in Settings.model_fields, key
            Settings(**{field: value})


def test_other_keys_are_allowlisted():
    others = {k for k in _entries() if not k.startswith(PREFIX)}
    assert others <= ALLOWLIST, others - ALLOWLIST


def test_secret_like_keys_hold_placeholders():
    for key, value in _entries().items():
        if SECRET_KEY_RE.search(key.upper()):
            assert value == "" or value.startswith("change-me"), key


def test_no_secret_shaped_values():
    for key, value in _entries().items():
        for pat in SECRET_SHAPES:
            assert not pat.search(value), key
