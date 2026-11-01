"""Static checks on the Dockerfile and .dockerignore. Text parsing only; no docker needed."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _lines(text: str | None = None) -> list[str]:
    if text is None:
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    text = re.sub(r"\\\r?\n", " ", text)  # join line continuations
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def _credential_names(lines: list[str]) -> list[str]:
    found = []
    for ln in lines:
        if not re.match(r"(ENV|ARG)\s", ln, re.IGNORECASE):
            continue
        names = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=", ln) or [ln.split()[1]]
        found += [n for n in names if re.search(r"KEY|SECRET|TOKEN|PASSWORD", n.upper())]
    return found


def _from_lines() -> list[str]:
    return [ln for ln in _lines() if ln.upper().startswith("FROM ")]


def test_from_images_have_explicit_tags_and_no_latest():
    froms = _from_lines()
    assert froms
    for ln in froms:
        image = ln.split()[1]
        assert ":" in image, f"untagged image: {image}"
        assert not image.endswith(":latest"), f"latest tag: {image}"


def test_python_and_uv_tags_are_pinned():
    images = [ln.split()[1] for ln in _from_lines()]
    python_tags = [i.split(":", 1)[1] for i in images if i.startswith("python:")]
    uv_tags = [i.split(":", 1)[1] for i in images if i.startswith("ghcr.io/astral-sh/uv:")]
    assert python_tags and uv_tags
    assert all(re.fullmatch(r"3\.12\.\d+-slim-\w+", t) for t in python_tags)
    assert all(re.fullmatch(r"\d+\.\d+\.\d+", t) for t in uv_tags)


def test_final_stage_runs_as_non_root():
    lines = _lines()
    last_from = max(i for i, ln in enumerate(lines) if ln.upper().startswith("FROM "))
    users = [ln.split(None, 1)[1].strip() for ln in lines[last_from:] if ln.upper().startswith("USER ")]
    assert users, "final stage has no USER"
    assert users[-1].split(":")[0] not in {"root", "0"}


def test_no_credential_named_env_or_arg():
    assert _credential_names(_lines()) == []


def test_credential_check_sees_continuation_lines():
    text = "FROM python:3.12.16-slim-trixie\nENV A=1 \\\n    OPENAI_API_KEY=x\n"
    assert _credential_names(_lines(text)) == ["OPENAI_API_KEY"]


def test_dockerignore_excludes_env_and_git():
    entries = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in entries
    assert ".git" in entries
