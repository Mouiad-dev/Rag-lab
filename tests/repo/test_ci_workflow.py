"""Static text checks on .github/workflows/ci.yml (stdlib only, no YAML dependency)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
USES = re.compile(r"^\s*(-\s+)?uses:\s+[\w.-]+/[\w.-]+@[0-9a-f]{40}\s+#\s+v\d+\.\d+\.\d+\s*$")


def test_every_action_is_pinned_to_a_sha_with_version_comment():
    lines = [line for line in TEXT.splitlines() if re.match(r"^\s*(-\s+)?uses:", line)]
    assert lines
    for line in lines:
        assert USES.match(line), line


def test_least_privilege():
    assert re.search(r"^permissions:\s*\n\s+contents: read\s*$", TEXT, re.MULTILINE)
    for banned in (": write", "secrets.", "pull_request_target"):
        assert banned not in TEXT, banned


def test_expected_commands_and_runners():
    for needle in ("uv sync --locked", "scripts/tasks.py check", "ubuntu-latest", "windows-latest"):
        assert needle in TEXT, needle


def test_concurrency_cancels_superseded_runs():
    assert "cancel-in-progress: true" in TEXT
