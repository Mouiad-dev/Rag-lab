#!/usr/bin/env python3
"""Narrow Claude Code development helpers. Not a security sandbox.

Input: documented hook JSON on stdin. Output: hook JSON only on stdout.
No network, subprocesses, dependency installation, telemetry, or source rewriting.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tokenize


ROOT = Path(__file__).resolve().parents[2]
MAX_INPUT = 8 * 1024 * 1024
MAX_FILE = 2 * 1024 * 1024
FILE_TOOLS = {"Read", "Write", "Edit", "NotebookEdit"}
WRITE_TOOLS = {"Write", "Edit", "NotebookEdit"}


def emit(event: str, **fields: str) -> None:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, **fields}}))


def inside_root(path: Path) -> bool:
    return path.is_relative_to(ROOT)


def protected(path: Path) -> bool:
    name = path.name.casefold()
    if any(part.casefold() in {".ssh", ".aws", "secrets"} for part in path.parts):
        return True
    if name in {".env.example", ".env.template"}:
        return False
    if name == ".env" or name.startswith(".env."):
        return True
    if name in {"id_rsa", "id_ed25519", "credentials.json", "service-account.json"}:
        return True
    if path.suffix.casefold() in {".pem", ".key", ".p12", ".pfx"}:
        return True
    return False


def target_paths(payload: dict) -> tuple[Path, Path] | None:
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        raise ValueError("missing tool_input object")
    raw = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("missing target file path")
    path = Path(raw)
    if not path.is_absolute():
        cwd = payload.get("cwd")
        if not isinstance(cwd, str) or not Path(cwd).is_absolute():
            raise ValueError("relative target requires an absolute cwd")
        path = Path(cwd) / path
    # Inspect both the spelled path and its resolved target, including symlinks.
    return path, path.resolve()


def orient(payload: dict) -> None:
    emit(
        "SessionStart",
        additionalContext=(
            "RAG Foundry: read docs/PROGRESS.md and docs/architecture/SUMMARY.md "
            "(not the full PROJECT_ARCH.md). Planning stage, not a running app. "
            "Preserve all 72 catalogue entries. Work one task spec at a time "
            "(planner -> coder -> reviewer); explain in the user's language. "
            "These development hooks are not the application's sandbox."
        ),
    )


def guard(payload: dict) -> None:
    tool = payload.get("tool_name")
    if tool not in FILE_TOOLS:
        return
    paths = target_paths(payload)
    assert paths is not None
    spelled, resolved = paths
    reason = None
    if protected(spelled) or protected(resolved):
        reason = (
            "Direct file access matches this project's secret-file guard. "
            "Use placeholder-only .env.example and ask the owner to provision real secrets. "
            "Do not route the same access through a shell to bypass this guard."
        )
    elif tool in WRITE_TOOLS and not inside_root(resolved):
        reason = (
            "Direct edit resolves outside the project, including through a symlink. "
            "Keep this task's edits inside the project or explain the scope conflict."
        )
    if reason:
        emit("PreToolUse", permissionDecision="deny", permissionDecisionReason=reason)
    # No allow decision: normal Claude Code permissions remain in force.


def syntax_check(payload: dict) -> None:
    if payload.get("tool_name") not in {"Write", "Edit"}:
        return
    paths = target_paths(payload)
    assert paths is not None
    spelled, path = paths
    if protected(spelled) or protected(path) or not inside_root(path):
        return
    if path.suffix not in {".py", ".json"} or not path.is_file():
        return
    if path.stat().st_size > MAX_FILE:
        emit("PostToolUse", additionalContext="Syntax check skipped: file exceeds the 2 MiB hook limit. Run a focused check explicitly.")
        return
    try:
        if path.suffix == ".py":
            with tokenize.open(path) as source:
                ast.parse(source.read(), filename=str(path))
        else:
            json.loads(path.read_text(encoding="utf-8"))
    except SyntaxError as error:
        emit("PostToolUse", additionalContext=f"Post-edit Python syntax failed at line {error.lineno}: {error.msg}. The edit already happened; fix and recheck before claiming success.")
    except json.JSONDecodeError as error:
        emit("PostToolUse", additionalContext=f"Post-edit JSON syntax failed at line {error.lineno}, column {error.colno}: {error.msg}. The edit already happened; fix and recheck before claiming success.")
    except (OSError, UnicodeError):
        emit("PostToolUse", additionalContext="Post-edit syntax validation could not read/decode this file. Run an explicit focused check; no success is implied.")
    # A clean syntax check is silent. Syntax validity is not behavioral correctness.


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    modes = {"orient": ("SessionStart", orient), "guard": ("PreToolUse", guard), "syntax": ("PostToolUse", syntax_check)}
    if mode not in modes:
        print("Expected one mode: orient, guard, or syntax.", file=sys.stderr)
        return 1
    event, handler = modes[mode]
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT + 1)
        if len(raw) > MAX_INPUT:
            raise ValueError("hook input exceeds the 8 MiB limit")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("hook_event_name") != event:
            raise ValueError("unexpected hook event or input shape")
        handler(payload)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        # Do not echo payloads, source contents, paths or credentials.
        print(f"RAG Foundry hook could not validate input ({type(error).__name__}). Check hook configuration.", file=sys.stderr)
        return 2 if mode == "guard" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
