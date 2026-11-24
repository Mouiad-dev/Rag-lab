# Development hooks: installation, behavior and limits

These are Claude Code project hooks, not hooks for the application's future agents. They run local Python scripts when Claude Code emits configured events. Ordinary Claude chat/project instructions do not execute these scripts.

## Enable (opt-in)

The repository ships the hook configuration as `.claude/settings.example.json`, which Claude Code does **not** load. Hooks are off until you enable them, so they never interfere with a session by surprise.

1. Check Python 3.11+ is on the PATH of the shell Claude Code uses: `python --version`.
2. Run the checks below.
3. Copy the example to your personal, git-ignored settings file and restart Claude Code:
   - macOS/Linux/WSL/Git Bash: `cp .claude/settings.example.json .claude/settings.local.json`
   - PowerShell: `Copy-Item .claude/settings.example.json .claude/settings.local.json`
   If you already have `.claude/settings.local.json`, merge the `hooks` block instead of overwriting.
4. Inspect `/hooks` in Claude Code and try a harmless edit to a temporary `.py` file.

Interpreter: the commands use `python`. If your system only has `python3` (common on macOS/Linux), change the first word of each command to `python3`; on Windows you can use `py -3`. The commands use `$CLAUDE_PROJECT_DIR`, which works when Claude Code runs hooks through a POSIX-style shell (bash, Git Bash, WSL). For a native PowerShell hook shell, use `$env:CLAUDE_PROJECT_DIR` instead. Windows behavior inside a real Claude Code session has not been verified by this repository's tests.

Committing a shared `.claude/settings.json` would turn the hooks on for every contributor; do that only by explicit team decision.

To disable: delete the `hooks` block from `.claude/settings.local.json`.

## Included behavior

| Event | Action | Outcome |
|---|---|---|
| SessionStart | Supplies a short orientation and progress-file reminder | Context only; no files changed |
| PreToolUse on Read/Write/Edit/NotebookEdit | Checks direct target paths for selected secret names and blocks direct writes outside the project | Explicit deny on a match; otherwise no permission decision |
| PostToolUse on Write/Edit | Parses project `.py` using AST and `.json` using JSON parser, up to 2 MiB | Syntax error context; no rollback, source import, formatting or modification |

Examples of guarded paths: `.env`, `.env.production`, private key extensions/names, `secrets/`, `.ssh/`, `.aws/`, `credentials.json`, `service-account.json`. `.env.example` and `.env.template` are allowed as placeholder-only templates. Both spelled and resolved paths are checked to cover simple symlink cases. Permission to edit a filename is not proof its content contains no secrets.

Malformed PreToolUse input exits 2 so the direct action is blocked under the documented contract. Other malformed events report a hook error. Successful non-matching calls receive no automatic approval from this script; Claude Code's normal permissions still apply.

## What these hooks do not enforce

- They do not intercept Bash, Python subprocesses launched elsewhere, MCP tools, network requests, manual editor changes, or every possible file tool.
- They do not scan content for secrets, validate business logic, certify citations, or execute application tests.
- They are not a sandbox, OS access-control policy or a complete prevention mechanism. They cannot guarantee protection against races or malicious code changing the filesystem.
- PostToolUse happens after the edit; a reported syntax error does not undo the change.
- They do not automatically install packages, run model calls, send telemetry, log transcripts, autoformat files or create commits.
- No Stop hook is included: correctness is checked explicitly, without risking a repetitive stop-block loop or a full test run on every response.

Application security must be enforced in auth, database roles, tools, workers and sandbox infrastructure as described in `docs/architecture/PROJECT_ARCH.md`.

## Verify before use

```bash
python scripts/verify_package.py
python -m unittest discover -s tests/hooks -v
```

The tests simulate documented payloads in disposable directories. For actual activation, use a harmless test edit to a temporary `.py` file in Claude Code and inspect `/hooks` output. If Python is missing, install/configure Python before expecting hook protection; a missing interpreter or host-level timeout is outside this script's decisions.

## Troubleshooting

- Hook not listed: confirm project root and that the hooks block is in `.claude/settings.local.json` (or another loaded settings file); check JSON and local/user settings conflicts.
- Interpreter error: verify the configured executable from Claude Code's actual shell.
- Secret-path denial: use placeholder templates; let the owner provision secrets through environment/secret management. Do not bypass the same denial through Bash.
- Outside-project denial: inspect relative cwd, symlinks and task scope. Prefer a project-local target; explain legitimate external scope changes.
- Syntax warning: fix the reported line and run focused validation. This hook is not a formatter or type checker.

Reference: https://code.claude.com/docs/en/hooks
