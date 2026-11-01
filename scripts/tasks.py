#!/usr/bin/env python3
"""Run repository checks in one command. Works on Windows without make.

Standard library only. Tools (ruff, pytest) run through the current interpreter,
so start this script with `uv run python scripts/tasks.py <command>`.
Commands are executed as argument lists without a shell.
"""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PY = sys.executable

GROUPS = {
    "repo": [
        [PY, str(SCRIPTS / "verify_package.py")],
        [PY, "-m", "unittest", "discover", "-s", "tests/hooks", "-v"],
        [PY, str(SCRIPTS / "build_flow.py"), "--check"],
    ],
    "lint": [
        [PY, "-m", "ruff", "check", "."],
        [PY, "-m", "ruff", "format", "--check", "."],
    ],
    "format": [
        [PY, "-m", "ruff", "format", "."],
        [PY, "-m", "ruff", "check", "--fix", "."],
    ],
    "test": [
        [PY, "-m", "pytest"],
    ],
}


def run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def run_group(name: str) -> int:
    for cmd in GROUPS[name]:
        code = run(cmd)
        if code != 0:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAG Foundry repository checks.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in GROUPS:
        sub.add_parser(name, help=f"run the {name} checks")
    sub.add_parser("check", help="run repo, then lint, then test; stop at first failure")
    args = parser.parse_args(argv)

    if args.command == "check":
        for name in ("repo", "lint", "test"):
            code = run_group(name)
            if code != 0:
                return code
        return 0
    return run_group(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
