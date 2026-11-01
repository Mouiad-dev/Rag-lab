#!/usr/bin/env python3
"""Validate the repository's planning contracts, not the future application.

Standard library only. No network, no installs, no model calls.
"""

import ast
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".idea", ".venv", "node_modules", "private", "__pycache__"}
PHASES, STEPS_PER_PHASE = 10, 5
EXPECTED_AGENTS = {"planner": "opus", "coder": "sonnet", "coder-light": "haiku", "reviewer": "opus"}
EXPECTED_RULES = 9
REQUIRED = [
    "README.md", "CLAUDE.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md", ".gitignore",
    "docs/architecture/PROJECT_ARCH.md", "docs/architecture/SUMMARY.md",
    "docs/architecture/REUSE_AND_INTEGRATION.md", "docs/architecture/SOURCE_ANCHORS_AND_MEDIA.md",
    "docs/plan/ROADMAP.md", "docs/plan/AGENT_WORKFLOW.md",
    "docs/techniques/TECHNIQUE_CATALOG.md", "docs/techniques/CACHE_AND_KNOWLEDGE_LABS.md",
    "docs/decisions/ADR-0001-provider-agnostic-layer.md", "docs/decisions/ADR-0002-technique-plugin-system.md",
    "docs/decisions/ADR-0003-in-app-tech-scout.md", "docs/decisions/ADR-0004-open-source-license.md",
    "docs/templates/TECHNIQUE.md", "docs/templates/EXPERIMENT.md", "docs/templates/ADR.md", "docs/templates/TASK.md",
    "docs/PROGRESS.md", "docs/HOOKS.md", "docs/SOURCES.md", "docs/VALIDATION.md",
    "config/techniques.json", "config/experiment.example.json", "config/source-media.example.json",
    "config/providers.example.json", "config/scout.example.json",
    ".claude/settings.example.json", ".claude/hooks/dev_hooks.py", "tests/hooks/test_dev_hooks.py",
    ".github/workflows/ci.yml", ".github/PULL_REQUEST_TEMPLATE.md",
]
PRIVATE_MARKERS = re.compile(r"\bTora\b|Tiger Lab|tora-main|DIPANSHU", re.IGNORECASE)


def files(pattern):
    for path in ROOT.rglob(pattern):
        if not SKIP_DIRS.intersection(path.relative_to(ROOT).parts[:-1]):
            yield path


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def check_syntax(errors):
    for path in files("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as error:
            errors.append(f"Invalid Python: {path.relative_to(ROOT)}: {type(error).__name__}")
    for path in files("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            errors.append(f"Invalid JSON: {path.relative_to(ROOT)}")


def check_roadmap(errors):
    road = read("docs/plan/ROADMAP.md")
    phases = re.findall(r"^## Phase (\d+):", road, re.M)
    if phases != [str(n) for n in range(1, PHASES + 1)]:
        errors.append(f"ROADMAP.md must have phases 1..{PHASES} in order; found {phases}.")
    steps = re.findall(r"^\| (P\d+\.\d+) \|", road, re.M)
    expected = [f"P{p}.{s}" for p in range(1, PHASES + 1) for s in range(1, STEPS_PER_PHASE + 1)]
    if steps != expected:
        errors.append(f"ROADMAP.md must have exactly {STEPS_PER_PHASE} steps per phase ({len(expected)} total); found {len(steps)}.")
    progress = re.findall(r"^\| (P\d+\.\d+) \|", read("docs/PROGRESS.md"), re.M)
    if progress != expected:
        errors.append(f"PROGRESS.md must track the same {len(expected)} steps as ROADMAP.md; found {len(progress)}.")
    return set(expected)


def check_catalogue(errors, roadmap_steps):
    registry = json.loads(read("config/techniques.json"))
    entries = registry["techniques"]
    ids = [entry["id"] for entry in entries]
    expected = {f"C{i:02}" for i in range(1, 22)} | {f"R{i:02}" for i in range(1, 52)}
    if len(ids) != len(set(ids)) or not expected.issubset(ids):
        errors.append("Technique IDs must be unique and include the committed 72-entry curriculum.")
    catalogue = read("docs/techniques/TECHNIQUE_CATALOG.md")
    for entry in entries:
        tid = entry["id"]
        if entry["status"] not in registry["allowed_statuses"]:
            errors.append(f"Invalid status: {tid}")
        if entry.get("catalog_status") not in registry["allowed_catalog_statuses"]:
            errors.append(f"Invalid catalog_status: {tid}")
        if entry.get("catalog_status") == "proposed":
            errors.append(f"Proposed entries belong in the app database, not the repository catalogue: {tid}")
        if entry.get("category") not in registry["allowed_categories"]:
            errors.append(f"Invalid category: {tid}")
        step = entry.get("roadmap_step")
        if step not in roadmap_steps:
            errors.append(f"Unknown roadmap_step for {tid}: {step}")
        elif entry.get("phase") != int(step[1:].split(".")[0]):
            errors.append(f"phase does not match roadmap_step: {tid}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(entry.get("date_added"))):
            errors.append(f"date_added must be YYYY-MM-DD: {tid}")
        source = entry.get("primary_source")
        if source is not None and not str(source).startswith("https://"):
            errors.append(f"primary_source must be null or an https URL: {tid}")
        if f"| {tid} | {entry['family']} / {step} |" not in catalogue:
            errors.append(f"Missing or out-of-sync catalogue row: {tid}")
        if entry["status"] == "validated" and not entry.get("evidence_path"):
            errors.append(f"Validated without evidence: {tid}")
        for field in ("evidence_path", "implementation_path", "design_reference"):
            if entry.get(field) and not (ROOT / entry[field]).exists():
                errors.append(f"Missing {field}: {tid}")
    if "\n\n| R47 |" in catalogue:
        errors.append("Catalogue table is broken before R47.")
    return len(entries)


def check_claude_config(errors):
    settings = json.loads(read(".claude/settings.example.json"))
    for event, mode in {"SessionStart": "orient", "PreToolUse": "guard", "PostToolUse": "syntax"}.items():
        definitions = settings.get("hooks", {}).get(event, [])
        if len(definitions) != 1 or len(definitions[0].get("hooks", [])) != 1:
            errors.append(f"Unexpected hook shape: {event}")
            continue
        command = definitions[0]["hooks"][0].get("command", "")
        if not command.startswith("python ") or not command.endswith(f'dev_hooks.py" {mode}'):
            errors.append(f"Unexpected hook command: {event}")
    rules = sorted((ROOT / ".claude/rules").glob("*.md"))
    if len(rules) != EXPECTED_RULES:
        errors.append(f"Expected {EXPECTED_RULES} rule files, found {len(rules)}.")
    for name, model in EXPECTED_AGENTS.items():
        path = ROOT / ".claude/agents" / f"{name}.md"
        if not path.is_file():
            errors.append(f"Missing agent: {name}")
            continue
        head = path.read_text(encoding="utf-8").split("---")
        meta = head[1] if len(head) > 2 else ""
        if f"name: {name}" not in meta or f"model: {model}" not in meta or "description:" not in meta or "tools:" not in meta:
            errors.append(f"Agent frontmatter incomplete or wrong model: {name}")
    if len(read("CLAUDE.md").splitlines()) > 60:
        errors.append("CLAUDE.md should stay under 60 lines (loaded every session).")
    if len(read("docs/architecture/SUMMARY.md").splitlines()) > 150:
        errors.append("SUMMARY.md should stay under 150 lines.")
    return len(rules)


def check_docs(errors):
    ignore = read(".gitignore").splitlines()
    for needed in ("private/", ".idea/", ".claude/settings.local.json"):
        if needed not in ignore:
            errors.append(f".gitignore must contain {needed}")
    for path in files("*.md"):
        body = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)\s]+)\)", body):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if target and not (path.parent / target).exists():
                errors.append(f"Broken local link in {path.relative_to(ROOT)}: {target}")
    this = Path(__file__).resolve()
    for pattern in ("*.md", "*.json", "*.py", "*.yml"):
        for path in files(pattern):
            if path.resolve() != this and PRIVATE_MARKERS.search(path.read_text(encoding="utf-8")):
                errors.append(f"Private/proprietary reference in public file: {path.relative_to(ROOT)}")


def main():
    errors = [f"Missing required file: {name}" for name in REQUIRED if not (ROOT / name).is_file()]
    check_syntax(errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    steps = check_roadmap(errors)
    count = check_catalogue(errors, steps)
    rules = check_claude_config(errors)
    check_docs(errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: structure, Python/JSON syntax, {count} catalogue entries (schema v2, in sync), "
          f"roadmap {PHASES}x{STEPS_PER_PHASE} steps tracked in PROGRESS, hook example commands, "
          f"{rules} rules, {len(EXPECTED_AGENTS)} agents, local Markdown links, no private references.")
    if (ROOT / ".claude/settings.json").exists():
        print("Note: .claude/settings.json exists and is active for everyone who opens this repo; see docs/HOOKS.md.")
    print("Scope: planning/configuration validation only; no application was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
