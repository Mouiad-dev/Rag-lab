"""Behavioral checks of the shipped hooks, without Claude or network access."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[2] / ".claude/hooks/dev_hooks.py"


class HookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rag hook tests ")
        self.base = Path(self.tmp.name)
        self.root = self.base / "project with spaces"
        self.script = self.root / ".claude/hooks/dev_hooks.py"
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(SOURCE, self.script)

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, mode, tool="Write", path="sample.py", raw=None, **extra):
        events = {"orient": "SessionStart", "guard": "PreToolUse", "syntax": "PostToolUse"}
        payload = {"hook_event_name": events[mode], "cwd": str(self.root), "tool_name": tool, "tool_input": {"file_path": path}}
        payload.update(extra)
        result = subprocess.run([sys.executable, str(self.script), mode], input=raw if raw is not None else json.dumps(payload), text=True, capture_output=True, timeout=5)
        output = json.loads(result.stdout) if result.stdout.strip() else None
        return result, output

    def assert_denied(self, output):
        self.assertEqual(output["hookSpecificOutput"]["permissionDecision"], "deny")

    def test_orientation_is_context_only(self):
        result, output = self.call("orient")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(output["hookSpecificOutput"]["hookEventName"], "SessionStart")
        context = output["hookSpecificOutput"]["additionalContext"]
        self.assertIn("PROGRESS.md", context)
        self.assertIn("all 72 catalogue entries", context)

    def test_normal_write_has_no_automatic_allow(self):
        result, output = self.call("guard")
        self.assertEqual(result.returncode, 0)
        self.assertIsNone(output)

    def test_env_read_and_write_denied(self):
        for tool in ("Read", "Write", "Edit"):
            for name in (".env", ".env.production", "nested/.env.local"):
                with self.subTest(tool=tool, name=name):
                    result, output = self.call("guard", tool, name)
                    self.assertEqual(result.returncode, 0)
                    self.assert_denied(output)

    def test_templates_allowed(self):
        for name in (".env.example", ".env.template"):
            result, output = self.call("guard", path=name)
            self.assertEqual(result.returncode, 0)
            self.assertIsNone(output)

    def test_secret_directories_override_template_exception(self):
        for name in ("secrets/.env.example", ".aws/config", ".ssh/config", "secrets/normal.json"):
            _, output = self.call("guard", path=name)
            self.assert_denied(output)

    def test_private_keys_denied(self):
        for name in ("id_ed25519", "tls.pem", "client.p12", "credentials.json"):
            _, output = self.call("guard", path=name)
            self.assert_denied(output)

    def test_path_traversal_write_denied(self):
        _, output = self.call("guard", path="../outside.py")
        self.assert_denied(output)

    def test_normal_external_read_not_claimed_protected(self):
        _, output = self.call("guard", tool="Read", path=str(self.base / "public.txt"))
        self.assertIsNone(output)

    def test_symlink_to_outside_write_denied(self):
        target = self.base / "outside.py"
        target.write_text("x = 1\n")
        try:
            (self.root / "link.py").symlink_to(target)
        except OSError:
            self.skipTest("Symlinks not supported in this environment")
        _, output = self.call("guard", path="link.py")
        self.assert_denied(output)

    def test_symlink_to_secret_read_denied(self):
        secret = self.root / ".env"
        secret.write_text("EXAMPLE=not-a-real-secret\n")
        try:
            (self.root / "innocent.txt").symlink_to(secret)
        except OSError:
            self.skipTest("Symlinks not supported in this environment")
        _, output = self.call("guard", tool="Read", path="innocent.txt")
        self.assert_denied(output)

    def test_bad_json_pretool_exits_two_without_echo(self):
        result, output = self.call("guard", raw="{invalid-private-content")
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(output)
        self.assertNotIn("private-content", result.stderr)

    def test_missing_cwd_relative_path_fails_closed(self):
        result, _ = self.call("guard", cwd=None)
        self.assertEqual(result.returncode, 2)

    def test_wrong_event_fails_closed(self):
        result, _ = self.call("guard", hook_event_name="PostToolUse")
        self.assertEqual(result.returncode, 2)

    def test_notebook_target_is_guarded(self):
        _, output = self.call("guard", tool="NotebookEdit", tool_input={"notebook_path": "../external.ipynb"})
        self.assert_denied(output)

    def test_non_file_tool_not_claimed_protected(self):
        result, output = self.call("guard", tool="Bash", tool_input={"command": "echo example"})
        self.assertEqual(result.returncode, 0)
        self.assertIsNone(output)

    def test_python_is_parsed_not_executed(self):
        (self.root / "sample.py").write_text("raise RuntimeError('must never execute')\n")
        result, output = self.call("syntax")
        self.assertEqual(result.returncode, 0)
        self.assertIsNone(output)

    def test_invalid_python_reports_after_edit_without_changing_it(self):
        path = self.root / "sample.py"
        body = "def broken(:\n"
        path.write_text(body)
        result, output = self.call("syntax")
        self.assertEqual(result.returncode, 0)
        self.assertIn("syntax failed", output["hookSpecificOutput"]["additionalContext"])
        self.assertEqual(path.read_text(), body)

    def test_invalid_json_reports_location_without_source(self):
        (self.root / "sample.json").write_text('{"private-marker": }')
        result, output = self.call("syntax", path="sample.json")
        self.assertEqual(result.returncode, 0)
        self.assertIn("JSON syntax failed", output["hookSpecificOutput"]["additionalContext"])
        self.assertNotIn("private-marker", result.stdout)

    def test_valid_json_silent(self):
        (self.root / "sample.json").write_text('{"ok": true}')
        result, output = self.call("syntax", path="sample.json")
        self.assertEqual(result.returncode, 0)
        self.assertIsNone(output)

    def test_secret_json_is_not_read_by_posthook(self):
        folder = self.root / "secrets"
        folder.mkdir()
        (folder / "broken.json").write_text("not-json")
        result, output = self.call("syntax", path="secrets/broken.json")
        self.assertEqual(result.returncode, 0)
        self.assertIsNone(output)

    def test_large_file_reports_skip(self):
        (self.root / "sample.py").write_text("#" + "x" * (2 * 1024 * 1024))
        _, output = self.call("syntax")
        self.assertIn("skipped", output["hookSpecificOutput"]["additionalContext"])


if __name__ == "__main__":
    unittest.main()
