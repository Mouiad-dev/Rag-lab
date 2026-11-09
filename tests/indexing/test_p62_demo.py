import datetime
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "docs/techniques/lessons/demos/p62_representations_demo.py"
EVIDENCE = ROOT / "docs/evidence/P6.2/representations_demo.md"
LIVE = "## Live run (orchestrator)"
LIVE_SECTION = (
    LIVE
    + "\n\n"
    + "Filled by the orchestrator after the P6.2-05 live test (`tests/live/test_representation_builders_live.py`, "
    "local Ollama, cost 0). Not run by this demo.\n\n"
    "| item | value |\n|---|---|\n"
    "| date | <orchestrator> |\n"
    "| command | <orchestrator> |\n"
    "| chat model | <orchestrator> |\n"
    "| embedding model | <orchestrator> |\n"
    "| result | <orchestrator> |\n"
)


def run_demo():
    done = subprocess.run([sys.executable, str(DEMO)], capture_output=True, check=True, cwd=ROOT)
    assert done.stderr == b""
    return done.stdout.decode("utf-8")


def test_demo_output_is_deterministic_ascii_and_matches_evidence():
    first = run_demo()
    assert first == run_demo()
    assert DEMO.read_bytes().decode("ascii")
    if os.getenv("RAG_WRITE_EVIDENCE") == "1":
        header = (
            f"<!-- generated {datetime.date.today().isoformat()} by "
            "`uv run python docs/techniques/lessons/demos/p62_representations_demo.py`; "
            "offline; scripted chat and scripted embedder; no model; not a benchmark -->\n"
        )
        with EVIDENCE.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(header + first + LIVE_SECTION)
        return
    committed = EVIDENCE.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed.startswith("<!-- generated ")
    body, live = committed.split("\n", 1)[1].split(LIVE, 1)
    assert body == first
    assert LIVE + live == LIVE_SECTION
