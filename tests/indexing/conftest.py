import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for relative in ("tests/indexing", "tests/ingestion"):
    path = str(ROOT / relative)
    if path not in sys.path:
        sys.path.insert(0, path)
