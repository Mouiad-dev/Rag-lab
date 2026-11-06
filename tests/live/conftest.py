import sys
from pathlib import Path

# The index live test reuses the offline fixture and ingestion helpers.
ROOT = Path(__file__).resolve().parents[2]
for relative in ("tests/indexing", "tests/ingestion"):
    path = str(ROOT / relative)
    if path not in sys.path:
        sys.path.insert(0, path)
