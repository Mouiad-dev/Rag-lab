"""Export committed JSON Schemas: ``python -m rag_foundry_contracts.export <dir>``."""

import json
import sys
from pathlib import Path

from rag_foundry_contracts.source import SourceAnchor


def schemas() -> dict[str, dict]:
    return {"source-anchor.v1.json": SourceAnchor.model_json_schema()}


def render(schema: dict) -> str:
    return json.dumps(schema, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m rag_foundry_contracts.export <dir>", file=sys.stderr)
        return 2
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    for name, schema in schemas().items():
        with (out / name).open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(render(schema))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
