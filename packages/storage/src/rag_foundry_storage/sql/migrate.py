"""Run Alembic migrations from code. The URL goes through `config.attributes`, never an ini file or a log."""

import sys
from pathlib import Path
from typing import Literal

from alembic import command
from alembic.config import Config
from pydantic import SecretStr

_SCRIPT_LOCATION = Path(__file__).resolve().parent / "migrations"


def run_migrations(url: SecretStr, action: Literal["upgrade", "downgrade"], revision: str) -> None:
    """Synchronous: env.py runs its own event loop, so call this outside any running loop."""
    if action not in ("upgrade", "downgrade"):
        raise ValueError("action must be 'upgrade' or 'downgrade'")
    config = Config()
    config.set_main_option("script_location", str(_SCRIPT_LOCATION))
    config.attributes["database_url"] = url
    if action == "upgrade":
        command.upgrade(config, revision)
    else:
        command.downgrade(config, revision)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    action, revision = "upgrade", "head"
    if args and args[0] in ("upgrade", "downgrade"):
        action, args = args[0], args[1:]
        revision = "head" if action == "upgrade" else ""
    if args:
        revision = args[0]
        args = args[1:]
    if args or not revision:
        print("usage: rag-foundry-migrate [upgrade|downgrade] [revision]", file=sys.stderr)
        return 2

    from rag_foundry_core.settings import load_settings

    try:
        url = load_settings().database_url
    except ValueError:
        print("invalid configuration: check RAG_FOUNDRY_* variables (values are not shown)", file=sys.stderr)
        return 2
    if url is None:
        print("RAG_FOUNDRY_DATABASE_URL is not set; nothing to migrate", file=sys.stderr)
        return 2
    try:
        run_migrations(url, action, revision)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - never echo driver text: it can contain the URL
        print(f"migration failed ({type(exc).__name__}); see database logs", file=sys.stderr)
        return 1
    print(f"migration {action} {revision} done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
