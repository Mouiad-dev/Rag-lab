"""Dev/test CLI: run ExtractionService for one stored document version (ADR-0010 D4).

Not an HTTP route and not a job; the queued job arrives at P8.4. Prints safe summary fields only: never extracted
text, paths, URLs or secrets. Refuses in prod because there is no auth context yet (P10.1).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from rag_foundry_core.settings import Settings, load_settings
from rag_foundry_domain import LOCAL_DEV_SCOPE, IntakeError
from rag_foundry_domain.extraction import ExtractionOutcome
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from rag_foundry_storage.local_store import LocalObjectStore
from rag_foundry_storage.sql.engine import create_sessionmaker
from rag_foundry_storage.sql.extraction_repository import SqlExtractionRepository
from rag_foundry_storage.sql.repository import SqlDocumentRepository

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _fail(exit_code: int, message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return exit_code


def _summary_lines(outcome: ExtractionOutcome) -> list[str]:
    view = outcome.view
    page_count = "none" if view.page_count is None else str(view.page_count)
    warnings = ",".join(view.warnings) if view.warnings else "none"
    return [
        f"view_id={view.id}",
        f"created={'true' if outcome.created else 'false'}",
        f"char_count={view.char_count}",
        f"block_count={view.block_count}",
        f"page_count={page_count}",
        f"warnings={warnings}",
    ]


async def _extract(settings: Settings, document_id: str, version_id: str) -> ExtractionOutcome:
    assert settings.database_url is not None  # checked by the caller
    sessionmaker = create_sessionmaker(settings.database_url)
    service = ExtractionService(
        LocalObjectStore(settings.object_store_root),
        SqlDocumentRepository(sessionmaker),
        SqlExtractionRepository(sessionmaker),
        extractors=EXTRACTORS,
        limits=ExtractionLimits(),
    )
    try:
        return await service.extract(LOCAL_DEV_SCOPE, document_id, version_id)
    finally:
        # Close pooled connections while the loop still runs (SQLAlchemy asyncio requirement).
        await sessionmaker.kw["bind"].dispose()


def _extract_one(settings: Settings, document_id: str, version_id: str) -> ExtractionOutcome:
    # psycopg async cannot use the Windows Proactor loop (same workaround as the storage tests).
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        return runner.run(_extract(settings, document_id, version_id))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-foundry-extract",
        description="Dev/test only: extract one stored document version and print a safe summary.",
    )
    parser.add_argument("--document-id", required=True, metavar="ID", help="document ID in the local dev scope")
    parser.add_argument("--version-id", required=True, metavar="ID", help="document version ID to extract")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ValueError:
        # The settings error can name unknown variables; keep the output fixed and value-free.
        return _fail(EXIT_USAGE, "invalid RAG_FOUNDRY_* settings")

    if settings.environment == "prod":
        return _fail(EXIT_USAGE, "no auth context; this CLI is refused in prod")
    if settings.database_url is None:
        return _fail(EXIT_USAGE, "RAG_FOUNDRY_DATABASE_URL is not set")

    try:
        outcome = _extract_one(settings, args.document_id, args.version_id)
    except IntakeError as exc:
        return _fail(EXIT_FAILED, exc.code)
    except Exception as exc:  # no traceback: it can echo connection details; only the class name is printed
        return _fail(EXIT_FAILED, f"unexpected {type(exc).__name__}")

    for line in _summary_lines(outcome):
        print(line)
    return EXIT_OK
