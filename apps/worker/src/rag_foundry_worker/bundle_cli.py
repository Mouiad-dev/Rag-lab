"""Local development source bundle export and verification commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rag_foundry_core.settings import load_settings
from rag_foundry_domain import LOCAL_DEV_SCOPE
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.bundle import (
    BundleAnchorCollector,
    BundleError,
    BundleExportService,
    import_bundle,
    read_bundle_dir,
    write_bundle_dir,
)
from rag_foundry_storage.local_store import LocalObjectStore
from rag_foundry_storage.sql.database import open_database, run_db_main

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


def _fail(code: int, message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


async def _export(settings, args):
    async with open_database(settings.database_url) as db:
        store = LocalObjectStore(settings.object_store_root)
        anchors = AnchorService(store, db.uow, provider_namespace=settings.provider_namespace)
        collector = BundleAnchorCollector(store, db.uow, anchors)
        values = await collector.anchors_for_views(LOCAL_DEV_SCOPE, args.views, args.index)
        service = BundleExportService(
            store,
            db.uow,
            provider_namespace=settings.provider_namespace,
            clock=lambda: datetime.now(UTC),
        )
        files = await service.export(LOCAL_DEV_SCOPE, args.views, values)
        write_bundle_dir(files, args.out)
    return files


def _run(settings, args):
    if args.command == "verify":
        return import_bundle(read_bundle_dir(args.directory))
    return run_db_main(_export(settings, args))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-foundry-bundle")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--views", nargs="+", required=True, metavar="ID")
    export.add_argument("--index", metavar="ID")
    export.add_argument("--out", required=True, type=Path, metavar="DIR")
    verify = commands.add_parser("verify")
    verify.add_argument("directory", type=Path, metavar="DIR")
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
    except ValueError:
        return _fail(EXIT_USAGE, "invalid RAG_FOUNDRY_* settings")
    if settings.environment == "prod":
        return _fail(EXIT_USAGE, "no auth context; this CLI is refused in prod")
    if args.command == "export" and settings.database_url is None:
        return _fail(EXIT_USAGE, "RAG_FOUNDRY_DATABASE_URL is not set")
    try:
        result = _run(settings, args)
    except BundleError as exc:
        return _fail(EXIT_FAILED, exc.code)
    except Exception as exc:
        return _fail(EXIT_FAILED, f"unexpected {type(exc).__name__}")
    if args.command == "export":
        files = result
        imported = import_bundle(files)
        print(f"bundle_id={imported.manifest.bundle_id}")
        print(f"files={len(files)}")
        print(f"bytes={sum(map(len, files.values()))}")
        print(f"anchors={len(imported.anchors)}")
        print(f"views={len(imported.documents)}")
    else:
        imported = result
        print(f"bundle_id={imported.manifest.bundle_id}")
        print(f"files={len(imported.manifest.assets) + 1}")
        print(f"bytes={sum(len(body) for body in read_bundle_dir(args.directory).values())}")
        print(f"anchors={len(imported.anchors)}")
        print(f"views={len(imported.documents)}")
    return EXIT_OK
