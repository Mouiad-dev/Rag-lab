"""Local development index build and inspection commands."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from rag_foundry_answering.runner import RUN_RETRY
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_core.settings import Settings, load_settings
from rag_foundry_domain import LOCAL_DEV_SCOPE, IntakeError
from rag_foundry_indexing import IndexService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.adapters.ollama import OllamaEmbeddingModel, ollama_embedding_profile
from rag_foundry_providers.errors import ProviderError
from rag_foundry_providers.ledger import RecordingEmbeddingModel, UsageLedger
from rag_foundry_providers.resilience import ResilientEmbeddingModel
from rag_foundry_storage.local_store import LocalObjectStore
from rag_foundry_storage.sql.engine import create_sessionmaker
from rag_foundry_storage.sql.extraction_repository import SqlExtractionRepository
from rag_foundry_storage.sql.index_repository import SqlIndexRepository

from rag_foundry_worker.extract_cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, _fail


def inspection_line(row, show_text: bool = False) -> str:
    value = {
        "ordinal": row.ordinal,
        "extraction_view_id": row.extraction_view_id,
        "start": row.start,
        "end": row.end,
        "split_level": row.split_level,
        "block_ids": list(row.block_ids),
        "text_sha256": row.text_sha256,
        "anchor_ids": [anchor.anchor_id for anchor in row.anchors],
    }
    if show_text:
        value["text"] = row.text
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


async def _index(settings: Settings, args):
    maker = create_sessionmaker(settings.database_url)
    try:
        store = LocalObjectStore(settings.object_store_root)
        views, index = SqlExtractionRepository(maker), SqlIndexRepository(maker)
        anchors = AnchorService(store, views, provider_namespace=settings.provider_namespace)
        service = IndexService(store, views, index, anchors)
        if args.command == "build":
            chunker = RecursiveChunker(RecursiveConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap))
            embedder = ResilientEmbeddingModel(
                RecordingEmbeddingModel(
                    OllamaEmbeddingModel(ollama_embedding_profile(), api_base=settings.ollama_base_url),
                    UsageLedger(),
                ),
                RUN_RETRY,
            )
            version = await service.build(LOCAL_DEV_SCOPE, tuple(args.view_id), chunker, embedder)
            return [
                f"index_version_id={version.id}",
                f"state={version.state.value}",
                f"chunk_count={version.chunk_count}",
                f"unanchored_chunk_count={version.unanchored_chunk_count}",
                f"embedding_model={version.embedding_model}",
                f"embedding_dim={version.embedding_dim}",
                f"embedding_key={version.embedding_key}",
            ]
        rows = await service.inspect_chunks(
            LOCAL_DEV_SCOPE, args.index_version_id, offset=args.offset, limit=args.limit
        )
        return [inspection_line(row, args.show_text) for row in rows]
    finally:
        await maker.kw["bind"].dispose()


def _run(settings, args):
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        return runner.run(_index(settings, args))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-foundry-index")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--view-id", action="append", required=True)
    build.add_argument("--chunk-size", type=int, default=800)
    build.add_argument("--chunk-overlap", type=int, default=100)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--index-version-id", required=True)
    inspect.add_argument("--offset", type=int, default=0)
    inspect.add_argument("--limit", type=int, default=20)
    inspect.add_argument("--show-text", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
    except ValueError:
        return _fail(EXIT_USAGE, "invalid RAG_FOUNDRY_* settings")
    if settings.environment == "prod":
        return _fail(EXIT_USAGE, "no auth context; this CLI is refused in prod")
    if settings.database_url is None:
        return _fail(EXIT_USAGE, "RAG_FOUNDRY_DATABASE_URL is not set")
    try:
        lines = _run(settings, args)
    except IntakeError as exc:
        return _fail(EXIT_FAILED, exc.code)
    except ProviderError:
        return _fail(EXIT_FAILED, "provider_unavailable")
    except Exception as exc:
        return _fail(EXIT_FAILED, f"unexpected {type(exc).__name__}")
    for line in lines:
        print(line)
    return EXIT_OK
