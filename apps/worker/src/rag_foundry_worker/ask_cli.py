"""Run one local grounded or ungrounded answer request."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError
from rag_foundry_answering.runner import LocalRunner, ollama_model_factory
from rag_foundry_contracts.runs import RunRequest, RunResponse
from rag_foundry_core.settings import load_settings
from rag_foundry_domain import LOCAL_DEV_SCOPE, IntakeError
from rag_foundry_domain.errors import PromptError, SettingsError
from rag_foundry_domain.runtime_settings import RuntimeSettings, load_runtime_settings_file
from rag_foundry_domain.settings_store import RuntimeSettingsService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.errors import ProviderError
from rag_foundry_storage.local_store import LocalObjectStore
from rag_foundry_storage.sql.engine import create_sessionmaker
from rag_foundry_storage.sql.extraction_repository import SqlExtractionRepository
from rag_foundry_storage.sql.index_repository import SqlIndexRepository
from rag_foundry_storage.sql.prompt_store import SqlPromptTemplateStore
from rag_foundry_storage.sql.settings_repository import SqlSettingsOverrideRepository

from rag_foundry_worker.extract_cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, _fail


def format_response(resp: RunResponse) -> str:
    return json.dumps(resp.model_dump(mode="json"), ensure_ascii=True, indent=2)


async def _ask(settings, request: RunRequest) -> RunResponse:
    maker = create_sessionmaker(settings.database_url)
    try:
        base, file_keys = (RuntimeSettings(), frozenset())
        if settings.runtime_config_path is not None:
            base, file_keys = load_runtime_settings_file(settings.runtime_config_path)
        runtime = RuntimeSettingsService(base, file_keys=file_keys, repo=SqlSettingsOverrideRepository(maker))
        store = LocalObjectStore(settings.object_store_root)
        views = SqlExtractionRepository(maker)
        anchors = AnchorService(store, views, provider_namespace=settings.provider_namespace)
        runner = LocalRunner(
            SqlIndexRepository(maker),
            anchors,
            model_factory=ollama_model_factory(settings.ollama_base_url),
            settings=runtime,
            prompts=SqlPromptTemplateStore(maker),
        )
        return await runner.run(LOCAL_DEV_SCOPE, request)
    finally:
        await maker.kw["bind"].dispose()


def _run(settings, request):
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        return runner.run(_ask(settings, request))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-foundry-ask")
    question = parser.add_mutually_exclusive_group(required=True)
    question.add_argument("--question")
    question.add_argument("--question-file", type=Path)
    parser.add_argument("--index-version-id")
    parser.add_argument("--technique", default="R02")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-score", type=float)
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
        text = args.question if args.question is not None else args.question_file.read_text(encoding="utf-8")
        if text.endswith("\n"):
            text = text[:-1]
        request = RunRequest(
            question=text,
            index_version_id=args.index_version_id,
            technique_id=args.technique,
            k=args.k,
            min_score=args.min_score,
        )
    except (OSError, UnicodeError, ValidationError, ValueError):
        return _fail(EXIT_USAGE, "invalid run request")
    try:
        response = _run(settings, request)
    except SettingsError as exc:
        if exc.code == "store_unavailable":
            return _fail(EXIT_FAILED, "settings unavailable")
        return _fail(EXIT_USAGE, "invalid runtime settings")
    except PromptError as exc:
        if exc.code in ("store_unavailable", "no_active_template"):
            return _fail(EXIT_FAILED, "prompt template unavailable")
        return _fail(EXIT_FAILED, "prompt template invalid")
    except IntakeError as exc:
        return _fail(EXIT_FAILED, exc.code)
    except ProviderError:
        return _fail(EXIT_FAILED, "provider_unavailable")
    except Exception as exc:
        return _fail(EXIT_FAILED, f"unexpected {type(exc).__name__}")
    print(format_response(response))
    return EXIT_OK
