"""Seeded prompt store used by answering tests."""

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

from rag_foundry_domain.prompts import InMemoryPromptTemplateStore, PromptActivation, PromptTemplate, prompt_sha256

_MIGRATION = (
    Path(__file__).parents[2]
    / "packages/storage/src/rag_foundry_storage/sql/migrations/versions/0004_prompt_templates.py"
)
_SPEC = importlib.util.spec_from_file_location("prompt_seed_0004", _MIGRATION)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def seeded_prompt_store():
    rows = (
        ("grounded-answer-v1", "grounded-answer", 1, "grounded", _MODULE._GROUNDED_V1),
        ("grounded-answer-v2", "grounded-answer", 2, "grounded", _MODULE._GROUNDED_V2),
        ("ungrounded-v1", "ungrounded", 1, "ungrounded", _MODULE._UNGROUNDED_V1),
    )
    templates = tuple(
        PromptTemplate(
            id=template_id,
            family=family,
            version=version,
            purpose=purpose,
            system_text=system_text,
            content_sha256=prompt_sha256(system_text),
        )
        for template_id, family, version, purpose, system_text in rows
    )
    activations = tuple(
        PromptActivation(
            seq=seq,
            purpose=purpose,
            template_id=template_id,
            activated_by="migration:0004",
            activated_at=datetime.now(UTC),
            decision_ref="ADR-0013",
        )
        for seq, (purpose, template_id) in enumerate(
            (("grounded", "grounded-answer-v1"), ("grounded", "grounded-answer-v2"), ("ungrounded", "ungrounded-v1")),
            1,
        )
    )
    return InMemoryPromptTemplateStore(templates, activations)


def prompt_templates(grounded_version=2):
    grounded_text = _MODULE._GROUNDED_V1 if grounded_version == 1 else _MODULE._GROUNDED_V2
    return PromptTemplate(
        id=f"grounded-answer-v{grounded_version}",
        family="grounded-answer",
        version=grounded_version,
        purpose="grounded",
        system_text=grounded_text,
        content_sha256=prompt_sha256(grounded_text),
    ), PromptTemplate(
        id="ungrounded-v1",
        family="ungrounded",
        version=1,
        purpose="ungrounded",
        system_text=_MODULE._UNGROUNDED_V1,
        content_sha256=prompt_sha256(_MODULE._UNGROUNDED_V1),
    )
