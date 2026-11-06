"""Offline byte identity checks for migration 0004 prompt seeds."""

import hashlib
import importlib.util
from pathlib import Path

from rag_foundry_answering.prompt import GROUNDED_SYSTEM, GROUNDED_SYSTEM_V1, UNGROUNDED_SYSTEM

_MIGRATION = (
    Path(__file__).parents[2]
    / "packages/storage/src/rag_foundry_storage/sql/migrations/versions/0004_prompt_templates.py"
)
_SPEC = importlib.util.spec_from_file_location("migration_0004", _MIGRATION)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_prompt_seed_texts_match_current_prompts_byte_for_byte():
    assert (_MODULE._GROUNDED_V1, _MODULE._GROUNDED_V2, _MODULE._UNGROUNDED_V1) == (
        GROUNDED_SYSTEM_V1,
        GROUNDED_SYSTEM,
        UNGROUNDED_SYSTEM,
    )
    assert all(text.isascii() for text in (_MODULE._GROUNDED_V1, _MODULE._GROUNDED_V2, _MODULE._UNGROUNDED_V1))


def test_prompt_seed_hashes_match_recorded_sha256_values():
    assert tuple(
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in (_MODULE._GROUNDED_V1, _MODULE._GROUNDED_V2, _MODULE._UNGROUNDED_V1)
    ) == (
        "73e46a3d97b96d5d2663e42b039310afb900629247dd7fb4a86403a3c8ddf07d",
        "fc18e5aa821a6678522db84e009ee1e6d4d4a80bcbad633c645f573cd82766ed",
        "de8c323862a79698d80a318364fcb0babd6388c5c8f8769ba8ef5a6bb5320a99",
    )
