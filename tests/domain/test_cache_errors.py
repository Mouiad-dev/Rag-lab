import pytest
from rag_foundry_domain.errors import CacheError, RunError

NEW_RUN_CODES = (
    "cag_overflow",
    "cag_too_many_chunks",
    "cag_nonce_conflict",
    "cag_snapshot_mismatch",
    "cag_empty",
)


@pytest.mark.parametrize("code", NEW_RUN_CODES)
def test_new_run_codes_construct(code: str) -> None:
    assert code in RunError.CODES
    assert RunError(code).code == code


@pytest.mark.parametrize("code", CacheError.CODES)
def test_cache_codes_construct(code: str) -> None:
    assert CacheError(code).code == code


def test_cache_codes_exact() -> None:
    assert CacheError.CODES == (
        "cache_key_invalid",
        "cache_entry_invalid",
        "cache_scope_mismatch",
        "cache_unavailable",
        "cache_policy_invalid",
        "prefix_probe_invalid",
        "cache_exists",
    )


def test_unknown_codes_fail() -> None:
    with pytest.raises(ValueError):
        CacheError("nope")
    with pytest.raises(ValueError):
        RunError("cache_key_invalid")
