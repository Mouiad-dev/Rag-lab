import pytest
from rag_foundry_domain.errors import ExtractionError, IndexingError, RunError

NEW_INDEXING_CODES = (
    "token_encoding_count_mismatch",
    "token_offsets_invalid",
    "late_chunk_span_empty",
    "late_chunk_window_invalid",
    "multivector_index_missing",
    "multivector_dim_mismatch",
    "visual_index_missing",
)
NEW_EXTRACTION_CODES = ("page_render_limit", "page_image_invalid")
NEW_RUN_CODES = (
    "selfrag_tokens_missing",
    "selfrag_logprobs_unavailable",
    "selfrag_output_invalid",
)


@pytest.mark.parametrize("code", NEW_INDEXING_CODES)
def test_new_indexing_codes_construct(code: str) -> None:
    assert code in IndexingError.CODES
    assert IndexingError(code).code == code


@pytest.mark.parametrize("code", NEW_EXTRACTION_CODES)
def test_new_extraction_codes_construct(code: str) -> None:
    assert code in ExtractionError.CODES
    assert ExtractionError(code).code == code


@pytest.mark.parametrize("code", NEW_RUN_CODES)
def test_new_run_codes_construct(code: str) -> None:
    assert code in RunError.CODES
    assert RunError(code).code == code


def test_unknown_codes_fail() -> None:
    with pytest.raises(ValueError):
        IndexingError("nope")
    with pytest.raises(ValueError):
        ExtractionError("nope")
    with pytest.raises(ValueError):
        RunError("nope")
    with pytest.raises(ValueError):
        RunError("page_render_limit")
    with pytest.raises(ValueError):
        IndexingError("selfrag_tokens_missing")
