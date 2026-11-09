"""Typed intake errors. Messages are fixed text: never file content, paths or secrets.

None of these subclass KeyError, LookupError or OSError, so callers cannot catch them by accident
through broad standard-library handlers.
"""


class IntakeError(Exception):
    """Base class for intake failures. `code` is a stable machine-readable string."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        self.message = message or code.replace("_", " ")
        super().__init__(self.message)


class UploadTooLargeError(IntakeError):
    def __init__(self, message: str | None = None) -> None:
        super().__init__("upload_too_large", message or "upload exceeds the size limit")


class EmptyUploadError(IntakeError):
    def __init__(self, message: str | None = None) -> None:
        super().__init__("empty_upload", message or "upload is empty")


class UnsupportedMediaTypeError(IntakeError):
    def __init__(self, message: str | None = None) -> None:
        super().__init__("unsupported_media_type", message or "media type is not supported")


class MediaTypeMismatchError(IntakeError):
    def __init__(self, message: str | None = None) -> None:
        super().__init__("media_type_mismatch", message or "content does not match the declared media type")


class InvalidTextEncodingError(IntakeError):
    def __init__(self, message: str | None = None) -> None:
        super().__init__("invalid_text_encoding", message or "text is not valid UTF-8")


class InvalidFilenameError(IntakeError):
    def __init__(self, message: str | None = None) -> None:
        super().__init__("invalid_filename", message or "filename is not valid")


class NotFoundError(IntakeError):
    """Unknown and out-of-scope IDs raise this same error with the same message."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__("not_found", message or "not found")


class ObjectStoreError(IntakeError):
    def __init__(self, message: str | None = None) -> None:
        super().__init__("object_store_error", message or "object store failure")


class ExtractionError(IntakeError):
    """Typed extraction failure. Codes: no_text_layer, pdf_encrypted, pdf_malformed, too_many_pages,
    text_too_long, integrity_mismatch, unsupported_extractor, html_too_complex (and alignment_failed, via
    ExtractionAlignmentError). Messages never carry content or paths."""

    CODES = (
        "no_text_layer",
        "pdf_encrypted",
        "pdf_malformed",
        "too_many_pages",
        "text_too_long",
        "integrity_mismatch",
        "unsupported_extractor",
        "html_too_complex",
    )

    def __init__(self, code: str, message: str | None = None) -> None:
        if code not in self.CODES:
            raise ValueError("unknown extraction error code")
        super().__init__(code, message)


class ExtractionAlignmentError(ExtractionError):
    """Text could not be aligned to the source; never guessed."""

    def __init__(self, message: str | None = None) -> None:
        IntakeError.__init__(self, "alignment_failed", message or "text could not be aligned to the source")


class AnchorError(IntakeError):
    """Anchor construction failure with a fixed, content-free message."""

    CODES = (
        "block_not_found",
        "range_out_of_bounds",
        "empty_range",
        "no_block_in_span",
        "invalid_namespace",
        "too_many_views",
    )

    def __init__(self, code: str) -> None:
        if code not in self.CODES:
            raise ValueError("unknown anchor error code")
        super().__init__(code)


class IndexingError(IntakeError):
    """Content-free index lifecycle and vector errors."""

    CODES = (
        "not_building",
        "not_ready",
        "dimension_mismatch",
        "invalid_vector",
        "duplicate_ordinal",
        "empty_index",
        "invalid_k",
        "invalid_page",
        "incompatible_embedding",
        "missing_embedding_dim",
        "empty_query",
        "query_too_long",
        "too_many_views",
        "embedding_count_mismatch",
        "no_query_terms",
        "lexical_corpus_too_large",
        "parent_span_out_of_bounds",
        "fts_unavailable",
        "fusion_too_few_branches",
        "fusion_duplicate_branch",
        "fusion_version_mismatch",
        "fusion_invalid_ranks",
        "fusion_duplicate_candidate",
        "fusion_candidate_conflict",
        "fusion_weights_mismatch",
        "selection_invalid_input",
        "selection_version_mismatch",
        "incompatible_sparse_encoder",
        "sparse_encoding_count_mismatch",
        "duplicate_sparse_vector",
        "sparse_unavailable",
        "selection_missing_vector",
        "selection_vectors_unavailable",
        "too_many_ids",
    )

    def __init__(self, code: str) -> None:
        if code not in self.CODES:
            raise ValueError("unknown indexing error code")
        super().__init__(code, code)


class QueryTransformError(IntakeError):
    """Content-free query transformation failures."""

    CODES = (
        "missing_model",
        "wrong_model_kind",
        "empty_query",
        "query_too_long",
        "too_many_choices",
        "invalid_model_output",
        "unknown_filter_field",
        "invalid_filter_value",
        "filter_scope_widening",
        "constraint_loss",
        "invalid_dependency",
        "too_many_queries",
    )

    def __init__(self, code: str) -> None:
        if code not in self.CODES:
            raise ValueError("unknown query transform error code")
        super().__init__(code, code)


class RunError(IntakeError):
    """Content-free run configuration and execution errors."""

    CODES = (
        "abstention_threshold_unset",
        "unsupported_technique",
        "index_version_required",
        "index_version_not_allowed",
        "non_local_model",
        "too_many_runs",
        "run_timeout",
    )

    def __init__(self, code: str) -> None:
        if code not in self.CODES:
            raise ValueError("unknown run error code")
        super().__init__(code, code)


class SettingsError(IntakeError):
    """Content-free runtime settings errors."""

    CODES = (
        "unknown_key",
        "scope_not_allowed",
        "invalid_value",
        "invalid_file",
        "invalid_override",
        "version_conflict",
        "store_unavailable",
    )

    def __init__(self, code: str) -> None:
        if code not in self.CODES:
            raise ValueError("unknown settings error code")
        super().__init__(code, code)


class PromptError(IntakeError):
    """Content-free prompt template store errors."""

    CODES = (
        "template_not_found",
        "no_active_template",
        "template_exists",
        "purpose_mismatch",
        "hash_mismatch",
        "invalid_template",
        "store_unavailable",
    )

    def __init__(self, code: str) -> None:
        if code not in self.CODES:
            raise ValueError("unknown prompt error code")
        super().__init__(code, code)


class StoreError(IntakeError):
    """Content-free persistence failures."""

    CODES = ("store_unavailable", "commit_failed")

    def __init__(self, code: str) -> None:
        if code not in self.CODES:
            raise ValueError("unknown store error code")
        super().__init__(code, code)
