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
    )

    def __init__(self, code: str) -> None:
        if code not in self.CODES:
            raise ValueError("unknown indexing error code")
        super().__init__(code, code)
