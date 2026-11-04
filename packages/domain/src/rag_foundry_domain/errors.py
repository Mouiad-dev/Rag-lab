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
