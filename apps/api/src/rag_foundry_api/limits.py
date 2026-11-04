"""Pure ASGI upload size limit: Content-Length check, then a byte count while the body streams."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from rag_foundry_api.errors import error_response

MULTIPART_OVERHEAD = 64 * 1024

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

_UPLOAD_PATH = re.compile(r"^/api/v1/projects/[^/]+/documents(/[^/]+/versions)?/?$")


class _BodyTooLargeError(Exception):
    pass


class UploadSizeLimitMiddleware:
    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]], *, max_upload_bytes: int) -> None:
        self.app = app
        self.limit = max_upload_bytes + MULTIPART_OVERHEAD

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or not _UPLOAD_PATH.match(scope["path"]):
            await self.app(scope, receive, send)
            return

        async def reject() -> None:
            response = error_response(413, "upload_too_large", "upload exceeds the size limit")
            await response(scope, receive, send)

        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break  # the streaming count below still applies
                if declared > self.limit:
                    await reject()
                    return
                break

        seen = 0
        started = False

        async def counting_receive() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise _BodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLargeError:
            if started:
                raise
            await reject()
