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

_UPLOAD_PATH = re.compile(r"^/api/v1/projects/[^/]+/documents(?:/[^/]+/versions)?/?$")


class _BodyTooLargeError(Exception):
    pass


class UploadSizeLimitMiddleware:
    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]], *, max_upload_bytes: int) -> None:
        self.app = app
        self.limit = max_upload_bytes + MULTIPART_OVERHEAD

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        upload = _UPLOAD_PATH.match(scope["path"]) is not None
        limit = self.limit if upload else 1024 * 1024

        async def reject() -> None:
            code = "upload_too_large" if upload else "request_too_large"
            message = "upload exceeds the size limit" if upload else "request body exceeds the size limit"
            response = error_response(413, code, message)
            await response(scope, receive, send)

        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break  # the streaming count below still applies
                if declared > limit:
                    await reject()
                    return
                break

        # JSON parsing happens inside FastAPI's request.body(); an exception raised
        # by a streaming receive is translated there to a 400. Buffer bounded
        # non-upload bodies here so oversized chunked requests get our 413 envelope.
        if not upload:
            buffered: list[Message] = []
            seen = 0
            while True:
                message = await receive()
                if message["type"] != "http.request":
                    buffered.append(message)
                    break
                seen += len(message.get("body", b""))
                if seen > limit:
                    await reject()
                    return
                buffered.append(message)
                if not message.get("more_body", False):
                    break

            index = 0

            async def replay_receive() -> Message:
                nonlocal index
                if index < len(buffered):
                    message = buffered[index]
                    index += 1
                    return message
                return await receive()

            await self.app(scope, replay_receive, send)
            return

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
