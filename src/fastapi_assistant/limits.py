"""Request body size limit (HTTP 413), enforced before the body is parsed.

FastAPI reads and parses a JSON body before it resolves dependencies such as
authentication, so without a cap any caller could make the process buffer and parse
an arbitrarily large body. This middleware wraps ``receive``: it rejects a declared
``Content-Length`` over the limit on the first read, and counts bytes as they arrive
for chunked uploads that declare no length.

The rejection is raised as an ``HTTPException`` from inside the endpoint's read, so
the app's own exception handlers format it (``{"detail": ...}`` or the OpenAI error
envelope under ``/v1``) and the CORS middleware still decorates the response.
"""

from __future__ import annotations

from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.max_bytes <= 0:
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        limit = self.max_bytes
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            if declared is not None and declared > limit:
                raise HTTPException(status_code=413, detail=_detail(limit))
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413, detail=_detail(limit))
            return message

        await self.app(scope, limited_receive, send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _detail(limit: int) -> str:
    return f"Request body is larger than {limit} bytes"
