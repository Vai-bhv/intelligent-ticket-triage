"""Limit prediction request bodies before FastAPI parses them."""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


MAX_BODY_BYTES = 64 * 1024


class PredictionBodyLimit:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"].rstrip("/") != "/v1/predict"
        ):
            await self.app(scope, receive, send)
            return

        too_large = JSONResponse({"detail": "Request body too large."}, status_code=413)
        for name, value in scope["headers"]:
            if name == b"content-length" and value.isdigit():
                length = value.lstrip(b"0") or b"0"
                limit = str(MAX_BODY_BYTES).encode("ascii")
                if (len(length), length) > (len(limit), limit):
                    await too_large(scope, receive, send)
                    return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                await too_large(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        async def replay():
            nonlocal body
            if body is None:
                return await receive()
            payload = bytes(body)
            body = None
            return {"type": "http.request", "body": payload, "more_body": False}

        await self.app(scope, replay, send)
