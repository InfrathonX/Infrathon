#!/usr/bin/env python3
"""Editable transparent Direct gateway supplied as the candidate starting point.

This module deliberately has no cross-request or cross-workflow scheduling state.
Every supported data-plane request is opened against the backend immediately.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web
from multidict import CIMultiDict


BACKEND_URL = os.environ["BACKEND_URL"].rstrip("/")
PORT = int(os.environ.get("PORT", "9000"))
MAX_REQUEST_BYTES = 64 * 1024 * 1024

# RFC 9110/9112 hop-by-hop fields, plus the widely used non-standard
# Proxy-Connection field.  Fields named by Connection are filtered separately.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def _text_header_pair(pair: tuple[bytes, bytes]) -> tuple[str, str]:
    name, value = pair
    return name.decode("latin-1"), value.decode("latin-1")


def _forward_headers(
    raw_headers: Iterable[tuple[bytes, bytes]],
    *,
    request_side: bool,
) -> CIMultiDict[str]:
    """Retain duplicate end-to-end headers and remove hop-by-hop fields."""

    pairs = [_text_header_pair(pair) for pair in raw_headers]
    connection_tokens = {
        token.strip().lower()
        for name, value in pairs
        if name.lower() == "connection"
        for token in value.split(",")
        if token.strip()
    }
    blocked = set(HOP_BY_HOP) | connection_tokens
    if request_side:
        # The client-facing Host names this gateway.  Let aiohttp generate the
        # correct Host for the fixed backend origin.
        blocked.add("host")

    forwarded: CIMultiDict[str] = CIMultiDict()
    for name, value in pairs:
        if name.lower() not in blocked:
            forwarded.add(name, value)
    return forwarded


def _backend_target(request: web.Request) -> str:
    suffix = request.path.removeprefix("/v1")
    target = f"{BACKEND_URL}{suffix}"
    if request.query_string:
        target = f"{target}?{request.query_string}"
    return target


async def _open_upstream(
    session: ClientSession,
    request: web.Request,
    body: bytes,
):
    return await session.request(
        request.method,
        _backend_target(request),
        data=body if body else None,
        headers=_forward_headers(request.raw_headers, request_side=True),
        allow_redirects=False,
        auto_decompress=False,
        skip_auto_headers={"Accept-Encoding", "Content-Type", "User-Agent"},
    )


async def proxy(request: web.Request) -> web.StreamResponse:
    """Forward one request exactly once without shared admission state."""

    body = await request.read()
    session: ClientSession = request.app["upstream_session"]
    upstream = None
    downstream = None
    try:
        upstream = await _open_upstream(session, request, body)
        response_headers = _forward_headers(
            upstream.raw_headers, request_side=False
        )
        downstream = web.StreamResponse(
            status=upstream.status,
            reason=upstream.reason,
            headers=response_headers,
        )
        await downstream.prepare(request)

        async for chunk in upstream.content.iter_any():
            await downstream.write(chunk)
        await downstream.write_eof()
        return downstream
    except asyncio.CancelledError:
        raise
    except (ConnectionResetError, BrokenPipeError):
        raise
    except ClientError as exc:
        if downstream is not None:
            # Response headers may already be on the wire.  Do not replace a
            # partial backend stream with a synthesized second response.
            raise ConnectionResetError(
                "backend response stream ended early"
            ) from exc
        return web.json_response(
            {"error": {"type": "backend_unavailable", "message": str(exc)}},
            status=502,
        )
    finally:
        if upstream is not None:
            upstream.close()


async def health(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "mode": "direct"})


async def release_program(request: web.Request) -> web.Response:
    # The starter stores no workflow state.  Acknowledge the optional trusted
    # lifecycle hint locally and never forward it to the model backend.
    await request.read()
    return web.Response(status=204)


async def upstream_session(app: web.Application):
    connector = TCPConnector(limit=0)
    timeout = ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=None)
    session = ClientSession(
        connector=connector,
        timeout=timeout,
        auto_decompress=False,
        skip_auto_headers={"Accept-Encoding", "User-Agent"},
    )
    app["upstream_session"] = session
    try:
        yield
    finally:
        await session.close()


def make_app() -> web.Application:
    # Preserve compressed request bytes.  aiohttp otherwise inflates inbound
    # Content-Encoding before the handler can forward the body.
    app = web.Application(
        client_max_size=MAX_REQUEST_BYTES,
        handler_args={
            "auto_decompress": False,
            # Abort the matching upstream request as soon as aiohttp observes
            # a downstream connection loss, including between SSE chunks.
            "handler_cancellation": True,
        },
    )
    app.cleanup_ctx.append(upstream_session)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", proxy)
    app.router.add_post("/v1/chat/completions", proxy)
    app.router.add_post("/programs/release", release_program)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="0.0.0.0", port=PORT, access_log=None)
