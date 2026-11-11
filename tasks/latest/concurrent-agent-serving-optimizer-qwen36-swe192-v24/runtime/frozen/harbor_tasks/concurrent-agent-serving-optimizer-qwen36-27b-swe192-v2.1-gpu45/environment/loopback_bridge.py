#!/usr/bin/env python3
"""Credentialed streaming bridge for the candidate's loopback model URL.

The credential is derived by the root entrypoint and stored in a file that is
readable by the dedicated bridge group, but not by candidate UID 1000.  The
bridge overwrites the private transport header on every forwarded request so a
candidate can neither learn nor choose the credential sent to model-api.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import os
import secrets
import stat
from pathlib import Path
from typing import Any, AsyncIterator

from aiohttp import ClientConnectionError, ClientSession, ClientTimeout, TCPConnector, web


TRANSPORT_AUTH_HEADER = "X-KVBench-Transport-Auth"
TRANSPORT_AUTH_CONTEXT = b"kvbench-model-api-transport-v1"
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}


def derive_transport_credential(admin_token: bytes) -> str:
    if not admin_token:
        raise ValueError("admin token is empty")
    return hmac.new(
        admin_token,
        TRANSPORT_AUTH_CONTEXT,
        hashlib.sha256,
    ).hexdigest()


def provision_credential(
    admin_token_file: Path,
    credential_file: Path,
    *,
    owner_uid: int,
    owner_gid: int,
) -> None:
    """Atomically create a bridge-readable, candidate-inaccessible secret."""
    token = admin_token_file.read_bytes().strip()
    credential = derive_transport_credential(token).encode() + b"\n"
    credential_file.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = credential_file.with_name(
        f".{credential_file.name}.{os.getpid()}.{secrets.token_hex(8)}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o440)
    try:
        os.write(descriptor, credential)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o440)
        os.fchown(descriptor, owner_uid, owner_gid)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, credential_file)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_credential(path: Path) -> str:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o007:
        raise RuntimeError("bridge credential must be a regular non-world-readable file")
    credential = path.read_text().strip()
    if len(credential) != 64 or any(character not in "0123456789abcdef" for character in credential):
        raise RuntimeError("bridge credential has an invalid encoding")
    return credential


def filtered_headers(headers: Any, *, request_headers: bool = False) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered in HOP_HEADERS:
            continue
        if request_headers and lowered == TRANSPORT_AUTH_HEADER.lower():
            continue
        result[name] = value
    return result


async def iter_request_body(request: web.Request) -> AsyncIterator[bytes]:
    async for chunk in request.content.iter_any():
        yield chunk


async def proxy(request: web.Request) -> web.StreamResponse:
    app = request.app
    headers = filtered_headers(request.headers, request_headers=True)
    # Assignment after filtering is intentional: candidate input can never
    # override the credential held only by UID 900.
    headers[TRANSPORT_AUTH_HEADER] = app["transport_credential"]
    body = iter_request_body(request) if request.can_read_body else None
    downstream: web.StreamResponse | None = None
    # Scheduler metrics are control-plane observations.  Keep them off the
    # connector that can be fully occupied by 192 long-lived generation
    # streams, otherwise the official ThunderAgent metrics loop could lose the
    # very pressure signal used for admission.
    client = (
        app["control_client"]
        if request.path == "/v1/metrics"
        else app["data_client"]
        if request.path.startswith("/v1/")
        else app["control_client"]
    )
    try:
        async with client.request(
            request.method,
            f"{app['upstream_origin']}{request.path_qs}",
            headers=headers,
            data=body,
            allow_redirects=False,
        ) as upstream:
            downstream = web.StreamResponse(
                status=upstream.status,
                reason=upstream.reason,
                headers=filtered_headers(upstream.headers),
            )
            await downstream.prepare(request)
            # Preserve response entity bytes and backpressure.  There is no
            # full-response buffer and no retry path before or after prepare.
            async for chunk in upstream.content.iter_any():
                await downstream.write(chunk)
            await downstream.write_eof()
            return downstream
    except asyncio.CancelledError:
        raise
    except (OSError, asyncio.TimeoutError, ClientConnectionError):
        # Once response headers are committed, close the stream on an
        # upstream failure.  Sending a second buffered error response would
        # corrupt the byte-for-byte proxy contract.
        if downstream is not None or not request.transport or request.transport.is_closing():
            raise
        return web.json_response(
            {"error": {"message": "trusted model bridge unavailable", "type": "bridge_error"}},
            status=502,
        )


async def on_startup(app: web.Application) -> None:
    app["data_client"] = ClientSession(
        # Match the frozen 192-sequence Qwen data plane.  Each client request maps
        # to exactly one model-api request; force-close avoids stale stream
        # transports without introducing a retry.
        connector=TCPConnector(
            limit=192,
            limit_per_host=0,
            force_close=True,
        ),
        timeout=ClientTimeout(total=None, connect=30, sock_read=None),
        auto_decompress=False,
    )
    # Keep health/admin/scheduler-metrics traffic out of the 192 long-lived
    # generation streams.
    app["control_client"] = ClientSession(
        connector=TCPConnector(
            limit=8,
            limit_per_host=0,
            force_close=True,
        ),
        timeout=ClientTimeout(total=30, connect=10, sock_read=30),
        auto_decompress=False,
    )


async def on_cleanup(app: web.Application) -> None:
    await app["data_client"].close()
    await app["control_client"].close()


def make_app(*, upstream_origin: str, transport_credential: str) -> web.Application:
    if not upstream_origin.startswith("http://"):
        raise ValueError("bridge upstream must use internal plain HTTP")
    app = web.Application(
        client_max_size=128 * 1024**2,
        # Abort the matching model-api request as soon as the candidate-side
        # controller closes this hop at a trusted request deadline.
        handler_args={"handler_cancellation": True},
    )
    app["upstream_origin"] = upstream_origin.rstrip("/")
    app["transport_credential"] = transport_credential
    app.router.add_route("*", "/{path_info:.*}", proxy)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    provision = subcommands.add_parser("provision")
    provision.add_argument("--admin-token-file", type=Path, required=True)
    provision.add_argument("--credential-file", type=Path, required=True)
    provision.add_argument("--owner-uid", type=int, required=True)
    provision.add_argument("--owner-gid", type=int, required=True)
    serve = subcommands.add_parser("serve")
    serve.add_argument("--credential-file", type=Path, required=True)
    serve.add_argument("--upstream-origin", default="http://model-api:8100")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "provision":
        provision_credential(
            args.admin_token_file,
            args.credential_file,
            owner_uid=args.owner_uid,
            owner_gid=args.owner_gid,
        )
        return
    credential = load_credential(args.credential_file)
    web.run_app(
        make_app(
            upstream_origin=args.upstream_origin,
            transport_credential=credential,
        ),
        host=args.host,
        port=args.port,
        access_log=None,
    )


if __name__ == "__main__":
    main()
