#!/usr/bin/env python3
"""Fail-closed CONNECT proxy and fixed Judge TCP forwarder.

Work is allowed to open TLS tunnels only to the Codex model and
authentication hosts below.  On hosts whose public egress is available only
through another HTTP proxy, ``--upstream-proxy-host`` and
``--upstream-proxy-port`` form a required pair.  Once that pair is configured,
an upstream failure is terminal for the request: this process never retries
the destination directly.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import signal
from collections.abc import Awaitable, Callable


MAX_HEADER_BYTES = 64 * 1024
ALLOWED = frozenset({"api.openai.com", "auth.openai.com", "chatgpt.com"})

ConnectionFactory = Callable[
    [str, int], Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]]
]


class UpstreamProxyError(ConnectionError):
    """The configured upstream HTTP proxy did not establish the tunnel."""


class ConnectDenied(PermissionError):
    """The Work request is not one of the explicitly permitted tunnels."""


def parse_target(value: str) -> tuple[str, int]:
    """Validate and canonicalize a Work CONNECT authority."""

    if value.count(":") != 1:
        raise ValueError("target must be host:port")
    host, raw_port = value.rsplit(":", 1)
    host = host.rstrip(".").lower()
    if not host or not raw_port.isdigit():
        raise ValueError("invalid target")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("literal IP targets are forbidden")
    port = int(raw_port)
    if host not in ALLOWED or port != 443:
        raise ConnectDenied("target is not allowlisted")
    return host, port


def validate_upstream_proxy(
    host: str | None, port: int | None
) -> tuple[str, int] | None:
    """Require an all-or-nothing, usable upstream proxy address."""

    if (host is None) != (port is None):
        raise ValueError("upstream proxy host and port must be configured together")
    if host is None:
        return None
    host = host.strip()
    if not host:
        raise ValueError("upstream proxy host must not be empty")
    assert port is not None
    if not 1 <= port <= 65535:
        raise ValueError("upstream proxy port must be between 1 and 65535")
    return host, port


async def copy_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(64 * 1024):
            writer.write(data)
            await writer.drain()
    except (BrokenPipeError, ConnectionError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.write_eof()


async def close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()


async def tunnel(
    a_r: asyncio.StreamReader,
    a_w: asyncio.StreamWriter,
    b_r: asyncio.StreamReader,
    b_w: asyncio.StreamWriter,
) -> None:
    try:
        await asyncio.gather(copy_stream(a_r, b_w), copy_stream(b_r, a_w))
    finally:
        await close_writer(a_w)
        await close_writer(b_w)


async def open_target_connection(
    host: str,
    port: int,
    *,
    upstream_proxy_host: str | None = None,
    upstream_proxy_port: int | None = None,
    connection_factory: ConnectionFactory = asyncio.open_connection,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open an allowed target directly or through the configured HTTP proxy.

    There is deliberately no exception handler which retries ``host:port``.
    In upstream mode every network connection goes only to the configured
    proxy, so connection failures and non-2xx CONNECT replies fail closed.
    """

    upstream = validate_upstream_proxy(upstream_proxy_host, upstream_proxy_port)
    if upstream is None:
        return await asyncio.wait_for(connection_factory(host, port), 15.0)

    upstream_reader: asyncio.StreamReader
    upstream_writer: asyncio.StreamWriter
    upstream_reader, upstream_writer = await asyncio.wait_for(
        connection_factory(*upstream), 15.0
    )
    authority = f"{host}:{port}"
    try:
        upstream_writer.write(
            (
                f"CONNECT {authority} HTTP/1.1\r\n"
                f"Host: {authority}\r\n"
                "Proxy-Connection: Keep-Alive\r\n"
                "\r\n"
            ).encode("ascii")
        )
        await upstream_writer.drain()
        response = await asyncio.wait_for(
            upstream_reader.readuntil(b"\r\n\r\n"), 15.0
        )
        if len(response) > MAX_HEADER_BYTES:
            raise UpstreamProxyError("upstream proxy response header is too large")
        status_parts = response.split(b"\r\n", 1)[0].decode("ascii", "strict").split()
        if (
            len(status_parts) < 2
            or status_parts[0] not in {"HTTP/1.0", "HTTP/1.1"}
            or not status_parts[1].isdigit()
            or int(status_parts[1]) // 100 != 2
        ):
            raise UpstreamProxyError("upstream proxy rejected CONNECT")
        return upstream_reader, upstream_writer
    except BaseException:
        await close_writer(upstream_writer)
        raise


async def handle_connect(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    upstream_proxy_host: str | None = None,
    upstream_proxy_port: int | None = None,
    *,
    connection_factory: ConnectionFactory = asyncio.open_connection,
) -> None:
    try:
        request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10.0)
        if len(request) > MAX_HEADER_BYTES:
            raise ValueError("header too large")
        request_parts = request.split(b"\r\n", 1)[0].decode("ascii", "strict").split()
        if len(request_parts) != 3:
            raise ValueError("malformed proxy request line")
        method, authority, protocol = request_parts
        if method == "GET" and authority == "/health":
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n"
                b"Connection: close\r\n\r\nok\n"
            )
            await writer.drain()
            return
        if method != "CONNECT" or not protocol.startswith("HTTP/1."):
            raise ConnectDenied("only CONNECT is supported")
        host, port = parse_target(authority)
        upstream_r, upstream_w = await open_target_connection(
            host,
            port,
            upstream_proxy_host=upstream_proxy_host,
            upstream_proxy_port=upstream_proxy_port,
            connection_factory=connection_factory,
        )
        writer.write(
            b"HTTP/1.1 200 Connection Established\r\n"
            b"Proxy-Agent: mlsysforge-qvg\r\n\r\n"
        )
        await writer.drain()
        await tunnel(reader, writer, upstream_r, upstream_w)
        return
    except asyncio.IncompleteReadError:
        pass
    except ConnectDenied:
        writer.write(
            b"HTTP/1.1 403 Forbidden\r\nContent-Length: 10\r\n"
            b"Connection: close\r\n\r\nforbidden\n"
        )
        with contextlib.suppress(Exception):
            await writer.drain()
    except Exception:
        writer.write(
            b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 12\r\n"
            b"Connection: close\r\n\r\nbad gateway\n"
        )
        with contextlib.suppress(Exception):
            await writer.drain()
    finally:
        if not writer.is_closing():
            await close_writer(writer)


async def handle_judge_forward(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    host: str,
    port: int,
) -> None:
    try:
        upstream_r, upstream_w = await asyncio.wait_for(
            asyncio.open_connection(host, port), 10.0
        )
    except Exception:
        await close_writer(writer)
        return
    await tunnel(reader, writer, upstream_r, upstream_w)


async def serve(args: argparse.Namespace) -> None:
    validate_upstream_proxy(args.upstream_proxy_host, args.upstream_proxy_port)

    async def connect_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await handle_connect(
            reader,
            writer,
            args.upstream_proxy_host,
            args.upstream_proxy_port,
        )

    connect_server = await asyncio.start_server(
        connect_handler, args.listen_host, args.proxy_port, limit=MAX_HEADER_BYTES + 1
    )

    async def judge_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await handle_judge_forward(reader, writer, args.judge_host, args.judge_port)

    judge_server = await asyncio.start_server(
        judge_handler, args.listen_host, args.judge_forward_port
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    async with connect_server, judge_server:
        await stop.wait()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--proxy-port", type=int, default=3128)
    parser.add_argument("--judge-forward-port", type=int, default=18384)
    parser.add_argument("--judge-host", default="host.docker.internal")
    parser.add_argument("--judge-port", type=int, required=True)
    parser.add_argument("--upstream-proxy-host")
    parser.add_argument("--upstream-proxy-port", type=int)
    args = parser.parse_args(argv)
    try:
        validate_upstream_proxy(args.upstream_proxy_host, args.upstream_proxy_port)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    asyncio.run(serve(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
