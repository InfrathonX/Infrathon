#!/usr/bin/env python3
"""Forward only the HTTPS hosts needed to authenticate and run Codex.

The host on this machine reaches the public network through a loopback HTTP
proxy. Work receives this restricted proxy instead: normal client traffic to
GitHub and other research/source hosts is rejected, while Codex authentication
remains reachable. Node and Codex are already pinned in the Work image.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import signal


MAX_HEADER_BYTES = 64 * 1024
ALLOWED_HOSTS = frozenset(
    {
        "api.openai.com",
        "auth.openai.com",
        "chatgpt.com",
    }
)


def parse_authority(value: str) -> tuple[str, int]:
    if value.count(":") != 1:
        raise ValueError("CONNECT authority must be host:port")
    host, raw_port = value.rsplit(":", 1)
    host = host.rstrip(".").lower()
    if not host or not raw_port.isdigit():
        raise ValueError("invalid CONNECT authority")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise PermissionError("literal IP targets are forbidden")
    port = int(raw_port)
    if port != 443 or host not in ALLOWED_HOSTS:
        raise PermissionError("target is not allowlisted")
    return host, port


async def close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()


async def copy_stream(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        while data := await reader.read(64 * 1024):
            writer.write(data)
            await writer.drain()
    except (BrokenPipeError, ConnectionError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.write_eof()


async def open_upstream_tunnel(
    host: str,
    port: int,
    upstream_host: str,
    upstream_port: int,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(upstream_host, upstream_port), 15.0
    )
    authority = f"{host}:{port}"
    try:
        writer.write(
            (
                f"CONNECT {authority} HTTP/1.1\r\n"
                f"Host: {authority}\r\n"
                "Proxy-Connection: Keep-Alive\r\n\r\n"
            ).encode("ascii")
        )
        await writer.drain()
        response = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 15.0)
        if len(response) > MAX_HEADER_BYTES:
            raise ConnectionError("upstream response header is too large")
        status = response.split(b"\r\n", 1)[0].decode("ascii", "strict").split()
        if (
            len(status) < 2
            or status[0] not in {"HTTP/1.0", "HTTP/1.1"}
            or not status[1].isdigit()
            or int(status[1]) // 100 != 2
        ):
            raise ConnectionError("upstream proxy rejected CONNECT")
        return reader, writer
    except BaseException:
        await close_writer(writer)
        raise


async def handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    upstream_host: str,
    upstream_port: int,
) -> None:
    try:
        request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10.0)
        if len(request) > MAX_HEADER_BYTES:
            raise ValueError("request header is too large")
        parts = request.split(b"\r\n", 1)[0].decode("ascii", "strict").split()
        if len(parts) != 3:
            raise ValueError("malformed proxy request")
        method, authority, protocol = parts
        if method == "GET" and authority == "/health":
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n"
                b"Connection: close\r\n\r\nok\n"
            )
            await writer.drain()
            return
        if method != "CONNECT" or not protocol.startswith("HTTP/1."):
            raise PermissionError("only CONNECT is supported")
        host, port = parse_authority(authority)
        upstream_reader, upstream_writer = await open_upstream_tunnel(
            host, port, upstream_host, upstream_port
        )
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        await asyncio.gather(
            copy_stream(reader, upstream_writer),
            copy_stream(upstream_reader, writer),
        )
        await close_writer(upstream_writer)
    except PermissionError:
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


async def serve(args: argparse.Namespace) -> None:
    async def handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await handle_connection(
            reader,
            writer,
            args.upstream_host,
            args.upstream_port,
        )

    server = await asyncio.start_server(
        handler,
        args.listen_host,
        args.listen_port,
        limit=MAX_HEADER_BYTES + 1,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    async with server:
        await stop.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", required=True, type=int)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", default=18080, type=int)
    args = parser.parse_args()
    if not 1 <= args.listen_port <= 65535:
        parser.error("listen port must be between 1 and 65535")
    if not 1 <= args.upstream_port <= 65535:
        parser.error("upstream port must be between 1 and 65535")
    return args


def main() -> int:
    asyncio.run(serve(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
