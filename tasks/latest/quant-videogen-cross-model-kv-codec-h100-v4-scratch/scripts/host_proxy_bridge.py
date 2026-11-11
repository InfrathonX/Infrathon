#!/usr/bin/env python3
"""Expose the host's loopback HTTP proxy (127.0.0.1:18080) on one explicitly selected host address.

The local HTTP-to-SOCKS proxy listens on ``127.0.0.1:18080`` and therefore is
not reachable from Docker.  This process binds the caller-supplied Docker
bridge IP (normally on port 19080) and performs only byte-for-byte TCP
forwarding to that fixed loopback endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import signal
from collections.abc import Awaitable, Callable


DEFAULT_LISTEN_PORT = 19080
TARGET_HOST = "127.0.0.1"
TARGET_PORT = 18080

ConnectionFactory = Callable[
    [str, int], Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]]
]


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
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_reader: asyncio.StreamReader,
    target_writer: asyncio.StreamWriter,
) -> None:
    try:
        await asyncio.gather(
            copy_stream(client_reader, target_writer),
            copy_stream(target_reader, client_writer),
        )
    finally:
        await close_writer(target_writer)
        await close_writer(client_writer)


async def handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    target_host: str = TARGET_HOST,
    target_port: int = TARGET_PORT,
    connection_factory: ConnectionFactory = asyncio.open_connection,
) -> None:
    """Forward one TCP connection; the CLI always uses the fixed loopback proxy target."""

    try:
        target_reader, target_writer = await asyncio.wait_for(
            connection_factory(target_host, target_port), 10.0
        )
    except Exception:
        await close_writer(writer)
        return
    await tunnel(reader, writer, target_reader, target_writer)


async def serve(args: argparse.Namespace) -> None:
    server = await asyncio.start_server(
        handle_connection,
        args.listen_host,
        args.listen_port,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    async with server:
        await stop.wait()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--listen-host",
        required=True,
        help="Docker bridge IP to expose; do not use 0.0.0.0",
    )
    parser.add_argument("--listen-port", type=int, default=DEFAULT_LISTEN_PORT)
    args = parser.parse_args(argv)
    try:
        listen_address = ipaddress.ip_address(args.listen_host)
    except ValueError:
        parser.error("listen host must be a Docker bridge IP address")
    if listen_address.is_unspecified:
        parser.error("listen host must name one Docker bridge address")
    if not 1 <= args.listen_port <= 65535:
        parser.error("listen port must be between 1 and 65535")
    return args


def main() -> int:
    asyncio.run(serve(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
