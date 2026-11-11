#!/usr/bin/env python3
"""Expose the fixed loopback HTTP proxy only on one Docker bridge address."""

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
    allowed_peer: str | None = None,
    target_host: str = TARGET_HOST,
    target_port: int = TARGET_PORT,
    connection_factory: ConnectionFactory = asyncio.open_connection,
) -> None:
    peer = writer.get_extra_info("peername")
    if (
        allowed_peer is not None
        and (not isinstance(peer, tuple) or not peer or peer[0] != allowed_peer)
    ):
        await close_writer(writer)
        return
    try:
        target_reader, target_writer = await asyncio.wait_for(
            connection_factory(target_host, target_port), timeout=10.0
        )
    except Exception:
        await close_writer(writer)
        return
    await tunnel(reader, writer, target_reader, target_writer)


async def serve(args: argparse.Namespace) -> None:
    async def restricted_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await handle_connection(
            reader,
            writer,
            allowed_peer=args.allowed_peer,
        )

    server = await asyncio.start_server(
        restricted_handler,
        args.listen_host,
        args.listen_port,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_number, stop.set)
    async with server:
        await stop.wait()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", type=int, default=DEFAULT_LISTEN_PORT)
    parser.add_argument("--allowed-peer", required=True)
    args = parser.parse_args(argv)
    try:
        listen_address = ipaddress.ip_address(args.listen_host)
    except ValueError:
        parser.error("listen host must be a Docker bridge IP address")
    if listen_address.is_unspecified or listen_address.is_loopback:
        parser.error("listen host must name one non-loopback bridge address")
    try:
        allowed_peer = ipaddress.ip_address(args.allowed_peer)
    except ValueError:
        parser.error("allowed peer must be one container IP address")
    if allowed_peer.is_unspecified or allowed_peer.is_loopback:
        parser.error("allowed peer must be one non-loopback container address")
    if not 1 <= args.listen_port <= 65535:
        parser.error("listen port is outside the TCP range")
    return args


def main() -> int:
    asyncio.run(serve(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
