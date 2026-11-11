#!/usr/bin/env python3
"""Expose an already-hosted Unix-socket vLLM as a plain HTTP origin.

This is a host-deployment adapter, not part of the MLSys Forge scoring runtime.
The task receives only the resulting URL; GPU placement, container names, and
the Unix socket remain outside the Judge/worker contract.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import signal
import socket
import stat
from typing import Any


BUFFER_BYTES = 1024 * 1024


def require_socket(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    mode = resolved.lstat().st_mode
    if path.is_symlink() or not stat.S_ISSOCK(mode):
        raise ValueError(f"upstream is not a real Unix socket: {path}")
    return resolved


async def relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(BUFFER_BYTES):
            writer.write(chunk)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
            await writer.drain()
    except (BrokenPipeError, ConnectionError, asyncio.CancelledError):
        pass


async def close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except (BrokenPipeError, ConnectionError):
        pass


async def handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    *,
    upstream_socket: Path,
    slots: asyncio.Semaphore,
) -> None:
    async with slots:
        upstream_writer: asyncio.StreamWriter | None = None
        try:
            upstream_reader, upstream_writer = await asyncio.open_unix_connection(
                str(upstream_socket)
            )
            directions = {
                asyncio.create_task(relay(client_reader, upstream_writer)),
                asyncio.create_task(relay(upstream_reader, client_writer)),
            }
            done, pending = await asyncio.wait(
                directions, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                await task
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except (ConnectionError, OSError):
            pass
        finally:
            if upstream_writer is not None:
                await close_writer(upstream_writer)
            await close_writer(client_writer)


async def serve(args: argparse.Namespace) -> None:
    upstream = require_socket(args.unix_socket)
    slots = asyncio.Semaphore(args.max_connections)
    server = await asyncio.start_server(
        lambda reader, writer: handle_client(
            reader,
            writer,
            upstream_socket=upstream,
            slots=slots,
        ),
        host=args.listen_host,
        port=args.listen_port,
        family=socket.AF_INET,
        reuse_address=True,
        backlog=min(args.max_connections, 4096),
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)

    sockets = server.sockets or []
    addresses: list[Any] = [item.getsockname() for item in sockets]
    print(
        json.dumps(
            {
                "status": "ready",
                "listen": addresses,
                "upstream_transport": "unix",
                "max_connections": args.max_connections,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    async with server:
        await stop.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unix-socket", required=True, type=Path)
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", required=True, type=int)
    parser.add_argument("--max-connections", type=int, default=512)
    args = parser.parse_args()
    if not 1 <= args.listen_port <= 65535:
        parser.error("--listen-port must be between 1 and 65535")
    if not 1 <= args.max_connections <= 4096:
        parser.error("--max-connections must be between 1 and 4096")
    return args


def main() -> int:
    asyncio.run(serve(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
