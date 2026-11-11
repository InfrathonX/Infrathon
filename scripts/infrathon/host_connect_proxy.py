#!/usr/bin/env python3
"""Loopback-only CONNECT proxy for the concurrent-serving task.

The concurrent-serving run script expects a host proxy at the fixed address
127.0.0.1:18080. Its restricted Work-side proxy reaches this address through
scripts/host_proxy_bridge.py and sends only CONNECT requests, after applying its
own allowlist. A host that reaches the internet directly can run this proxy:
it tunnels `CONNECT host:port` to the destination, answers 405 to anything
else, binds loopback only and never closes an idle tunnel (the agent client can
stay silent for a long time on an open connection).

    python3 scripts/infrathon/host_connect_proxy.py [--port 18080]

Keep it running for the whole run, for example in tmux. If the host itself
reaches the internet only through an HTTP proxy, point the task at that proxy
on 127.0.0.1:18080 instead.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import time

HEADER_TIMEOUT_S = 15
CONNECT_TIMEOUT_S = 15


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(64 * 1024):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.write_eof()


async def reply_and_close(writer: asyncio.StreamWriter, status: bytes) -> None:
    writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
    with contextlib.suppress(Exception):
        await writer.drain()
    writer.close()


async def handle(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
    try:
        head = await asyncio.wait_for(client_reader.readuntil(b"\r\n\r\n"), HEADER_TIMEOUT_S)
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        client_writer.close()
        return
    parts = head.split(b"\r\n", 1)[0].decode("latin-1").split()
    if len(parts) != 3 or parts[0] != "CONNECT" or ":" not in parts[1]:
        await reply_and_close(client_writer, b"405 Method Not Allowed")
        return
    host, _, port = parts[1].rpartition(":")
    host = host.strip("[]")
    try:
        upstream_reader, upstream_writer = await asyncio.wait_for(
            asyncio.open_connection(host, int(port)), CONNECT_TIMEOUT_S
        )
    except Exception as exc:  # noqa: BLE001 - any connect failure is a 502
        log(f"CONNECT {parts[1]} failed: {exc!r}")
        await reply_and_close(client_writer, b"502 Bad Gateway")
        return
    client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await client_writer.drain()
    try:
        await asyncio.gather(
            pipe(client_reader, upstream_writer), pipe(upstream_reader, client_writer)
        )
    finally:
        for writer in (upstream_writer, client_writer):
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    server = await asyncio.start_server(handle, "127.0.0.1", args.port, limit=64 * 1024)
    log(f"listening on 127.0.0.1:{args.port} (CONNECT only)")
    async with server:
        await server.serve_forever()
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
