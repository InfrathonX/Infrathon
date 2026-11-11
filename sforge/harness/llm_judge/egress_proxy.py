#!/usr/bin/env python3
"""Fail-closed egress sidecar for integrity-review containers (stdlib only).

The review container sits on an internal Docker network whose only other
member is this sidecar.  The sidecar offers exactly two things:

* an HTTP CONNECT proxy that tunnels only to allowlisted ``host:443``
  authorities (for HTTPS model APIs such as chatgpt.com / api.openai.com);
* fixed raw TCP relays ``LISTEN_PORT -> HOST:PORT`` for plain-HTTP model
  endpoints the operator hosts.

Nothing else is reachable: literal-IP CONNECT targets, other hosts and other
ports are refused. The sidecar also sits on the default bridge (its egress),
so every listener only serves peers from the review network's subnet
(``--allow-peer``); containers on the bridge cannot borrow it.  Runs inside the review image with ``python3``; it is
bind-mounted read-only by sforge/harness/llm_judge/egress.py.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import signal

MAX_HEADER_BYTES = 64 * 1024


def parse_connect_target(value: str, allowed: frozenset[str]) -> tuple[str, int]:
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
        raise PermissionError("literal IP CONNECT targets are prohibited")
    port = int(raw_port)
    if host not in allowed or port != 443:
        raise PermissionError("CONNECT target is not allowlisted")
    return host, port


async def close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()


async def copy_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(64 * 1024):
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        if not writer.is_closing():
            await close_writer(writer)


async def tunnel(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    upstream_reader: asyncio.StreamReader,
    upstream_writer: asyncio.StreamWriter,
) -> None:
    await asyncio.gather(
        copy_stream(client_reader, upstream_writer),
        copy_stream(upstream_reader, client_writer),
    )


def peer_allowed(writer: asyncio.StreamWriter, peers) -> bool:
    if not peers:
        return True
    peer = writer.get_extra_info("peername")
    try:
        address = ipaddress.ip_address(peer[0])
    except (TypeError, ValueError, IndexError):
        return False
    return any(address in network for network in peers)


async def handle_connect(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    allowed: frozenset[str],
    peers=(),
) -> None:
    if not peer_allowed(writer, peers):
        await close_writer(writer)
        return
    try:
        request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10.0)
        if len(request) > MAX_HEADER_BYTES:
            raise ValueError("request header is too large")
        parts = request.split(b"\r\n", 1)[0].decode("ascii", "strict").split()
        if len(parts) != 3:
            raise ValueError("malformed proxy request line")
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
        host, port = parse_connect_target(authority, allowed)
        upstream_reader, upstream_writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=15.0
        )
        writer.write(
            b"HTTP/1.1 200 Connection Established\r\n"
            b"Proxy-Agent: sforge-review-egress\r\n\r\n"
        )
        await writer.drain()
        await tunnel(reader, writer, upstream_reader, upstream_writer)
        return
    except asyncio.IncompleteReadError:
        pass
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


async def handle_forward(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    host: str,
    port: int,
    peers=(),
) -> None:
    if not peer_allowed(writer, peers):
        await close_writer(writer)
        return
    try:
        upstream_reader, upstream_writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=10.0
        )
    except Exception:
        await close_writer(writer)
        return
    await tunnel(reader, writer, upstream_reader, upstream_writer)


def parse_forward(value: str) -> tuple[int, str, int]:
    listen_port, host, port = value.rsplit(":", 2)
    return int(listen_port), host, int(port)


async def serve(args: argparse.Namespace) -> None:
    allowed = frozenset(
        item.strip().rstrip(".").lower()
        for item in args.allow.split(",")
        if item.strip()
    )
    peers = tuple(
        ipaddress.ip_network(item.strip(), strict=False)
        for item in args.allow_peer
        if item.strip()
    )
    servers = [
        await asyncio.start_server(
            lambda r, w: handle_connect(r, w, allowed, peers),
            args.listen_host,
            args.proxy_port,
            limit=MAX_HEADER_BYTES + 1,
        )
    ]
    for forward in args.forward:
        listen_port, host, port = parse_forward(forward)
        servers.append(
            await asyncio.start_server(
                lambda r, w, h=host, p=port: handle_forward(r, w, h, p, peers),
                args.listen_host,
                listen_port,
            )
        )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_number, stop.set)
    try:
        await stop.wait()
    finally:
        for server in servers:
            server.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--proxy-port", type=int, default=3128)
    parser.add_argument("--allow", default="", help="comma-separated CONNECT hosts (port 443)")
    parser.add_argument(
        "--allow-peer",
        action="append",
        default=[],
        help="CIDR whose clients may connect (repeatable; default: anyone)",
    )
    parser.add_argument(
        "--forward",
        action="append",
        default=[],
        help="LISTEN_PORT:HOST:PORT raw TCP relay to one fixed endpoint (repeatable)",
    )
    asyncio.run(serve(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
