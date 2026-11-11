#!/usr/bin/env python3
"""Small fail-closed network sidecar for the work session.

The proxy is a host-controlled sidecar. The work container is attached only
to an internal Docker network and has no route of its own; this sidecar,
which also sits on Docker's default bridge, is its only way out.  It offers:

* an HTTP CONNECT proxy.  By default (restricted mode) TLS tunnels are allowed
  solely to the model-authentication hosts (``ALLOWED_CONNECT_HOSTS`` on 443)
  and literal-IP targets are refused.  With ``--allow-public-egress`` (tasks
  whose JSON says ``internet: true``) it additionally tunnels to any host name
  or literal IP on port 443 or 80 whose resolved addresses are *all* publicly
  routable: loopback, RFC 1918 (including Docker bridges), link-local, CGNAT,
  multicast and reserved destinations are still refused, so a tunnel cannot
  reach the Docker host's bridge address, the Judge server or other
  containers.  The proxy resolves the name once, vets every address, and
  dials one of those vetted addresses itself, so a second DNS answer cannot
  redirect the tunnel (no DNS rebinding).  Refused targets get
  ``403 Forbidden`` in both modes;
* a separate TCP listener that forwards only to the run-scoped evaluation
  service;
* optionally, one raw TCP relay to an operator-hosted model API
  (``--model-forward``).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import ipaddress
import signal
import socket
from collections.abc import Awaitable


MAX_HEADER_BYTES = 64 * 1024
ALLOWED_CONNECT_HOSTS = frozenset(
    {
        "api.openai.com",
        "auth.openai.com",
        "chatgpt.com",
    }
)
# Ports a public-egress CONNECT may target (HTTPS and plain HTTP).
PUBLIC_EGRESS_PORTS = frozenset({80, 443})
UPSTREAM_CONNECT_TIMEOUT_SECONDS = 15.0
RESOLVE_TIMEOUT_SECONDS = 10.0
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")


def _literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def is_public_address(value: object) -> bool:
    """True only for a globally routable unicast address.

    ``ipaddress.is_global`` alone is not enough on every Python release (for
    example 224.0.0.0/4 multicast counts as global on 3.11), so the special
    ranges are refused explicitly, and an IPv4 address embedded in an IPv6
    one (mapped, 6to4, NAT64) must itself be public.
    """

    try:
        address = ipaddress.ip_address(value)  # type: ignore[arg-type]
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if getattr(address, "scope_id", None):
            return False
        embedded = address.ipv4_mapped or address.sixtofour
        if embedded is None and address in _NAT64_PREFIX:
            embedded = ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
        if embedded is not None and not is_public_address(embedded):
            return False
    return bool(
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
    )


def parse_connect_target(
    value: str,
    *,
    allow_public_egress: bool = False,
) -> tuple[str, int]:
    """Validate a CONNECT authority and return its canonical host and port.

    Restricted mode accepts only ``ALLOWED_CONNECT_HOSTS`` on 443.  Public
    mode also accepts any host name or literal IP on ``PUBLIC_EGRESS_PORTS``;
    a host name must then still pass :func:`resolve_public_addresses` before
    anything is dialled.
    """

    if value.count(":") != 1:
        raise ValueError("CONNECT authority must be host:port")
    host, raw_port = value.rsplit(":", 1)
    host = host.rstrip(".").lower()
    if not host or not raw_port.isdigit():
        raise ValueError("invalid CONNECT authority")
    port = int(raw_port)
    literal = _literal_ip(host)
    if host in ALLOWED_CONNECT_HOSTS and port == 443:
        return host, port
    if not allow_public_egress:
        if literal is not None:
            raise ValueError("literal IP CONNECT targets are prohibited")
        raise PermissionError("CONNECT target is not allowlisted")
    if port not in PUBLIC_EGRESS_PORTS:
        raise PermissionError("public CONNECT targets are limited to ports 80 and 443")
    if literal is not None and not is_public_address(literal):
        raise PermissionError("CONNECT target is not a public address")
    return host, port


async def resolve_public_addresses(host: str, port: int) -> list[str]:
    """Resolve ``host`` once; every address must be public, else refuse."""

    loop = asyncio.get_running_loop()
    infos = await asyncio.wait_for(
        loop.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        ),
        timeout=RESOLVE_TIMEOUT_SECONDS,
    )
    addresses: list[str] = []
    for family, _type, _proto, _canonname, sockaddr in infos:
        address = str(sockaddr[0])
        if family not in (socket.AF_INET, socket.AF_INET6) or not is_public_address(
            address
        ):
            raise PermissionError(f"{host} resolves to a non-public address")
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise PermissionError(f"{host} did not resolve to any address")
    return addresses


async def open_vetted_connection(
    addresses: list[str],
    port: int,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Dial the vetted literal addresses in order, never re-resolving."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + UPSTREAM_CONNECT_TIMEOUT_SECONDS
    last_error: BaseException | None = None
    for address in addresses:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            return await asyncio.wait_for(
                asyncio.open_connection(
                    address,
                    port,
                    flags=socket.AI_NUMERICHOST,
                ),
                timeout=remaining,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            last_error = exc
    raise ConnectionError("no vetted upstream address accepted the connection") from last_error


async def open_upstream(
    host: str,
    port: int,
    *,
    allow_public_egress: bool = False,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    if host in ALLOWED_CONNECT_HOSTS and port == 443:
        return await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=UPSTREAM_CONNECT_TIMEOUT_SECONDS,
        )
    if not allow_public_egress:
        raise PermissionError("CONNECT target is not allowlisted")
    literal = _literal_ip(host)
    if literal is not None:
        if not is_public_address(literal):
            raise PermissionError("CONNECT target is not a public address")
        addresses = [str(literal)]
    else:
        addresses = await resolve_public_addresses(host, port)
    return await open_vetted_connection(addresses, port)


async def copy_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
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
    upstream_reader: asyncio.StreamReader,
    upstream_writer: asyncio.StreamWriter,
) -> None:
    try:
        await asyncio.gather(
            copy_stream(client_reader, upstream_writer),
            copy_stream(upstream_reader, client_writer),
        )
    finally:
        await close_writer(upstream_writer)
        await close_writer(client_writer)


async def handle_evaluation(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    upstream_host: str,
    upstream_port: int,
) -> None:
    """Forward one connection to the configured evaluation endpoint."""
    try:
        upstream_reader, upstream_writer = await asyncio.wait_for(
            asyncio.open_connection(upstream_host, upstream_port), timeout=10.0
        )
        await tunnel(reader, writer, upstream_reader, upstream_writer)
    except Exception:
        if not writer.is_closing():
            await close_writer(writer)


async def handle_connect(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    allow_public_egress: bool = False,
) -> None:
    try:
        request = await asyncio.wait_for(
            reader.readuntil(b"\r\n\r\n"),
            timeout=10.0,
        )
        if len(request) > MAX_HEADER_BYTES:
            raise ValueError("request header is too large")
        first_line = request.split(b"\r\n", 1)[0].decode("ascii", "strict")
        parts = first_line.split()
        if len(parts) != 3:
            raise ValueError("malformed proxy request line")
        method, authority, protocol = parts
        if method == "GET" and authority == "/health":
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 3\r\n"
                b"Connection: close\r\n\r\nok\n"
            )
            await writer.drain()
            return
        if method != "CONNECT" or not protocol.startswith("HTTP/1."):
            raise PermissionError("only CONNECT is supported")
        host, port = parse_connect_target(
            authority,
            allow_public_egress=allow_public_egress,
        )
        upstream_reader, upstream_writer = await open_upstream(
            host,
            port,
            allow_public_egress=allow_public_egress,
        )
        writer.write(
            b"HTTP/1.1 200 Connection Established\r\n"
            b"Proxy-Agent: mlsysforge-restricted\r\n\r\n"
        )
        await writer.drain()
        await tunnel(reader, writer, upstream_reader, upstream_writer)
        return
    except asyncio.IncompleteReadError:
        pass
    except PermissionError:
        writer.write(
            b"HTTP/1.1 403 Forbidden\r\n"
            b"Content-Length: 10\r\n"
            b"Connection: close\r\n\r\nforbidden\n"
        )
        with contextlib.suppress(Exception):
            await writer.drain()
    except Exception:
        writer.write(
            b"HTTP/1.1 502 Bad Gateway\r\n"
            b"Content-Length: 12\r\n"
            b"Connection: close\r\n\r\nbad gateway\n"
        )
        with contextlib.suppress(Exception):
            await writer.drain()
    finally:
        if not writer.is_closing():
            await close_writer(writer)


async def serve(args: argparse.Namespace) -> None:
    connect_server = await asyncio.start_server(
        functools.partial(
            handle_connect,
            allow_public_egress=args.allow_public_egress,
        ),
        args.listen_host,
        args.proxy_port,
        limit=MAX_HEADER_BYTES + 1,
    )
    evaluation_server = await asyncio.start_server(
        lambda reader, writer: handle_evaluation(
            reader,
            writer,
            args.evaluation_upstream_host,
            args.evaluation_upstream_port,
        ),
        args.listen_host,
        args.evaluation_proxy_port,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_number, stop.set)
    if args.model_forward:
        listen_port, model_host, model_port = parse_model_forward(args.model_forward)
        # Raw relay to the operator's model API (an OpenAI-compatible server);
        # not a general egress, it accepts no other destination.
        await asyncio.start_server(
            lambda reader, writer: handle_model_forward(
                reader, writer, model_host, model_port
            ),
            args.listen_host,
            listen_port,
        )
    async with connect_server, evaluation_server:
        await stop.wait()


def parse_model_forward(value: str) -> tuple[int, str, int]:
    listen_port, host, port = value.rsplit(":", 2)
    return int(listen_port), host, int(port)


async def handle_model_forward(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    host: str,
    port: int,
) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=10.0,
        )
    except Exception:
        await close_writer(writer)
        return
    await tunnel(reader, writer, upstream_reader, upstream_writer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--proxy-port", type=int, default=3128)
    parser.add_argument("--evaluation-proxy-port", type=int, default=3130)
    parser.add_argument("--evaluation-upstream-host", required=True)
    parser.add_argument("--evaluation-upstream-port", type=int, required=True)
    parser.add_argument(
        "--allow-public-egress",
        action="store_true",
        help="also tunnel CONNECT to any host name or literal IP on port 443 or 80 "
        "whose resolved addresses are all publicly routable (private, loopback, "
        "link-local, CGNAT, multicast and reserved addresses stay refused); "
        "default: only the allowlisted model hosts on 443",
    )
    parser.add_argument("--model-forward", default="", help="LISTEN_PORT:HOST:PORT — raw TCP relay from LISTEN_PORT on this proxy to one fixed model-API endpoint (an OpenAI-compatible server the operator hosts); the only extra destination Work can reach")
    return parser.parse_args()


def main() -> int:
    asyncio.run(serve(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
