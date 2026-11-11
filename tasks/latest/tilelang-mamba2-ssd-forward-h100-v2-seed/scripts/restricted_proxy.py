#!/usr/bin/env python3
"""Small fail-closed CONNECT proxy plus fixed Judge TCP forwarder.

The proxy is a host-controlled sidecar. The Work container is attached
only to an internal Docker network and can reach this sidecar, not the public
internet directly. CONNECT has two modes:

* default (allowlist): TLS tunnels are allowed solely to the
  model-authentication hosts in ALLOWED_CONNECT_HOSTS on port 443, and literal
  IP targets are refused.
* ``--allow-public-egress``: in addition, a CONNECT to any hostname or literal
  IP on port 80 or 443 is tunnelled when the proxy's own resolution of it
  yields only globally routable unicast addresses (``ipaddress.is_global``,
  not multicast).  A target with any loopback, private (RFC 1918, so also the
  Docker bridges), link-local, CGNAT, multicast or reserved answer is refused.
  The proxy then dials one of those vetted addresses itself rather than
  re-resolving the name, so a second DNS answer cannot rebind the tunnel onto
  an internal address.

Every policy refusal is answered with 403.  The Judge forwarder (and the
optional model-API relay) has a fixed destination in both modes.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
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
# --allow-public-egress: the only ports a vetted public target may use.
PUBLIC_EGRESS_PORTS = frozenset({80, 443})
# IPv6 prefixes that embed an IPv4 address (NAT64); refused outright because
# the embedded address is what is finally reached.
_IPV4_EMBEDDING_IPV6_NETWORKS = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
)


def _is_hostname(host: str) -> bool:
    """Return whether ``host`` is a syntactically plain DNS name."""

    if len(host) > 253:
        return False
    for label in host.split("."):
        if not 1 <= len(label) <= 63:
            return False
        if label[0] == "-" or label[-1] == "-":
            return False
        if any(not (c.isascii() and (c.isalnum() or c in "-_")) for c in label):
            return False
    return True


def parse_connect_target(
    value: str,
    *,
    allow_public_egress: bool = False,
) -> tuple[str, int]:
    """Validate a CONNECT authority and return its canonical host and port.

    In the default mode only ALLOWED_CONNECT_HOSTS on port 443 pass.  With
    ``allow_public_egress`` any hostname or literal IP (IPv6 in brackets) on a
    PUBLIC_EGRESS_PORTS port passes this syntactic check; every address it
    resolves to must still be vetted by resolve_public_addresses().
    """

    bracketed = allow_public_egress and value.startswith("[")
    if bracketed:
        host, separator, raw_port = value[1:].partition("]:")
        if not separator:
            raise ValueError("invalid CONNECT authority")
    else:
        if value.count(":") != 1:
            raise ValueError("CONNECT authority must be host:port")
        host, raw_port = value.rsplit(":", 1)
    host = host.rstrip(".").lower()
    if not host or not raw_port.isdigit():
        raise ValueError("invalid CONNECT authority")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    port = int(raw_port)
    if not allow_public_egress:
        if literal is not None:
            raise ValueError("literal IP CONNECT targets are prohibited")
        if host not in ALLOWED_CONNECT_HOSTS or port != 443:
            raise PermissionError("CONNECT target is not allowlisted")
        return host, port
    if bracketed and not isinstance(literal, ipaddress.IPv6Address):
        raise ValueError("only IPv6 literals may be bracketed")
    if literal is None and not _is_hostname(host):
        raise ValueError("invalid CONNECT host")
    if port not in PUBLIC_EGRESS_PORTS:
        raise PermissionError("CONNECT port is not allowed")
    return host, port


def is_public_address(address: str) -> bool:
    """Return whether ``address`` is a globally routable unicast IP address.

    ``is_global`` already rejects loopback, private (including the Docker
    bridges), link-local, CGNAT, unspecified and reserved ranges.  Multicast is
    rejected explicitly because ``is_global`` admits it, and IPv6 forms that
    embed an IPv4 address (mapped, 6to4, Teredo, NAT64) are rejected outright.
    """

    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if ip.is_multicast or not ip.is_global:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if (
            ip.ipv4_mapped is not None
            or ip.sixtofour is not None
            or ip.teredo is not None
            or any(ip in network for network in _IPV4_EMBEDDING_IPV6_NETWORKS)
        ):
            return False
    return True


async def resolve_public_addresses(host: str, port: int) -> list[str]:
    """Resolve ``host`` once and return its addresses if all are public.

    A literal IP resolves to itself.  A single non-public answer refuses the
    whole target (PermissionError), so a name cannot slip an internal address
    in among public ones.  IPv4 answers come first because the sidecar's
    Docker bridge normally has no IPv6 route.
    """

    loop = asyncio.get_running_loop()
    infos = await asyncio.wait_for(
        loop.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        ),
        timeout=10.0,
    )
    addresses: list[str] = []
    for family, _type, _proto, _canonname, sockaddr in infos:
        address = str(sockaddr[0])
        if family not in (socket.AF_INET, socket.AF_INET6):
            raise PermissionError("CONNECT target resolves to a non-IP address")
        if not is_public_address(address):
            raise PermissionError("CONNECT target resolves to a non-public address")
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise PermissionError("CONNECT target has no address")
    addresses.sort(key=lambda address: ipaddress.ip_address(address).version)
    return addresses


async def open_upstream(
    host: str,
    port: int,
    *,
    allow_public_egress: bool = False,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open the upstream TCP connection for an already validated target.

    Allowlisted model-auth hosts are dialled by name exactly as in the default
    mode.  Any other target (public-egress mode only) is dialled through the
    addresses vetted by resolve_public_addresses(), never by re-resolving the
    name.
    """

    if not allow_public_egress or (host in ALLOWED_CONNECT_HOSTS and port == 443):
        return await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=15.0,
        )
    addresses = await resolve_public_addresses(host, port)
    last_error: Exception = ConnectionError("no vetted address")
    for address in addresses:
        try:
            return await asyncio.wait_for(
                asyncio.open_connection(address, port),
                timeout=10.0,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            last_error = exc
    raise last_error


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


async def handle_judge_forward(
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


async def serve(args: argparse.Namespace) -> None:
    connect_server = await asyncio.start_server(
        lambda reader, writer: handle_connect(
            reader,
            writer,
            allow_public_egress=args.allow_public_egress,
        ),
        args.listen_host,
        args.proxy_port,
        limit=MAX_HEADER_BYTES + 1,
    )

    async def judge_handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        await handle_judge_forward(
            reader,
            writer,
            args.judge_host,
            args.judge_port,
        )

    judge_server = await asyncio.start_server(
        judge_handler,
        args.listen_host,
        args.judge_forward_port,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_number, stop.set)
    if args.model_forward:
        listen_port, model_host, model_port = parse_model_forward(args.model_forward)
        # Same raw relay as the Judge route, pointed at the operator's model
        # API; it is not a general egress and accepts no other destination.
        await asyncio.start_server(
            lambda reader, writer: handle_judge_forward(
                reader, writer, model_host, model_port
            ),
            args.listen_host,
            listen_port,
        )
    async with connect_server, judge_server:
        await stop.wait()


def parse_model_forward(value: str) -> tuple[int, str, int]:
    listen_port, host, port = value.rsplit(":", 2)
    return int(listen_port), host, int(port)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--proxy-port", type=int, default=3128)
    parser.add_argument("--judge-forward-port", type=int, default=18284)
    parser.add_argument("--judge-host", default="host.docker.internal")
    parser.add_argument("--judge-port", type=int, required=True)
    parser.add_argument(
        "--allow-public-egress",
        action="store_true",
        help=(
            "also tunnel CONNECT to any host or literal IP on port 80/443 whose "
            "resolved addresses are all globally routable; the proxy dials the "
            "vetted address itself (default: model-auth allowlist only)"
        ),
    )
    parser.add_argument("--model-forward", default="", help="LISTEN_PORT:HOST:PORT — raw TCP relay from LISTEN_PORT on this proxy to one fixed model-API endpoint (an OpenAI-compatible server the operator hosts); the only extra destination Work can reach")
    return parser.parse_args()


def main() -> int:
    asyncio.run(serve(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
