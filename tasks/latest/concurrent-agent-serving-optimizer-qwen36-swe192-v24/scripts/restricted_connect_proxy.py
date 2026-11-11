#!/usr/bin/env python3
"""Small fail-closed CONNECT proxy plus fixed Judge TCP forwarder.

The proxy is a host-controlled sidecar. The Work container is attached
only to an internal Docker network and can reach this sidecar, not the public
internet directly. By default TLS tunnels are allowed solely to the
model-authentication hosts. With ``--allow-public-egress`` (the launcher passes
it when the task JSON sets ``"internet": true``) a CONNECT to any public host
on port 80 or 443 is also accepted: the name is resolved here, every resolved
address must be globally routable (no loopback, RFC 1918, Docker bridge,
link-local, CGNAT, multicast or reserved space), and the tunnel goes to one of
those vetted addresses without a second resolution. Refusals are 403.
When public egress requires an upstream HTTP proxy, the configured upstream is
mandatory and failures never fall back to a direct destination connection.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import re
import signal
import socket
from collections.abc import Awaitable, Callable, Sequence


MAX_HEADER_BYTES = 64 * 1024
ALLOWED_CONNECT_HOSTS = frozenset(
    {
        "api.openai.com",
        "auth.openai.com",
        "chatgpt.com",
    }
)

# Destination ports a public-egress CONNECT may use (HTTPS and plain HTTP).
PUBLIC_EGRESS_PORTS = frozenset({80, 443})
_HOST_LABEL = r"[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?"
_HOSTNAME = re.compile(rf"(?=.{{1,253}}$){_HOST_LABEL}(?:\.{_HOST_LABEL})*")

ConnectionFactory = Callable[
    [str, int], Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]]
]
Resolver = Callable[..., Awaitable[list[tuple]]]


class UpstreamProxyError(ConnectionError):
    """The configured upstream proxy did not establish the TLS tunnel."""


def validate_upstream_proxy(
    host: str | None, port: int | None
) -> tuple[str, int] | None:
    if (host is None) != (port is None):
        raise ValueError("upstream proxy host and port must be configured together")
    if host is None:
        return None
    host = host.strip()
    if not host:
        raise ValueError("upstream proxy host must not be empty")
    assert port is not None
    if not 1 <= port <= 65535:
        raise ValueError("upstream proxy port is outside the TCP range")
    return host, port


def parse_connect_target(value: str) -> tuple[str, int]:
    """Validate a CONNECT authority and return its canonical host and port."""

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
        raise ValueError("literal IP CONNECT targets are prohibited")
    port = int(raw_port)
    if host not in ALLOWED_CONNECT_HOSTS or port != 443:
        raise PermissionError("CONNECT target is not allowlisted")
    return host, port


def split_connect_authority(value: str) -> tuple[str, int]:
    """Split ``host:port`` or ``[ipv6]:port`` into a canonical host and port."""

    if value.startswith("["):
        host, separator, raw_port = value[1:].partition("]:")
        if not separator or not host:
            raise ValueError("invalid bracketed CONNECT authority")
        try:
            ipaddress.IPv6Address(host)
        except ValueError as exc:
            raise ValueError("bracketed CONNECT host must be an IPv6 address") from exc
    else:
        if value.count(":") != 1:
            raise ValueError("CONNECT authority must be host:port")
        host, raw_port = value.rsplit(":", 1)
    host = host.rstrip(".").lower()
    if not host or not raw_port.isdigit():
        raise ValueError("invalid CONNECT authority")
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise ValueError("CONNECT port is outside the TCP range")
    return host, port


def is_public_address(address: str) -> bool:
    """True only for a globally routable unicast address."""

    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    if parsed.version == 6 and parsed.ipv4_mapped is not None:
        # IPv4-mapped forms are refused outright: newer Pythons judge them by the embedded IPv4 address.
        return False
    return (
        parsed.is_global
        and not parsed.is_private
        and not parsed.is_loopback
        and not parsed.is_link_local
        and not parsed.is_multicast
        and not parsed.is_reserved
        and not parsed.is_unspecified
    )


async def resolve_public_addresses(
    host: str,
    port: int,
    *,
    resolver: Resolver | None = None,
) -> tuple[str, ...]:
    """Resolve once and return the vetted addresses, IPv4 first.

    Every resolved address must be public; one private answer refuses the
    whole name, so DNS that mixes public and internal records cannot be used
    to reach the internal side.
    """

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        addresses = [str(literal)]
    else:
        if not _HOSTNAME.fullmatch(host):
            raise PermissionError("CONNECT host is not a valid DNS name")
        lookup = resolver or asyncio.get_running_loop().getaddrinfo
        try:
            infos = await asyncio.wait_for(
                lookup(
                    host,
                    port,
                    type=socket.SOCK_STREAM,
                    proto=socket.IPPROTO_TCP,
                ),
                timeout=10.0,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            raise ConnectionError(f"cannot resolve CONNECT host {host}") from exc
        addresses = [str(info[4][0]) for info in infos]
    if not addresses:
        raise PermissionError("CONNECT host resolved to no address")
    if not all(is_public_address(address) for address in addresses):
        raise PermissionError("CONNECT destination is not a public address")
    unique = list(dict.fromkeys(str(ipaddress.ip_address(a)) for a in addresses))
    unique.sort(key=lambda address: ipaddress.ip_address(address).version)
    return tuple(unique)


async def select_connect_targets(
    authority: str,
    *,
    allow_public_egress: bool = False,
    resolver: Resolver | None = None,
) -> tuple[Sequence[str], int]:
    """Return the destinations to try and the port for one CONNECT request.

    The model-host allowlist keeps its original behaviour in both modes (the
    hostname itself is tunnelled). Anything else needs public egress mode, a
    port in ``PUBLIC_EGRESS_PORTS`` and an all-public resolution.
    """

    if not allow_public_egress:
        host, port = parse_connect_target(authority)
        return (host,), port
    host, port = split_connect_authority(authority)
    if host in ALLOWED_CONNECT_HOSTS and port == 443:
        return (host,), port
    if port not in PUBLIC_EGRESS_PORTS:
        raise PermissionError("public CONNECT is limited to ports 80 and 443")
    return await resolve_public_addresses(host, port, resolver=resolver), port


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


async def open_target_connection(
    host: str,
    port: int,
    *,
    upstream_proxy_host: str | None = None,
    upstream_proxy_port: int | None = None,
    connection_factory: ConnectionFactory = asyncio.open_connection,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    upstream = validate_upstream_proxy(upstream_proxy_host, upstream_proxy_port)
    if upstream is None:
        return await asyncio.wait_for(connection_factory(host, port), timeout=15.0)

    upstream_reader, upstream_writer = await asyncio.wait_for(
        connection_factory(*upstream), timeout=15.0
    )
    authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    try:
        upstream_writer.write(
            (
                f"CONNECT {authority} HTTP/1.1\r\n"
                f"Host: {authority}\r\n"
                "Proxy-Connection: Keep-Alive\r\n\r\n"
            ).encode("ascii")
        )
        await upstream_writer.drain()
        response = await asyncio.wait_for(
            upstream_reader.readuntil(b"\r\n\r\n"), timeout=15.0
        )
        if len(response) > MAX_HEADER_BYTES:
            raise UpstreamProxyError("upstream proxy header is too large")
        status = response.split(b"\r\n", 1)[0].decode("ascii", "strict").split()
        if (
            len(status) < 2
            or status[0] not in {"HTTP/1.0", "HTTP/1.1"}
            or not status[1].isdigit()
            or int(status[1]) // 100 != 2
        ):
            raise UpstreamProxyError("upstream proxy rejected CONNECT")
        return upstream_reader, upstream_writer
    except BaseException:
        await close_writer(upstream_writer)
        raise


async def open_first_target(
    targets: Sequence[str],
    port: int,
    *,
    upstream_proxy_host: str | None = None,
    upstream_proxy_port: int | None = None,
    connection_factory: ConnectionFactory = asyncio.open_connection,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect to the first reachable destination among already-vetted targets."""

    last_error: BaseException | None = None
    for target in targets:
        try:
            return await open_target_connection(
                target,
                port,
                upstream_proxy_host=upstream_proxy_host,
                upstream_proxy_port=upstream_proxy_port,
                connection_factory=connection_factory,
            )
        except (OSError, asyncio.TimeoutError, UpstreamProxyError) as exc:
            last_error = exc
    raise ConnectionError("no CONNECT destination was reachable") from last_error


async def handle_connect(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    upstream_proxy_host: str | None = None,
    upstream_proxy_port: int | None = None,
    *,
    connection_factory: ConnectionFactory = asyncio.open_connection,
    allow_public_egress: bool = False,
    resolver: Resolver | None = None,
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
        targets, port = await select_connect_targets(
            authority,
            allow_public_egress=allow_public_egress,
            resolver=resolver,
        )
        upstream_reader, upstream_writer = await open_first_target(
            targets,
            port,
            upstream_proxy_host=upstream_proxy_host,
            upstream_proxy_port=upstream_proxy_port,
            connection_factory=connection_factory,
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
    validate_upstream_proxy(args.upstream_proxy_host, args.upstream_proxy_port)

    async def connect_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await handle_connect(
            reader,
            writer,
            args.upstream_proxy_host,
            args.upstream_proxy_port,
            allow_public_egress=args.allow_public_egress,
        )

    connect_server = await asyncio.start_server(
        connect_handler,
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
    async with connect_server, judge_server:
        await stop.wait()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--proxy-port", type=int, default=3128)
    parser.add_argument("--judge-forward-port", type=int, default=18184)
    parser.add_argument("--judge-host", default="host.docker.internal")
    parser.add_argument("--judge-port", type=int, required=True)
    parser.add_argument("--upstream-proxy-host")
    parser.add_argument("--upstream-proxy-port", type=int)
    parser.add_argument(
        "--allow-public-egress",
        action="store_true",
        help=(
            "also tunnel to any public host on ports 80/443 after vetting every "
            "resolved address (default: model-host allowlist only)"
        ),
    )
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
