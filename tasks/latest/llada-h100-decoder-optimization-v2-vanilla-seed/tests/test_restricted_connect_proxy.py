"""Unit tests for the restricted CONNECT sidecar (scripts/restricted_connect_proxy.py).

Other tasks (flux, sdv2, orpo, gkd, Miles) mount this same file as their Work
egress sidecar, so both CONNECT policies are pinned here: the default
allowlist-only mode and ``--allow-public-egress``.
"""

from __future__ import annotations

import asyncio
import socket
import sys

import pytest

from conftest import TASK_ROOT, load_module


proxy = load_module(
    "llada_restricted_connect_proxy",
    TASK_ROOT / "scripts/restricted_connect_proxy.py",
)

PUBLIC_V4 = "93.184.216.34"
PUBLIC_V4_B = "151.101.1.69"
PUBLIC_V6 = "2606:2800:220:1:248:1893:25c8:1946"
NON_PUBLIC = (
    "127.0.0.1",
    "10.1.2.3",
    "172.17.0.1",
    "172.20.0.2",
    "192.168.1.1",
    "169.254.169.254",
    "100.64.0.1",
    "0.0.0.0",
    "224.0.0.1",
    "239.255.255.250",
    "240.0.0.1",
    "255.255.255.255",
    "192.0.2.1",
    "::1",
    "::",
    "fe80::1",
    "fc00::1",
    "fd12:3456::1",
    "ff02::1",
    "::ffff:10.0.0.1",
    "2002:a00:1::1",
    "64:ff9b::a00:1",
    "2001:db8::1",
)


def _addrinfo(address: str, port: int) -> tuple:
    if ":" in address:
        return (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port, 0, 0))
    return (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))


DNS = {
    "public.example": [PUBLIC_V4, PUBLIC_V6],
    "fallback.example": [PUBLIC_V4, PUBLIC_V4_B],
    "rebind.example": [PUBLIC_V4, "10.0.0.5"],
    "host.docker.internal": ["172.17.0.1"],
    "localhost": ["127.0.0.1", "::1"],
    "metadata.example": ["169.254.169.254"],
}


class _Harness:
    """Run the CONNECT handler against a fake resolver and fake upstreams."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, refuse: set[str] | None = None):
        self.monkeypatch = monkeypatch
        self.refuse = refuse or set()
        self.resolved: list[str] = []
        self.dialled: list[tuple[str, int, dict]] = []

    async def _getaddrinfo(self, host, port, *args, **kwargs):
        self.resolved.append(host)
        if host not in DNS:
            raise socket.gaierror(socket.EAI_NONAME, "unknown host")
        return [_addrinfo(address, port) for address in DNS[host]]

    async def exchange(self, request: bytes, *, allow_public_egress: bool) -> bytes:
        real_open_connection = asyncio.open_connection
        loop = asyncio.get_running_loop()
        self.monkeypatch.setattr(loop, "getaddrinfo", self._getaddrinfo)

        async def upstream_handler(reader, writer):
            writer.write(b"upstream-hello")
            await writer.drain()
            writer.close()

        upstream = await asyncio.start_server(upstream_handler, "127.0.0.1", 0)
        upstream_port = upstream.sockets[0].getsockname()[1]

        async def fake_open_connection(host, port, **kwargs):
            self.dialled.append((host, port, kwargs))
            if host in self.refuse:
                raise ConnectionRefusedError(host)
            return await real_open_connection("127.0.0.1", upstream_port)

        self.monkeypatch.setattr(proxy.asyncio, "open_connection", fake_open_connection)
        server = await asyncio.start_server(
            lambda r, w: proxy.handle_connect(r, w, allow_public_egress=allow_public_egress),
            "127.0.0.1",
            0,
        )
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await real_open_connection("127.0.0.1", port)
            writer.write(request)
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), timeout=5)
            writer.close()
            return response
        finally:
            self.monkeypatch.setattr(proxy.asyncio, "open_connection", real_open_connection)
            server.close()
            upstream.close()
            await server.wait_closed()
            await upstream.wait_closed()

    def run(self, authority: str, *, allow_public_egress: bool) -> bytes:
        request = f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n".encode()
        return asyncio.run(self.exchange(request, allow_public_egress=allow_public_egress))


def _status(response: bytes) -> int:
    return int(response.split(b"\r\n", 1)[0].split()[1])


# --- address classification -------------------------------------------------


@pytest.mark.parametrize("address", NON_PUBLIC)
def test_non_public_addresses_are_refused(address: str) -> None:
    assert proxy.is_public_address(address) is False


@pytest.mark.parametrize("address", [PUBLIC_V4, PUBLIC_V4_B, PUBLIC_V6, "8.8.8.8", "2606:4700:4700::1111"])
def test_global_unicast_addresses_are_public(address: str) -> None:
    assert proxy.is_public_address(address) is True


def test_unparseable_addresses_are_not_public() -> None:
    assert proxy.is_public_address("not-an-ip") is False
    assert proxy.is_public_address("fe80::1%eth0") is False


# --- authority parsing --------------------------------------------------------


def test_default_mode_keeps_the_model_host_allowlist() -> None:
    for host in sorted(proxy.ALLOWED_CONNECT_HOSTS):
        assert proxy.parse_connect_target(f"{host}:443") == (host, 443)
        assert proxy.parse_connect_target(f"{host.upper()}.:443") == (host, 443)
    for authority in ("github.com:443", "pypi.org:443", "chatgpt.com:80", "chatgpt.com:22"):
        with pytest.raises(PermissionError):
            proxy.parse_connect_target(authority)
    with pytest.raises(ValueError, match="literal IP"):
        proxy.parse_connect_target(f"{PUBLIC_V4}:443")
    with pytest.raises(ValueError):
        proxy.parse_connect_target("chatgpt.com")


def test_public_mode_accepts_names_and_public_literals_on_web_ports() -> None:
    parse = lambda value: proxy.parse_connect_target(value, allow_public_egress=True)
    assert parse("github.com:443") == ("github.com", 443)
    assert parse("GitHub.com.:80") == ("github.com", 80)
    assert parse(f"{PUBLIC_V4}:443") == (PUBLIC_V4, 443)
    assert parse("chatgpt.com:443") == ("chatgpt.com", 443)


@pytest.mark.parametrize("port", [22, 25, 53, 3128, 3130, 3131, 8000, 8080, 18384, 65535])
def test_public_mode_refuses_other_ports(port: int) -> None:
    with pytest.raises(PermissionError):
        proxy.parse_connect_target(f"github.com:{port}", allow_public_egress=True)
    with pytest.raises(PermissionError):
        proxy.parse_connect_target(f"{PUBLIC_V4}:{port}", allow_public_egress=True)


@pytest.mark.parametrize("address", [a for a in NON_PUBLIC if ":" not in a])
def test_public_mode_refuses_non_public_ipv4_literals(address: str) -> None:
    with pytest.raises(PermissionError):
        proxy.parse_connect_target(f"{address}:443", allow_public_egress=True)


# --- resolution ---------------------------------------------------------------


def test_resolution_requires_every_address_to_be_public(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch)

    async def resolve(host: str):
        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", harness._getaddrinfo)
        return await proxy.resolve_public_addresses(host, 443)

    assert asyncio.run(resolve("public.example")) == [PUBLIC_V4, PUBLIC_V6]
    for host in ("rebind.example", "host.docker.internal", "localhost", "metadata.example"):
        with pytest.raises(PermissionError):
            asyncio.run(resolve(host))


# --- end-to-end CONNECT handling ----------------------------------------------


def test_public_mode_tunnels_to_the_vetted_address_without_re_resolving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    response = harness.run("public.example:443", allow_public_egress=True)
    assert _status(response) == 200
    assert response.endswith(b"upstream-hello")
    assert harness.resolved == ["public.example"]
    assert harness.dialled == [(PUBLIC_V4, 443, {"flags": socket.AI_NUMERICHOST})]


def test_public_mode_falls_back_to_the_next_vetted_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, refuse={PUBLIC_V4})
    response = harness.run("fallback.example:80", allow_public_egress=True)
    assert _status(response) == 200
    assert [host for host, _port, _kw in harness.dialled] == [PUBLIC_V4, PUBLIC_V4_B]
    assert harness.resolved == ["fallback.example"]


def test_public_mode_dials_a_public_literal_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch)
    assert _status(harness.run(f"{PUBLIC_V4}:443", allow_public_egress=True)) == 200
    assert harness.resolved == []
    assert harness.dialled == [(PUBLIC_V4, 443, {"flags": socket.AI_NUMERICHOST})]


@pytest.mark.parametrize(
    "authority",
    [
        "rebind.example:443",
        "host.docker.internal:443",
        "localhost:80",
        "metadata.example:80",
        "172.17.0.1:443",
        "127.0.0.1:443",
        "10.0.0.5:80",
        "public.example:22",
        f"{PUBLIC_V4}:8080",
    ],
)
def test_public_mode_refuses_non_public_or_non_web_targets_with_403(
    monkeypatch: pytest.MonkeyPatch,
    authority: str,
) -> None:
    harness = _Harness(monkeypatch)
    assert _status(harness.run(authority, allow_public_egress=True)) == 403
    assert harness.dialled == []


def test_allowlisted_model_hosts_work_in_both_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    for allow_public_egress in (False, True):
        harness = _Harness(monkeypatch)
        assert _status(harness.run("chatgpt.com:443", allow_public_egress=allow_public_egress)) == 200
        assert harness.dialled == [("chatgpt.com", 443, {})]


def test_default_mode_still_refuses_public_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch)
    assert _status(harness.run("public.example:443", allow_public_egress=False)) == 403
    assert _status(harness.run("github.com:443", allow_public_egress=False)) == 403
    assert harness.resolved == [] and harness.dialled == []


def test_cli_flag_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    base = ["restricted_connect_proxy.py", "--evaluation-upstream-host", "172.17.0.1", "--evaluation-upstream-port", "18384"]
    monkeypatch.setattr(sys, "argv", base)
    assert proxy.parse_args().allow_public_egress is False
    monkeypatch.setattr(sys, "argv", [*base, "--allow-public-egress"])
    assert proxy.parse_args().allow_public_egress is True
