from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"qvg_{name}_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("authority", "expected"),
    [
        ("api.openai.com:443", ("api.openai.com", 443)),
        ("AUTH.OPENAI.COM:443", ("auth.openai.com", 443)),
        ("chatgpt.com.:443", ("chatgpt.com", 443)),
    ],
)
def test_connect_allowlist_accepts_only_named_tls_targets(
    authority: str, expected: tuple[str, int]
) -> None:
    proxy = load_script("restricted_connect_proxy")
    assert proxy.parse_target(authority) == expected


@pytest.mark.parametrize(
    "authority",
    [
        "api.openai.com:80",
        "example.com:443",
        "sub.api.openai.com:443",
        "api.openai.com.example:443",
        "127.0.0.1:443",
        "api.openai.com:443:443",
        "api.openai.com",
    ],
)
def test_connect_allowlist_rejects_every_other_authority(authority: str) -> None:
    proxy = load_script("restricted_connect_proxy")
    with pytest.raises((PermissionError, ValueError)):
        proxy.parse_target(authority)


def test_upstream_proxy_configuration_is_all_or_nothing() -> None:
    proxy = load_script("restricted_connect_proxy")
    assert proxy.validate_upstream_proxy(None, None) is None
    assert proxy.validate_upstream_proxy("bridge.local", 19080) == (
        "bridge.local",
        19080,
    )
    with pytest.raises(ValueError):
        proxy.validate_upstream_proxy("bridge.local", None)
    with pytest.raises(ValueError):
        proxy.validate_upstream_proxy(None, 19080)


def test_upstream_connect_success_tunnels_bytes() -> None:
    proxy = load_script("restricted_connect_proxy")

    async def scenario() -> None:
        received_requests: list[bytes] = []
        upstream_done = asyncio.Event()
        proxy_done = asyncio.Event()

        async def fake_upstream(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                received_requests.append(await reader.readuntil(b"\r\n\r\n"))
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
                payload = await reader.read()
                writer.write(b"via-upstream:" + payload)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                upstream_done.set()

        upstream_server = await asyncio.start_server(
            fake_upstream, "127.0.0.1", 0
        )
        upstream_port = upstream_server.sockets[0].getsockname()[1]

        async def proxy_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                await proxy.handle_connect(
                    reader,
                    writer,
                    "127.0.0.1",
                    upstream_port,
                )
            finally:
                proxy_done.set()

        proxy_server = await asyncio.start_server(proxy_handler, "127.0.0.1", 0)
        proxy_port = proxy_server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
            writer.write(
                b"CONNECT api.openai.com:443 HTTP/1.1\r\n"
                b"Host: api.openai.com:443\r\n\r\n"
            )
            await writer.drain()
            response = await reader.readuntil(b"\r\n\r\n")
            assert response.startswith(b"HTTP/1.1 200 Connection Established\r\n")

            writer.write(b"codex-request")
            await writer.drain()
            writer.write_eof()
            assert await reader.read() == b"via-upstream:codex-request"
            writer.close()
            await writer.wait_closed()

            await asyncio.wait_for(upstream_done.wait(), 2.0)
            await asyncio.wait_for(proxy_done.wait(), 2.0)
            assert received_requests == [
                b"CONNECT api.openai.com:443 HTTP/1.1\r\n"
                b"Host: api.openai.com:443\r\n"
                b"Proxy-Connection: Keep-Alive\r\n\r\n"
            ]
        finally:
            proxy_server.close()
            upstream_server.close()
            await proxy_server.wait_closed()
            await upstream_server.wait_closed()

    asyncio.run(scenario())


def test_upstream_connect_rejection_returns_bad_gateway() -> None:
    proxy = load_script("restricted_connect_proxy")

    async def scenario() -> None:
        connection_attempts: list[tuple[str, int]] = []
        upstream_done = asyncio.Event()
        proxy_done = asyncio.Event()

        async def rejecting_upstream(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(
                    b"HTTP/1.1 407 Proxy Authentication Required\r\n"
                    b"Content-Length: 0\r\n\r\n"
                )
                await writer.drain()
                await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()
                upstream_done.set()

        upstream_server = await asyncio.start_server(
            rejecting_upstream, "127.0.0.1", 0
        )
        upstream_port = upstream_server.sockets[0].getsockname()[1]

        async def recording_connection_factory(
            host: str, port: int
        ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
            connection_attempts.append((host, port))
            return await asyncio.open_connection(host, port)

        async def proxy_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                await proxy.handle_connect(
                    reader,
                    writer,
                    "127.0.0.1",
                    upstream_port,
                    connection_factory=recording_connection_factory,
                )
            finally:
                proxy_done.set()

        proxy_server = await asyncio.start_server(proxy_handler, "127.0.0.1", 0)
        proxy_port = proxy_server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
            writer.write(b"CONNECT chatgpt.com:443 HTTP/1.1\r\n\r\n")
            await writer.drain()
            response = await reader.read()
            assert response.startswith(b"HTTP/1.1 502 Bad Gateway\r\n")
            writer.close()
            await writer.wait_closed()

            await asyncio.wait_for(upstream_done.wait(), 2.0)
            await asyncio.wait_for(proxy_done.wait(), 2.0)
            assert connection_attempts == [("127.0.0.1", upstream_port)]
        finally:
            proxy_server.close()
            upstream_server.close()
            await proxy_server.wait_closed()
            await upstream_server.wait_closed()

    asyncio.run(scenario())


def test_upstream_connection_failure_has_no_direct_fallback() -> None:
    proxy = load_script("restricted_connect_proxy")

    async def scenario() -> None:
        connection_attempts: list[tuple[str, int]] = []

        async def unavailable_proxy(
            host: str, port: int
        ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
            connection_attempts.append((host, port))
            raise OSError("bridge unavailable")

        with pytest.raises(OSError, match="bridge unavailable"):
            await proxy.open_target_connection(
                "api.openai.com",
                443,
                upstream_proxy_host="docker-bridge",
                upstream_proxy_port=19080,
                connection_factory=unavailable_proxy,
            )
        assert connection_attempts == [("docker-bridge", 19080)]

    asyncio.run(scenario())


def test_upstream_proxy_is_optional_and_direct_mode_remains_explicit() -> None:
    proxy = load_script("restricted_connect_proxy")

    async def scenario() -> None:
        connection_attempts: list[tuple[str, int]] = []
        target_done = asyncio.Event()

        async def fake_target(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                writer.write(b"direct-mode")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                target_done.set()

        target_server = await asyncio.start_server(fake_target, "127.0.0.1", 0)
        target_port = target_server.sockets[0].getsockname()[1]

        async def direct_connector(
            host: str, port: int
        ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
            connection_attempts.append((host, port))
            return await asyncio.open_connection("127.0.0.1", target_port)

        try:
            reader, writer = await proxy.open_target_connection(
                "api.openai.com",
                443,
                connection_factory=direct_connector,
            )
            assert await reader.read() == b"direct-mode"
            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(target_done.wait(), 2.0)
            assert connection_attempts == [("api.openai.com", 443)]
        finally:
            target_server.close()
            await target_server.wait_closed()

    asyncio.run(scenario())


def test_host_bridge_forwards_to_fixed_loopback_proxy() -> None:
    bridge = load_script("host_proxy_bridge")
    assert bridge.TARGET_HOST == "127.0.0.1"
    assert bridge.TARGET_PORT == 18080
    args = bridge.parse_args(["--listen-host", "172.17.0.1"])
    assert args.listen_host == "172.17.0.1"
    assert args.listen_port == 19080

    async def scenario() -> None:
        connection_attempts: list[tuple[str, int]] = []
        bridge_done = asyncio.Event()

        async def loopback_proxy(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                payload = await reader.read()
                writer.write(b"loopback-proxy:" + payload)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        loopback_server = await asyncio.start_server(
            loopback_proxy, "127.0.0.1", 0
        )
        loopback_port = loopback_server.sockets[0].getsockname()[1]

        async def fixed_target_connector(
            host: str, port: int
        ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
            connection_attempts.append((host, port))
            return await asyncio.open_connection("127.0.0.1", loopback_port)

        async def bridge_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                await bridge.handle_connection(
                    reader,
                    writer,
                    connection_factory=fixed_target_connector,
                )
            finally:
                bridge_done.set()

        bridge_server = await asyncio.start_server(bridge_handler, "127.0.0.1", 0)
        bridge_port = bridge_server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", bridge_port)
            writer.write(b"connect-chain")
            await writer.drain()
            writer.write_eof()
            assert await reader.read() == b"loopback-proxy:connect-chain"
            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(bridge_done.wait(), 2.0)
            assert connection_attempts == [("127.0.0.1", 18080)]
        finally:
            bridge_server.close()
            loopback_server.close()
            await bridge_server.wait_closed()
            await loopback_server.wait_closed()

    asyncio.run(scenario())

