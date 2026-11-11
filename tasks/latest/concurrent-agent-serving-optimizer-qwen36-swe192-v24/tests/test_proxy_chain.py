from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"kvbench_{name}_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upstream_proxy_configuration_is_all_or_nothing() -> None:
    proxy = load_script("restricted_connect_proxy")
    assert proxy.validate_upstream_proxy(None, None) is None
    assert proxy.validate_upstream_proxy("172.16.0.1", 19080) == (
        "172.16.0.1",
        19080,
    )
    with pytest.raises(ValueError):
        proxy.validate_upstream_proxy("172.16.0.1", None)
    with pytest.raises(ValueError):
        proxy.validate_upstream_proxy(None, 19080)


def test_upstream_connect_success_and_no_direct_fallback() -> None:
    proxy = load_script("restricted_connect_proxy")

    async def scenario() -> None:
        requests: list[bytes] = []
        attempts: list[tuple[str, int]] = []

        async def fake_upstream(reader, writer) -> None:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(fake_upstream, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        async def connector(host: str, target_port: int):
            attempts.append((host, target_port))
            return await asyncio.open_connection(host, target_port)

        try:
            reader, writer = await proxy.open_target_connection(
                "chatgpt.com",
                443,
                upstream_proxy_host="127.0.0.1",
                upstream_proxy_port=port,
                connection_factory=connector,
            )
            writer.close()
            await writer.wait_closed()
            await reader.read()
            assert attempts == [("127.0.0.1", port)]
            assert requests == [
                b"CONNECT chatgpt.com:443 HTTP/1.1\r\n"
                b"Host: chatgpt.com:443\r\n"
                b"Proxy-Connection: Keep-Alive\r\n\r\n"
            ]
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_host_proxy_bridge_is_fixed_to_loopback_proxy() -> None:
    bridge = load_script("host_proxy_bridge")
    assert bridge.TARGET_HOST == "127.0.0.1"
    assert bridge.TARGET_PORT == 18080
    args = bridge.parse_args(
        [
            "--listen-host",
            "172.16.0.1",
            "--allowed-peer",
            "172.16.0.9",
        ]
    )
    assert args.listen_port == 19080
    with pytest.raises(SystemExit):
        bridge.parse_args(
            ["--listen-host", "0.0.0.0", "--allowed-peer", "172.16.0.9"]
        )
