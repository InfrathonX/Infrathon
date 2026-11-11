from __future__ import annotations

import argparse
import asyncio
import socket

import pytest

from sforge.harness.llm_judge import egress_proxy


def test_connect_targets_are_allowlisted_hosts_on_443_only() -> None:
    allowed = frozenset({"chatgpt.com"})
    assert egress_proxy.parse_connect_target("chatgpt.com:443", allowed) == ("chatgpt.com", 443)
    assert egress_proxy.parse_connect_target("ChatGPT.com.:443", allowed) == ("chatgpt.com", 443)
    for authority in ("evil.example:443", "chatgpt.com:80", "1.2.3.4:443"):
        with pytest.raises(PermissionError):
            egress_proxy.parse_connect_target(authority, allowed)
    with pytest.raises(ValueError):
        egress_proxy.parse_connect_target("chatgpt.com", allowed)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _exchange(port: int, payload: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(payload)
        await writer.drain()
        return await asyncio.wait_for(reader.read(4096), timeout=5)
    except ConnectionResetError:
        return b""
    finally:
        writer.close()


def _serve_and_probe(allow_peer: list[str], payload: bytes) -> bytes:
    port = _free_port()
    args = argparse.Namespace(
        listen_host="127.0.0.1",
        proxy_port=port,
        allow="chatgpt.com",
        allow_peer=allow_peer,
        forward=[],
    )

    async def main() -> bytes:
        server = asyncio.ensure_future(egress_proxy.serve(args))
        for _ in range(100):
            try:
                return await _exchange(port, payload)
            except ConnectionRefusedError:
                await asyncio.sleep(0.02)
        raise AssertionError("proxy did not start")

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(main())
    finally:
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()


def test_peers_inside_the_review_subnet_are_served() -> None:
    reply = _serve_and_probe(["127.0.0.0/8"], b"GET /health HTTP/1.1\r\n\r\n")
    assert reply.startswith(b"HTTP/1.1 200")


def test_disallowed_connect_is_forbidden() -> None:
    reply = _serve_and_probe(["127.0.0.0/8"], b"CONNECT evil.example:443 HTTP/1.1\r\n\r\n")
    assert reply.startswith(b"HTTP/1.1 403")


def test_peers_outside_the_review_subnet_get_nothing() -> None:
    reply = _serve_and_probe(["10.99.0.0/16"], b"GET /health HTTP/1.1\r\n\r\n")
    assert reply == b""


# --- ReviewEgress lifecycle (fake Docker client) ---

from types import SimpleNamespace  # noqa: E402

from sforge.harness.llm_judge import egress as egress_module  # noqa: E402
from sforge.harness.llm_judge.egress import EgressPlan, ReviewEgress  # noqa: E402


class _FakeNetwork:
    def __init__(self, name: str, kwargs: dict, removals_failing: int) -> None:
        self.name = name
        self.kwargs = kwargs
        self.attrs = {"IPAM": {"Config": [{"Subnet": "172.30.0.0/16"}]}}
        self.connected: list = []
        self.removals_failing = removals_failing
        self.removed = False

    def reload(self) -> None:
        pass

    def connect(self, container, aliases=None) -> None:
        self.connected.append(container)
        container.attrs["NetworkSettings"]["Networks"][self.name] = {"IPAddress": "172.30.0.2"}

    def remove(self) -> None:
        if self.removals_failing:
            self.removals_failing -= 1
            raise RuntimeError("network has active endpoints")
        self.removed = True


class _FakeSidecar:
    def __init__(self, kwargs: dict) -> None:
        self.kwargs = kwargs
        self.status = "running"
        self.attrs = {"NetworkSettings": {"Networks": {}}}
        self.removed = False

    def reload(self) -> None:
        pass

    def exec_run(self, cmd):
        return SimpleNamespace(exit_code=0, output=b"")

    def logs(self, tail=20):
        return b""

    def remove(self, force=False) -> None:
        self.removed = True
        self.status = "removed"


class _FakeEgressDocker:
    def __init__(self, removals_failing: int = 0) -> None:
        self.networks_created: list[_FakeNetwork] = []
        self.sidecars: list[_FakeSidecar] = []
        self.removals_failing = removals_failing
        self.networks = SimpleNamespace(create=self._create_network)
        self.containers = SimpleNamespace(run=self._run)

    def _create_network(self, name, **kwargs):
        network = _FakeNetwork(name, kwargs, self.removals_failing)
        self.networks_created.append(network)
        return network

    def _run(self, image, **kwargs):
        sidecar = _FakeSidecar(kwargs)
        self.sidecars.append(sidecar)
        return sidecar


def test_review_network_hides_the_host_and_survives_a_dead_sidecar(monkeypatch) -> None:
    monkeypatch.setattr(egress_module.time, "sleep", lambda seconds: None)
    client = _FakeEgressDocker(removals_failing=2)
    egress = ReviewEgress(
        client,
        mode="isolated",
        plan=EgressPlan(allow_hosts=("chatgpt.com",)),
        sidecar_image_resolver=lambda image: image,
    )

    endpoint = egress.endpoint("judge-image")
    (network,) = client.networks_created
    assert network.kwargs["internal"] is True
    assert network.kwargs["options"] == {"com.docker.network.bridge.inhibit_ipv4": "true"}
    assert "--allow-peer" in client.sidecars[0].kwargs["command"]
    assert "172.30.0.0/16" in client.sidecars[0].kwargs["command"]
    assert endpoint.proxy_url == "http://172.30.0.2:3128"

    client.sidecars[0].status = "exited"
    egress.endpoint("judge-image")
    assert len(client.networks_created) == 1  # same network, new sidecar
    assert len(client.sidecars) == 2 and client.sidecars[0].removed

    egress.close()
    assert network.removed  # retried until the last review detached
    assert client.sidecars[1].removed
