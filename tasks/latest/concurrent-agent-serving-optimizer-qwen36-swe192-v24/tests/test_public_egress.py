"""Public egress mode of the Work sidecar proxy (task JSON ``"internet": true``)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import socket
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK_SPEC = ROOT / "tasks/concurrent_agent_serving_optimizer_qwen36_swe192_v24.json"
LAUNCHER = ROOT / "scripts/run_fixed_budget.sh"


def load_proxy() -> ModuleType:
    path = ROOT / "scripts/restricted_connect_proxy.py"
    spec = importlib.util.spec_from_file_location("kvbench_public_egress_proxy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


proxy = load_proxy()


def fake_resolver(answers: dict[str, list[str]], calls: list[str] | None = None):
    async def resolve(host: str, port: int, **_: Any) -> list[tuple]:
        if calls is not None:
            calls.append(host)
        if host not in answers:
            raise socket.gaierror(socket.EAI_NONAME, "unknown host")
        result = []
        for address in answers[host]:
            family = socket.AF_INET6 if ":" in address else socket.AF_INET
            sockaddr = (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
            result.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr))
        return result

    return resolve


def forbidden_resolver(*_: Any, **__: Any):
    raise AssertionError("this request must not be resolved")


def select(authority: str, *, public: bool = True, resolver=None):
    return asyncio.run(
        proxy.select_connect_targets(
            authority, allow_public_egress=public, resolver=resolver
        )
    )


def test_flag_is_off_by_default_and_opt_in() -> None:
    assert proxy.parse_args(["--judge-port", "18188"]).allow_public_egress is False
    assert (
        proxy.parse_args(
            ["--judge-port", "18188", "--allow-public-egress"]
        ).allow_public_egress
        is True
    )


def test_default_mode_keeps_the_model_host_allowlist() -> None:
    assert select("chatgpt.com:443", public=False) == (("chatgpt.com",), 443)
    with pytest.raises(PermissionError):
        select("github.com:443", public=False, resolver=forbidden_resolver)
    with pytest.raises(ValueError):
        select("8.8.8.8:443", public=False)


def test_public_mode_keeps_allowlisted_hosts_on_their_hostname_tunnel() -> None:
    for host in sorted(proxy.ALLOWED_CONNECT_HOSTS):
        assert select(f"{host}:443", resolver=forbidden_resolver) == ((host,), 443)


def test_public_mode_resolves_once_and_returns_vetted_addresses_ipv4_first() -> None:
    calls: list[str] = []
    resolver = fake_resolver(
        {"github.com": ["2606:50c0:8000::153", "140.82.112.3", "140.82.112.3"]},
        calls,
    )
    assert select("GitHub.com.:443", resolver=resolver) == (
        ("140.82.112.3", "2606:50c0:8000::153"),
        443,
    )
    assert calls == ["github.com"]
    assert select("github.com:80", resolver=resolver)[1] == 80


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # loopback
        "10.1.2.3",  # RFC 1918
        "172.17.0.1",  # Docker default bridge gateway
        "172.18.0.5",  # user-defined Docker network
        "192.168.1.10",  # RFC 1918
        "169.254.169.254",  # link-local / cloud metadata
        "100.64.0.1",  # CGNAT
        "224.0.0.1",  # multicast (is_global is True for it)
        "240.0.0.1",  # reserved
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fc00::1",
        "ff02::1",  # IPv6 multicast (is_global is True for it)
        "::ffff:10.0.0.1",
        "::ffff:8.8.8.8",  # IPv4-mapped forms are refused outright
    ],
)
def test_public_mode_refuses_every_non_public_resolution(address: str) -> None:
    with pytest.raises(PermissionError):
        select("internal.example:443", resolver=fake_resolver({"internal.example": [address]}))


def test_one_private_answer_refuses_the_whole_name() -> None:
    resolver = fake_resolver({"rebind.example": ["93.184.215.14", "10.0.0.7"]})
    with pytest.raises(PermissionError):
        select("rebind.example:443", resolver=resolver)


def test_public_mode_literal_addresses() -> None:
    assert select("8.8.8.8:443", resolver=forbidden_resolver) == (("8.8.8.8",), 443)
    assert select("[2606:4700:4700::1111]:443", resolver=forbidden_resolver) == (
        ("2606:4700:4700::1111",),
        443,
    )
    for refused in ("127.0.0.1:443", "10.0.0.1:443", "172.17.0.1:80", "[::1]:443"):
        with pytest.raises(PermissionError):
            select(refused, resolver=forbidden_resolver)


@pytest.mark.parametrize("authority", ["github.com:22", "github.com:8080", "8.8.8.8:53"])
def test_public_mode_is_limited_to_web_ports(authority: str) -> None:
    with pytest.raises(PermissionError):
        select(authority, resolver=forbidden_resolver)


@pytest.mark.parametrize("authority", ["github.com", "a:b:443", "[::1:443", "host:99999", ":443"])
def test_malformed_authorities_are_rejected(authority: str) -> None:
    with pytest.raises(ValueError):
        select(authority, resolver=forbidden_resolver)


def test_invalid_dns_names_and_resolution_failures() -> None:
    with pytest.raises(PermissionError):
        select("bad name.example:443", resolver=forbidden_resolver)
    with pytest.raises(ConnectionError):
        select("missing.example:443", resolver=fake_resolver({}))
    with pytest.raises(PermissionError):
        select("empty.example:443", resolver=fake_resolver({"empty.example": []}))


async def _proxy_exchange(
    request: bytes,
    *,
    allow_public_egress: bool,
    resolver,
    connection_factory,
) -> bytes:
    async def handler(reader, writer) -> None:
        await proxy.handle_connect(
            reader,
            writer,
            allow_public_egress=allow_public_egress,
            resolver=resolver,
            connection_factory=connection_factory,
        )

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(request)
        await writer.drain()
        response = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
        writer.close()
        await writer.wait_closed()
        return response
    finally:
        server.close()
        await server.wait_closed()


class RecordingTarget:
    """Connection factory that records destinations and serves an idle stream."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.attempts: list[tuple[str, int]] = []
        self.fail = fail or set()

    async def __call__(self, host: str, port: int):
        self.attempts.append((host, port))
        if host in self.fail:
            raise ConnectionRefusedError(host)

        async def sink(reader, writer) -> None:
            await reader.read()
            writer.close()

        self.server = await asyncio.start_server(sink, "127.0.0.1", 0)
        return await asyncio.open_connection(
            "127.0.0.1", self.server.sockets[0].getsockname()[1]
        )


def test_public_connect_tunnels_to_a_vetted_address_without_re_resolution() -> None:
    calls: list[str] = []
    resolver = fake_resolver({"pypi.org": ["151.101.0.223", "151.101.64.223"]}, calls)
    # The first vetted address is unreachable; the proxy moves to the next one
    # from the same single resolution instead of resolving the name again.
    target = RecordingTarget(fail={"151.101.0.223"})

    async def scenario() -> bytes:
        try:
            return await _proxy_exchange(
                b"CONNECT pypi.org:443 HTTP/1.1\r\nHost: pypi.org:443\r\n\r\n",
                allow_public_egress=True,
                resolver=resolver,
                connection_factory=target,
            )
        finally:
            target.server.close()

    assert asyncio.run(scenario()).startswith(b"HTTP/1.1 200 ")
    assert calls == ["pypi.org"]
    assert target.attempts == [("151.101.0.223", 443), ("151.101.64.223", 443)]


def test_refused_connect_is_403_and_never_opens_a_connection() -> None:
    for public, request in (
        (True, b"CONNECT host.docker.internal:443 HTTP/1.1\r\n\r\n"),
        (True, b"CONNECT 169.254.169.254:80 HTTP/1.1\r\n\r\n"),
        (True, b"CONNECT github.com:22 HTTP/1.1\r\n\r\n"),
        (False, b"CONNECT github.com:443 HTTP/1.1\r\n\r\n"),
    ):
        target = RecordingTarget()
        response = asyncio.run(
            _proxy_exchange(
                request,
                allow_public_egress=public,
                resolver=fake_resolver({"host.docker.internal": ["172.17.0.1"]}),
                connection_factory=target,
            )
        )
        assert response.startswith(b"HTTP/1.1 403 Forbidden"), (public, request)
        assert target.attempts == []


def test_upstream_proxy_receives_the_vetted_address_not_the_name() -> None:
    async def scenario(address: str) -> bytes:
        requests: list[bytes] = []

        async def fake_upstream(reader, writer) -> None:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(fake_upstream, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            targets, target_port = await proxy.select_connect_targets(
                "example.org:443",
                allow_public_egress=True,
                resolver=fake_resolver({"example.org": [address]}),
            )
            reader, writer = await proxy.open_first_target(
                targets,
                target_port,
                upstream_proxy_host="127.0.0.1",
                upstream_proxy_port=port,
            )
            writer.close()
            await writer.wait_closed()
            return requests[0]
        finally:
            server.close()
            await server.wait_closed()

    assert asyncio.run(scenario("93.184.215.14")).startswith(
        b"CONNECT 93.184.215.14:443 HTTP/1.1\r\n"
    )
    assert asyncio.run(scenario("2606:2800:21f:cb07:6820:80da:af6b:8b2c")).startswith(
        b"CONNECT [2606:2800:21f:cb07:6820:80da:af6b:8b2c]:443 HTTP/1.1\r\n"
    )


def test_launcher_selects_the_sidecar_policy_from_the_task_json() -> None:
    assert json.loads(TASK_SPEC.read_text())["internet"] is True
    source = LAUNCHER.read_text()

    assert 'if has("internet") then .internet else true end' in source
    assert 'proxy_egress_args=(--allow-public-egress)' in source
    assert '${proxy_egress_args[@]+"${proxy_egress_args[@]}"}' in source
    # Public mode expects a public tunnel and refuses non-public destinations;
    # allowlist mode keeps refusing non-allowlisted hosts.
    public_branch = source[
        source.index('if [[ "${TASK_INTERNET}" == "true" ]]; then\n  connect_policy_probes=(') :
        source.index('connect_policy_probes=("github.com:443" 403)')
    ]
    assert '"github.com:443" 200' in public_branch
    for refused in (
        '"127.0.0.1:443" 403',
        '"${docker_bridge_gateway}:443" 403',
        '"${docker_bridge_gateway}:${DOCKER_PROXY_PORT}" 403',
        '"host.docker.internal:443" 403',
        '"169.254.169.254:80" 403',
        '"10.0.0.1:443" 403',
        '"github.com:22" 403',
    ):
        assert refused in public_branch
    assert "https://pypi.org/simple/pip/" in source
    # The internal Work network still has no direct route out.
    assert "--internal" in source
    assert 'die "Work network unexpectedly has direct public egress"' in source
    assert "network: $network_policy" in source
    assert "internet: $internet" in source
    assert "codex_web_search: $web_search" in source
    assert 'export MLSYSFORGE_WORK_INTERNET="${TASK_INTERNET}"' in source
    assert source.index("export MLSYSFORGE_WORK_INTERNET") < source.index('stage="agent_run"')


@pytest.mark.parametrize(
    "work_internet,mode,prompt_internet,search",
    [
        (None, "live", True, "live"),
        ("true", "live", True, "live"),
        ("true", "disabled", True, "disabled"),
        ("false", "live", False, "disabled"),
        ("false", "disabled", False, "disabled"),
    ],
)
def test_task_internet_flag_drives_prompt_and_search(
    runtime_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    work_internet: str | None,
    mode: str,
    prompt_internet: bool,
    search: str,
) -> None:
    seen: list[bool] = []

    def prompt(*_: Any, **kwargs: Any) -> str:
        seen.append(kwargs["internet"])
        return "prompt"

    monkeypatch.setattr(runtime_module, "_original_generate_prompt", prompt)
    monkeypatch.setattr(
        runtime_module,
        "_original_format_run_cmd",
        lambda _self, *args, **kwargs: (
            "codex exec -c 'web_search=\"%s\"' --example"
            % ("live" if kwargs["internet"] else "disabled")
        ),
    )
    if work_internet is None:
        monkeypatch.delenv("MLSYSFORGE_WORK_INTERNET", raising=False)
    else:
        monkeypatch.setenv("MLSYSFORGE_WORK_INTERNET", work_internet)
    monkeypatch.setenv("CODEX_WEB_SEARCH_MODE", mode)

    runtime_module.configured_generate_prompt("query", internet=True)
    command = runtime_module.configured_format_run_cmd(object(), internet=True)

    assert seen == [prompt_internet]
    assert f'web_search="{search}"' in command
    assert command.count("web_search=") == 1


def test_invalid_task_internet_value_fails_closed(
    runtime_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLSYSFORGE_WORK_INTERNET", "yes")
    with pytest.raises(RuntimeError, match="MLSYSFORGE_WORK_INTERNET"):
        runtime_module.configured_generate_prompt("query")


def test_live_search_option_is_not_duplicated(
    runtime_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_WEB_SEARCH_MODE", "live")
    monkeypatch.setattr(
        runtime_module,
        "_original_format_run_cmd",
        lambda _self, *args, **kwargs: (
            "codex exec -c 'web_search=\"live\"' --example"
            if kwargs["internet"]
            else "codex exec -c 'web_search=\"disabled\"' --example"
        ),
    )
    command = runtime_module.configured_format_run_cmd(object())
    assert command.count('web_search="live"') == 1
