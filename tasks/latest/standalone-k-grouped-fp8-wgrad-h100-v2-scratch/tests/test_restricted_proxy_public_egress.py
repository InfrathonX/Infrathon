"""Restricted CONNECT sidecar: default allowlist and --allow-public-egress."""

from __future__ import annotations

import asyncio
import importlib.util
import socket
from pathlib import Path

import pytest


PROXY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "restricted_proxy.py"


def _load_proxy():
    """Load the sidecar source as an isolated module."""
    spec = importlib.util.spec_from_file_location("restricted_proxy_under_test", PROXY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


proxy = _load_proxy()


def _fake_getaddrinfo(table, calls=None):
    """Return a socket.getaddrinfo stand-in answering names from ``table``.

    Literal IPs and any other host fall through to the real resolver, which
    answers literals locally without DNS.
    """
    real_getaddrinfo = socket.getaddrinfo

    def fake(host, port, family=0, type=0, proto=0, flags=0):
        if calls is not None:
            calls.append(host)
        if host not in table:
            return real_getaddrinfo(host, port, family, type, proto, flags)
        answers = table[host]
        if callable(answers):
            answers = answers()
        return [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (address, port) if ":" not in address else (address, port, 0, 0),
            )
            for address in answers
        ]

    return fake


async def _connect_status(target: str, *, allow_public_egress: bool) -> bytes:
    """Send one CONNECT through a loopback instance of the sidecar."""
    open_connection = asyncio.open_connection
    server = await asyncio.start_server(
        lambda reader, writer: proxy.handle_connect(
            reader, writer, allow_public_egress=allow_public_egress
        ),
        "127.0.0.1",
        0,
    )
    port = server.sockets[0].getsockname()[1]
    async with server:
        reader, writer = await open_connection("127.0.0.1", port)
        writer.write(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode("ascii"))
        await writer.drain()
        status = await asyncio.wait_for(reader.readline(), timeout=10.0)
        writer.close()
    return status


def test_default_mode_is_the_model_auth_allowlist() -> None:
    """Without the flag nothing but the model-auth hosts on 443 passes."""
    assert proxy.ALLOWED_CONNECT_HOSTS == frozenset(
        {"api.openai.com", "auth.openai.com", "chatgpt.com"}
    )
    assert proxy.parse_connect_target("chatgpt.com:443") == ("chatgpt.com", 443)
    for target in ("github.com:443", "chatgpt.com:80", "example.com:22"):
        with pytest.raises(PermissionError):
            proxy.parse_connect_target(target)
    for target in ("1.1.1.1:443", "[2606:4700::1111]:443", "chatgpt.com"):
        with pytest.raises(ValueError):
            proxy.parse_connect_target(target)


def test_public_mode_admits_hosts_and_literals_on_web_ports_only() -> None:
    """The flag widens the syntax check to any host or IP on port 80/443."""
    parse = proxy.parse_connect_target
    assert parse("GitHub.com.:443", allow_public_egress=True) == ("github.com", 443)
    assert parse("example.com:80", allow_public_egress=True) == ("example.com", 80)
    assert parse("1.1.1.1:443", allow_public_egress=True) == ("1.1.1.1", 443)
    assert parse("[2606:4700::1111]:443", allow_public_egress=True) == (
        "2606:4700::1111",
        443,
    )
    assert parse("chatgpt.com:443", allow_public_egress=True) == ("chatgpt.com", 443)
    for target in ("github.com:22", "github.com:8443", "1.1.1.1:25"):
        with pytest.raises(PermissionError):
            parse(target, allow_public_egress=True)
    for target in ("bad host!:443", "[github.com]:443", "github.com", "-x.com:443"):
        with pytest.raises(ValueError):
            parse(target, allow_public_egress=True)


@pytest.mark.parametrize("address", ["8.8.8.8", "140.82.112.3", "2606:4700:4700::1111"])
def test_global_unicast_addresses_are_public(address: str) -> None:
    """Globally routable unicast addresses pass the vetting."""
    assert proxy.is_public_address(address)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.17.0.1",
        "172.18.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "240.0.0.1",
        "255.255.255.255",
        "::1",
        "fe80::1",
        "fc00::1",
        "ff02::1",
        "::ffff:8.8.8.8",
        "2002:7f00:1::",
        "64:ff9b::7f00:1",
        "not-an-ip",
    ],
)
def test_non_global_addresses_are_refused(address: str) -> None:
    """Loopback, private, bridge, link-local, CGNAT, multicast and reserved fail."""
    assert not proxy.is_public_address(address)


def test_resolution_refuses_any_non_public_answer(monkeypatch) -> None:
    """One internal answer among public ones refuses the whole name."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        _fake_getaddrinfo(
            {
                "public.test": ["2606:4700:4700::1111", "1.1.1.1"],
                "mixed.test": ["1.1.1.1", "10.0.0.5"],
                "bridge.test": ["172.17.0.1"],
            }
        ),
    )
    resolve = proxy.resolve_public_addresses
    assert asyncio.run(resolve("public.test", 443)) == ["1.1.1.1", "2606:4700:4700::1111"]
    for host in ("mixed.test", "bridge.test"):
        with pytest.raises(PermissionError):
            asyncio.run(resolve(host, 443))


def test_dns_rebinding_is_guarded_by_dialling_the_vetted_address(monkeypatch) -> None:
    """The name is resolved once; the tunnel dials that vetted IP, not the name."""
    answers = iter([["93.184.215.14"], ["127.0.0.1"], ["127.0.0.1"]])
    lookups: list[str] = []
    dialled: list[tuple[str, int]] = []
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        _fake_getaddrinfo({"rebind.test": lambda: next(answers)}, lookups),
    )

    async def fake_open_connection(host, port, **kwargs):
        dialled.append((host, port))
        return "reader", "writer"

    monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)
    result = asyncio.run(
        proxy.open_upstream("rebind.test", 443, allow_public_egress=True)
    )
    assert result == ("reader", "writer")
    assert lookups == ["rebind.test"]
    assert dialled == [("93.184.215.14", 443)]


def test_allowlisted_hosts_are_dialled_by_name_in_both_modes(monkeypatch) -> None:
    """The model-auth route is unchanged by the flag."""
    dialled: list[tuple[str, int]] = []

    async def fake_open_connection(host, port, **kwargs):
        dialled.append((host, port))
        return "reader", "writer"

    monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)
    for allow in (False, True):
        asyncio.run(proxy.open_upstream("chatgpt.com", 443, allow_public_egress=allow))
    assert dialled == [("chatgpt.com", 443), ("chatgpt.com", 443)]


@pytest.mark.parametrize(
    "target",
    [
        "127.0.0.1:443",
        "172.17.0.1:443",
        "10.0.0.1:80",
        "169.254.169.254:80",
        "[::1]:443",
        "internal.test:443",
        "github.com:22",
    ],
)
def test_public_mode_answers_403_for_internal_targets_and_other_ports(
    monkeypatch, target: str
) -> None:
    """Refusals stay 403 on the wire in public-egress mode."""
    monkeypatch.setattr(
        socket, "getaddrinfo", _fake_getaddrinfo({"internal.test": ["192.168.0.10"]})
    )
    status = asyncio.run(_connect_status(target, allow_public_egress=True))
    assert status.startswith(b"HTTP/1.1 403 ")


def test_default_mode_answers_403_for_non_allowlisted_host() -> None:
    """The default mode still refuses an ordinary public host."""
    status = asyncio.run(_connect_status("github.com:443", allow_public_egress=False))
    assert status.startswith(b"HTTP/1.1 403 ")


def test_public_mode_tunnels_to_a_vetted_public_address(monkeypatch) -> None:
    """A public name is tunnelled (200) and the bytes reach the vetted upstream."""
    real_open_connection = asyncio.open_connection
    dialled: list[tuple[str, int]] = []

    async def scenario() -> tuple[bytes, bytes]:
        async def echo(reader, writer):
            writer.write(await reader.readline())
            await writer.drain()
            writer.close()

        upstream = await asyncio.start_server(echo, "127.0.0.1", 0)
        upstream_port = upstream.sockets[0].getsockname()[1]

        async def fake_open_connection(host, port, **kwargs):
            dialled.append((host, port))
            return await real_open_connection("127.0.0.1", upstream_port)

        monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)
        server = await asyncio.start_server(
            lambda reader, writer: proxy.handle_connect(
                reader, writer, allow_public_egress=True
            ),
            "127.0.0.1",
            0,
        )
        port = server.sockets[0].getsockname()[1]
        async with upstream, server:
            reader, writer = await real_open_connection("127.0.0.1", port)
            writer.write(b"CONNECT docs.test:443 HTTP/1.1\r\nHost: docs.test:443\r\n\r\n")
            await writer.drain()
            status = await asyncio.wait_for(reader.readline(), timeout=10.0)
            await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10.0)
            writer.write(b"ping\n")
            await writer.drain()
            echoed = await asyncio.wait_for(reader.readline(), timeout=10.0)
            writer.close()
        return status, echoed

    monkeypatch.setattr(
        socket, "getaddrinfo", _fake_getaddrinfo({"docs.test": ["140.82.112.3"]})
    )
    status, echoed = asyncio.run(scenario())
    assert status.startswith(b"HTTP/1.1 200 ")
    assert echoed == b"ping\n"
    assert dialled == [("140.82.112.3", 443)]



ENTRY_NAMES = ("sforge_isolated_entry.py", "sforge_h100_entry.py", "sforge_entry.py")


@pytest.fixture
def entry(monkeypatch):
    """Import the task's SForge entry; undo its import-time SForge patches after."""
    import os

    from sforge.harness import run_agent
    from sforge.harness.agent.codex import CodexAgent
    from sforge.harness.backend.docker_backend import DockerBackend

    for owner, attribute in (
        (DockerBackend, "create_container"),
        (CodexAgent, "format_run_cmd"),
        (CodexAgent, "install_cmds"),
        (run_agent, "generate_evolve_prompt"),
    ):
        monkeypatch.setattr(owner, attribute, getattr(owner, attribute))
    monkeypatch.setenv(
        "SFORGE_CODEX_CLI_VERSION", os.environ.get("SFORGE_CODEX_CLI_VERSION", "0.148.0")
    )
    paths = [PROXY_PATH.parent / name for name in ENTRY_NAMES if (PROXY_PATH.parent / name).is_file()]
    assert len(paths) == 1, paths
    spec = importlib.util.spec_from_file_location("task_entry_under_test", paths[0])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "flag,web_search,offline_prompt",
    [(None, "live", False), ("true", "live", False), (" TRUE ", "live", False), ("false", "disabled", True)],
)
def test_codex_and_prompt_follow_the_task_internet_flag(
    entry, monkeypatch, tmp_path, flag, web_search, offline_prompt
) -> None:
    """MLSYSFORGE_WORK_INTERNET: unset/true pass through, false forces offline."""
    from sforge.harness.config import SForgeConfig

    # The launcher always passes --enable-internet, so the caller asks for
    # internet=True; the task JSON's flag arrives through the environment.
    if flag is None:
        monkeypatch.delenv(entry.WORK_INTERNET_ENV, raising=False)
    else:
        monkeypatch.setenv(entry.WORK_INTERNET_ENV, flag)
    assert entry.WORK_INTERNET_ENV == "MLSYSFORGE_WORK_INTERNET"
    assert entry.CodexAgent.format_run_cmd is entry.task_internet_format_run_cmd
    assert entry.run_agent.generate_evolve_prompt is entry.task_internet_generate_prompt
    agent = entry.CodexAgent(SForgeConfig(codex_auth_json_path=tmp_path / "auth.json"))
    command = entry.task_internet_format_run_cmd(agent, "/tmp/prompt.md", internet=True)
    assert f'web_search="{web_search}"' in command
    prompt = entry.task_internet_generate_prompt(
        "Optimize the kernel.", submit_paths=["lab/solutions"], internet=True
    )
    assert ("NO internet access" in prompt) is offline_prompt
    if not offline_prompt:
        assert "internet" not in prompt.lower()


def test_unset_flag_passes_an_offline_caller_through(entry, monkeypatch, tmp_path) -> None:
    """Pass-through means the caller's own internet=False is also kept."""
    from sforge.harness.config import SForgeConfig

    monkeypatch.delenv(entry.WORK_INTERNET_ENV, raising=False)
    agent = entry.CodexAgent(SForgeConfig(codex_auth_json_path=tmp_path / "auth.json"))
    command = entry.task_internet_format_run_cmd(agent, "/tmp/prompt.md", internet=False)
    assert 'web_search="disabled"' in command


@pytest.mark.parametrize("flag", ["1", "yes", "off", "live"])
def test_invalid_task_internet_flag_is_rejected(entry, monkeypatch, tmp_path, flag) -> None:
    """Anything but true/false/unset fails closed."""
    from sforge.harness.config import SForgeConfig

    monkeypatch.setenv(entry.WORK_INTERNET_ENV, flag)
    agent = entry.CodexAgent(SForgeConfig(codex_auth_json_path=tmp_path / "auth.json"))
    with pytest.raises(RuntimeError, match=entry.WORK_INTERNET_ENV):
        entry.task_internet_format_run_cmd(agent, "/tmp/prompt.md", internet=True)
    with pytest.raises(RuntimeError, match=entry.WORK_INTERNET_ENV):
        entry.task_internet_generate_prompt("Optimize the kernel.", internet=True)


def test_entry_never_hard_codes_the_internet_flag() -> None:
    """The old forced-offline overrides are gone from the entry source."""
    for name in ENTRY_NAMES:
        path = PROXY_PATH.parent / name
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert 'kwargs["internet"] = False' not in text
            assert 'kwargs["internet"] = True' not in text


def test_launcher_derives_the_sidecar_mode_from_the_task_json() -> None:
    """internet=true adds --allow-public-egress and the matching preflight."""
    text = (PROXY_PATH.parent / "run_fixed_budget.sh").read_text(encoding="utf-8")
    assert "task_internet=\"$(jq -r '.internet' \"${TASKS_DIR}/${TASK_ID}.json\")\"" in text
    assert 'true) sidecar_public_egress="--allow-public-egress"; codex_web_search="live" ;;' in text
    assert 'false) sidecar_public_egress=""; codex_web_search="disabled" ;;' in text
    assert '  ${sidecar_public_egress:+"${sidecar_public_egress}"} \\\n' in text
    assert 'export MLSYSFORGE_WORK_INTERNET="${task_internet}"' in text
    # Allowlist mode is unchanged; public mode needs github 200 and the
    # (non-global) Docker bridge gateway 403.
    assert 'probe("chatgpt.com", b" 200 ")' in text
    assert '    probe("github.com", b" 403 ")' in text
    assert '    probe("github.com", b" 200 ")' in text
    assert '    probe(sys.argv[3], b" 403 ")' in text
    assert '"${restricted_proxy_ip}" "${task_internet}" "${bridge_gateway}"' in text
    assert "Public-egress network: PASS" in text
    assert "Restricted network: PASS" in text
    assert "-fsS https://example.com" in text
    assert "direct public egress" in text
    assert '--arg codex_web_search "${codex_web_search}"' in text
    assert '--argjson public_connect_egress "${task_internet}"' in text
    assert "--enable-internet" in text
