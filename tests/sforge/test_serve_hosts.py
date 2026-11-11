"""``sforge serve --host`` may name several explicit host-local addresses."""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from sforge import cli


def test_serve_hosts_default_and_deduplication() -> None:
    assert cli._serve_hosts(None) == ["0.0.0.0"]
    assert cli._serve_hosts("172.17.0.1") == ["172.17.0.1"]
    assert cli._serve_hosts(["172.17.0.1", "127.0.0.1", "172.17.0.1"]) == [
        "172.17.0.1",
        "127.0.0.1",
    ]
    with pytest.raises(ValueError, match="must not be empty"):
        cli._serve_hosts(["172.17.0.1", " "])
    with pytest.raises(ValueError, match="at least one"):
        cli._serve_hosts([])


def test_bind_serve_socket_binds_the_exact_address() -> None:
    sock = cli._bind_serve_socket("127.0.0.1", 0)
    try:
        host, port = sock.getsockname()[:2]
        assert host == "127.0.0.1" and port > 0
        assert sock.get_inheritable() is True
        sock.listen(1)
        with pytest.raises(RuntimeError, match="cannot bind judge server"):
            cli._bind_serve_socket("127.0.0.1", port)
    finally:
        sock.close()


def _wire(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}
    monkeypatch.setattr(cli, "_make_config", lambda args: SimpleNamespace())
    fake_app = object()
    monkeypatch.setattr(
        "sforge.harness.judge_server.create_app", lambda config: fake_app
    )

    class FakeConfig:
        def __init__(self, app, port):
            calls["config"] = (app, port)

    class FakeServer:
        def __init__(self, config):
            calls["server_config"] = config

        def run(self, sockets=None):
            calls["sockets"] = sockets

    fake_uvicorn = SimpleNamespace(
        run=lambda app, host, port: calls.__setitem__("run", (app, host, port)),
        Config=FakeConfig,
        Server=FakeServer,
    )
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(
        cli, "_bind_serve_socket", lambda host, port: f"socket:{host}:{port}"
    )
    return calls, fake_app


def test_serve_single_host_uses_uvicorn_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, app = _wire(monkeypatch)
    cli.cmd_serve(SimpleNamespace(host=["172.17.0.1"], port=18384))
    assert calls["run"] == (app, "172.17.0.1", 18384)
    assert "sockets" not in calls


def test_serve_several_hosts_binds_each_explicit_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, app = _wire(monkeypatch)
    cli.cmd_serve(SimpleNamespace(host=["172.17.0.1", "127.0.0.1"], port=18384))
    assert "run" not in calls
    assert calls["config"] == (app, 18384)
    assert calls["sockets"] == ["socket:172.17.0.1:18384", "socket:127.0.0.1:18384"]


def test_serve_parser_accepts_repeated_host(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    monkeypatch.setattr(cli, "cmd_serve", lambda args: seen.append(args))
    monkeypatch.setattr(
        "sys.argv",
        ["sforge", "serve", "--host", "172.17.0.1", "--host", "127.0.0.1", "--port", "18384"],
    )
    cli.main()
    monkeypatch.setattr("sys.argv", ["sforge", "serve"])
    cli.main()
    assert seen[0].host == ["172.17.0.1", "127.0.0.1"] and seen[0].port == 18384
    assert seen[1].host is None and seen[1].port == 8080
