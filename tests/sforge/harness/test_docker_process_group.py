from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from sforge.harness.backend import docker_backend
from sforge.harness.backend.docker_backend import (
    DockerBackend,
    DockerContainerHandle,
)


class _FakeExecAPI:
    def __init__(self, container: "_FakeContainer") -> None:
        self.container = container
        self.created_command = None

    def exec_create(self, container_id, command, **kwargs):
        self.created_command = command
        return {"Id": "exec-id"}

    def exec_start(self, exec_id, stream=True):
        self.container.stream_done.wait(timeout=2)
        if False:
            yield b""

    def exec_inspect(self, exec_id):
        return {"Pid": self.container.exec_pid, "ExitCode": 0}


class _FakeContainer:
    def __init__(
        self,
        *,
        term_kills_group: bool,
        kill_kills_group: bool = True,
    ) -> None:
        self.id = "container-id"
        self.name = "container"
        self.exec_pid = 4242
        self.group_alive = True
        self.term_kills_group = term_kills_group
        self.kill_kills_group = kill_kills_group
        self.stream_done = threading.Event()
        self.commands: list[str] = []
        self.client = SimpleNamespace(api=_FakeExecAPI(self))

    def exec_run(self, command, **kwargs):
        shell = command[-1] if isinstance(command, list) else str(command)
        self.commands.append(shell)
        if shell.startswith("kill -0 ") or "ps -eo pgid=,stat=" in shell:
            return SimpleNamespace(
                exit_code=0 if self.group_alive else 1
            )
        if shell.startswith("kill -TERM "):
            # Model the exec leader closing its stream while a descendant may
            # remain alive in the same process group.
            self.stream_done.set()
            if self.term_kills_group:
                self.group_alive = False
            return SimpleNamespace(exit_code=0)
        if shell.startswith("kill -KILL "):
            self.stream_done.set()
            if self.kill_kills_group:
                self.group_alive = False
            return SimpleNamespace(exit_code=0)
        raise AssertionError(f"unexpected exec command: {shell}")


def _run_timed_exec(
    monkeypatch: pytest.MonkeyPatch,
    container: _FakeContainer,
):
    monkeypatch.setattr(
        docker_backend, "EXEC_TERM_GRACE_SECONDS", 0.01
    )
    monkeypatch.setattr(
        docker_backend, "EXEC_KILL_GRACE_SECONDS", 0.01
    )
    backend = DockerBackend(client=SimpleNamespace())
    handle = DockerContainerHandle(container)
    return backend.exec_run_with_timeout(
        handle,
        ["setsid", "/bin/bash", "-c", "agent"],
        timeout=0.001,
    )


def _sent_signals(container: _FakeContainer) -> list[str]:
    return [
        command.split()[1]
        for command in container.commands
        if command.startswith(("kill -TERM ", "kill -KILL "))
    ]


def test_timeout_terms_process_group_and_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _FakeContainer(term_kills_group=True)

    result = _run_timed_exec(monkeypatch, container)

    assert result.timed_out is True
    assert result.exit_code == -1
    assert container.group_alive is False
    assert container.stream_done.is_set()
    assert _sent_signals(container) == ["-TERM"]


def test_timeout_kills_descendant_after_leader_stream_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _FakeContainer(term_kills_group=False)

    result = _run_timed_exec(monkeypatch, container)

    assert result.timed_out is True
    assert result.exit_code == -1
    assert container.group_alive is False
    assert _sent_signals(container) == ["-TERM", "-KILL"]


def test_timeout_fails_if_group_survives_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _FakeContainer(
        term_kills_group=False,
        kill_kills_group=False,
    )

    with pytest.raises(
        RuntimeError,
        match="did not stop after TERM/KILL",
    ):
        _run_timed_exec(monkeypatch, container)

    assert _sent_signals(container) == ["-TERM", "-KILL"]
