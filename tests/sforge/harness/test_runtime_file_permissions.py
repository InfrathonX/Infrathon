from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import pytest

from sforge.harness.backend import StreamingExecResult
from sforge.harness.container_files import copy_container_file


class _PermissionBackend:
    def __init__(self, *, exit_code: int = 0, timed_out: bool = False) -> None:
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.copies: list[tuple[Path, PurePosixPath]] = []
        self.commands: list[tuple[object, int | None, str | None]] = []

    def copy_to_container(self, _handle, source: Path, destination: PurePosixPath) -> None:
        self.copies.append((source, destination))

    def exec_run_with_exit_code(
        self,
        _handle,
        command,
        timeout: int | None = None,
        *,
        user: str | None = None,
    ) -> StreamingExecResult:
        self.commands.append((command, timeout, user))
        return StreamingExecResult(
            output="permission failure" if self.exit_code else "",
            exit_code=self.exit_code,
            timed_out=self.timed_out,
            elapsed_seconds=0.01,
        )


def test_copy_container_file_normalizes_umask_077_metadata(tmp_path: Path) -> None:
    previous = os.umask(0o077)
    try:
        source = tmp_path / "agent_prompt.md"
        source.write_text("authoritative task prompt", encoding="utf-8")
    finally:
        os.umask(previous)
    assert source.stat().st_mode & 0o777 == 0o600

    backend = _PermissionBackend()
    destination = PurePosixPath("/tmp/agent_prompt.md")
    copy_container_file(
        backend,
        object(),
        source,
        destination,
        owner="agent:agent",
        mode=0o600,
        description="agent prompt",
    )

    assert backend.copies == [(source, destination)]
    command, timeout, user = backend.commands[0]
    assert timeout == 30
    assert user == "root"
    assert command[:2] == ["/bin/sh", "-c"]
    assert "chown -- agent:agent /tmp/agent_prompt.md" in command[2]
    assert "chmod -- 0600 /tmp/agent_prompt.md" in command[2]
    assert "stat -c '%U:%G:%a'" in command[2]
    assert 'test "$actual" = agent:agent:600' in command[2]


@pytest.mark.parametrize(
    ("exit_code", "timed_out"),
    ((17, False), (0, True)),
)
def test_copy_container_file_fails_closed_on_permission_transition(
    tmp_path: Path,
    exit_code: int,
    timed_out: bool,
) -> None:
    source = tmp_path / "runtime-file"
    source.write_text("payload", encoding="utf-8")
    backend = _PermissionBackend(exit_code=exit_code, timed_out=timed_out)

    with pytest.raises(RuntimeError, match="Failed to secure test runtime file"):
        copy_container_file(
            backend,
            object(),
            source,
            PurePosixPath("/usr/local/bin/sforge-submit"),
            owner="root:root",
            mode=0o755,
            description="test runtime file",
        )


@pytest.mark.parametrize(
    ("destination", "owner", "mode"),
    (
        (PurePosixPath("relative"), "root:root", 0o644),
        (PurePosixPath("/tmp/file"), "1000:1000", 0o644),
        (PurePosixPath("/tmp/file"), "root:root", 0o1000),
    ),
)
def test_copy_container_file_rejects_unsafe_metadata_contract(
    tmp_path: Path,
    destination: PurePosixPath,
    owner: str,
    mode: int,
) -> None:
    source = tmp_path / "runtime-file"
    source.write_text("payload", encoding="utf-8")

    with pytest.raises(ValueError):
        copy_container_file(
            _PermissionBackend(),
            object(),
            source,
            destination,
            owner=owner,
            mode=mode,
            description="test runtime file",
        )
