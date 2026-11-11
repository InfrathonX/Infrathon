from __future__ import annotations

import os
import subprocess
from pathlib import Path

from sforge.harness.evolve_scripts import generate_submit_script


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _run_submit(
    tmp_path: Path,
    *,
    hook: str,
    args: tuple[str, ...] = (),
) -> tuple[subprocess.CompletedProcess[str], Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    transport_marker = tmp_path / "transport-was-reached"
    for command in ("tar", "curl"):
        _write_executable(
            fake_bin / command,
            "#!/bin/sh\n"
            f"printf '%s\\n' {command} >> \"{transport_marker}\"\n"
            "exit 99\n",
        )
    submit = tmp_path / "sforge-submit"
    _write_executable(submit, generate_submit_script())
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "SFORGE_JUDGE_URL": "http://127.0.0.1:9",
        "SFORGE_TOKEN": "test-token",
        "SFORGE_PATCH_DIR": str(workspace),
        "SFORGE_PRE_SUBMIT_HOOK": hook,
    }
    result = subprocess.run(
        ["bash", str(submit), *args],
        cwd=workspace,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, transport_marker


def test_failing_pre_submit_hook_stops_before_archive_or_http(tmp_path: Path) -> None:
    hook = tmp_path / "workspace" / "cleanup.sh"
    hook.parent.mkdir()
    _write_executable(hook, "#!/bin/sh\nexit 73\n")

    result, transport_marker = _run_submit(tmp_path, hook=str(hook))

    assert result.returncode == 73
    assert "submission cancelled" in result.stderr
    assert not transport_marker.exists()


def test_pre_submit_hook_rejects_relative_path_before_execution(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    relative_hook = workspace / "cleanup.sh"
    _write_executable(relative_hook, "#!/bin/sh\nexit 0\n")

    result, transport_marker = _run_submit(tmp_path, hook="cleanup.sh")

    assert result.returncode == 64
    assert "must be an absolute path" in result.stderr
    assert not transport_marker.exists()


def test_pre_submit_hook_must_resolve_inside_patch_directory(
    tmp_path: Path,
) -> None:
    outside_hook = tmp_path / "outside.sh"
    _write_executable(outside_hook, "#!/bin/sh\nexit 0\n")

    result, transport_marker = _run_submit(tmp_path, hook=str(outside_hook))

    assert result.returncode == 64
    assert "must resolve inside SFORGE_PATCH_DIR" in result.stderr
    assert not transport_marker.exists()


def test_pre_submit_hook_rejects_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "cleanup-target.sh"
    _write_executable(target, "#!/bin/sh\nexit 0\n")
    hook = workspace / "cleanup.sh"
    hook.symlink_to(target)

    result, transport_marker = _run_submit(tmp_path, hook=str(hook))

    assert result.returncode == 64
    assert "regular, non-symlink executable" in result.stderr
    assert not transport_marker.exists()


def test_unset_pre_submit_hook_preserves_archive_path(tmp_path: Path) -> None:
    result, transport_marker = _run_submit(tmp_path, hook="")

    assert result.returncode == 99
    assert transport_marker.read_text(encoding="utf-8") == "tar\n"


def test_list_mode_does_not_run_pre_submit_hook(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    hook_marker = tmp_path / "hook-was-reached"
    hook = workspace / "cleanup.sh"
    _write_executable(
        hook,
        "#!/bin/sh\n"
        f"touch \"{hook_marker}\"\n"
        "exit 88\n",
    )

    result, transport_marker = _run_submit(
        tmp_path,
        hook=str(hook),
        args=("--list",),
    )

    assert result.returncode != 88
    assert not hook_marker.exists()
    assert transport_marker.read_text(encoding="utf-8") == "curl\n"


def test_pre_submit_hook_is_immediately_before_archive_creation() -> None:
    script = generate_submit_script()

    hook_call = script.index('"$HOOK_REAL" || {')
    archive_creation = script.index("ARCHIVE_FILE=$(mktemp", hook_call)
    archive_transport = script.index('curl -s -o /tmp/_submit_resp.json')
    assert hook_call < archive_creation < archive_transport
