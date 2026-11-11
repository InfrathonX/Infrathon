"""Contract tests for host-provided agent runtimes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sforge.harness.agent_runtime import (
    RUNTIME_ENV_FILE,
    RUNTIME_MARKER_NAME,
    RUNTIME_MOUNT_TARGET,
    AgentRuntimeError,
    load_agent_runtime,
    runtime_environment,
    runtime_probe_cmds,
)
from sforge.harness.backend.docker_backend import DockerBackend


def _write_runtime(
    root: Path,
    *,
    marker_overrides: dict | None = None,
    cli: str = "codex",
    with_env: bool = True,
    with_cli: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    marker = {
        "schema_version": 1,
        "kind": "sforge-agent-runtime",
        "agent": "codex",
        "cli": cli,
        "version": "0.148.0",
        "node_version": "v20.18.0",
        "tree_sha256": "a" * 64,
    }
    marker.update(marker_overrides or {})
    (root / RUNTIME_MARKER_NAME).write_text(
        json.dumps(marker) + "\n", encoding="utf-8"
    )
    if with_env:
        (root / RUNTIME_ENV_FILE).write_text(
            f'export PATH="{RUNTIME_MOUNT_TARGET}/bin:${{PATH}}"\n',
            encoding="utf-8",
        )
    if with_cli:
        (root / "bin").mkdir(exist_ok=True)
        cli_path = root / "bin" / cli
        cli_path.write_text("#!/bin/sh\necho 0.148.0\n", encoding="utf-8")
        cli_path.chmod(0o755)
    return root


def test_load_returns_marker_identity_and_mount_facts(tmp_path: Path) -> None:
    root = _write_runtime(tmp_path / "codex-0.148.0")

    runtime = load_agent_runtime(root)

    assert runtime.agent == "codex"
    assert runtime.cli == "codex"
    assert runtime.version == "0.148.0"
    assert runtime.node_version == "v20.18.0"
    assert runtime.tree_sha256 == "a" * 64
    assert runtime.mount_target == RUNTIME_MOUNT_TARGET
    assert runtime.env_file == f"{RUNTIME_MOUNT_TARGET}/{RUNTIME_ENV_FILE}"
    described = runtime.describe()
    assert described["root"] == str(root.resolve())
    assert described["version"] == "0.148.0"


def test_load_accepts_a_cli_symlink_inside_the_runtime(tmp_path: Path) -> None:
    root = _write_runtime(tmp_path / "runtime", with_cli=False)
    (root / "lib").mkdir()
    real = root / "lib" / "codex.js"
    real.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    real.chmod(0o755)
    (root / "bin").mkdir()
    (root / "bin" / "codex").symlink_to(Path("..") / "lib" / "codex.js")

    assert load_agent_runtime(root).cli == "codex"


def test_load_rejects_a_cli_symlink_escaping_the_runtime(tmp_path: Path) -> None:
    outside = tmp_path / "outside-cli"
    outside.write_text("#!/bin/sh\n", encoding="utf-8")
    outside.chmod(0o755)
    root = _write_runtime(tmp_path / "runtime", with_cli=False)
    (root / "bin").mkdir()
    (root / "bin" / "codex").symlink_to(outside)

    with pytest.raises(AgentRuntimeError, match="escapes"):
        load_agent_runtime(root)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"kind": "something-else"},
        {"agent": "Bad Agent"},
        {"agent": None},
        {"cli": "../evil"},
        {"version": ""},
        {"node_version": "spaces here"},
        {"tree_sha256": "zz"},
    ],
)
def test_load_rejects_invalid_markers(tmp_path: Path, overrides: dict) -> None:
    root = _write_runtime(tmp_path / "runtime", marker_overrides=overrides)
    with pytest.raises(AgentRuntimeError):
        load_agent_runtime(root)


def test_load_rejects_missing_pieces(tmp_path: Path) -> None:
    with pytest.raises(AgentRuntimeError, match="does not exist"):
        load_agent_runtime(tmp_path / "missing")

    no_env = _write_runtime(tmp_path / "no-env", with_env=False)
    with pytest.raises(AgentRuntimeError, match="env file"):
        load_agent_runtime(no_env)

    no_cli = _write_runtime(tmp_path / "no-cli", with_cli=False)
    with pytest.raises(AgentRuntimeError, match="CLI is missing"):
        load_agent_runtime(no_cli)

    no_marker = tmp_path / "no-marker"
    no_marker.mkdir()
    with pytest.raises(AgentRuntimeError, match="marker is missing"):
        load_agent_runtime(no_marker)


def test_probe_replaces_the_network_install(tmp_path: Path) -> None:
    runtime = load_agent_runtime(_write_runtime(tmp_path / "runtime"))

    cmds = runtime_probe_cmds(runtime)

    assert len(cmds) == 1
    assert f"test -x {RUNTIME_MOUNT_TARGET}/bin/codex" in cmds[0]
    assert 'codex --version | grep -F "0.148.0"' in cmds[0]
    assert "npm install" not in cmds[0]
    assert "sudo" not in cmds[0]


def test_runtime_environment_targets_every_bash_exec(tmp_path: Path) -> None:
    runtime = load_agent_runtime(_write_runtime(tmp_path / "runtime"))

    env = runtime_environment(runtime)

    assert env["BASH_ENV"] == f"{RUNTIME_MOUNT_TARGET}/{RUNTIME_ENV_FILE}"
    assert env["SFORGE_AGENT_RUNTIME"] == "codex@0.148.0"


def test_docker_backend_translates_mount_triples(tmp_path: Path) -> None:
    captured: dict = {}

    class _Containers:
        def create(self, image, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(id="cid", name=kwargs.get("name"))

    backend = DockerBackend.__new__(DockerBackend)
    backend._client = SimpleNamespace(containers=_Containers())

    backend.create_container(
        "image:tag",
        "container-name",
        mounts=[(str(tmp_path), RUNTIME_MOUNT_TARGET, "ro")],
    )

    assert captured["volumes"] == {
        str(tmp_path): {"bind": RUNTIME_MOUNT_TARGET, "mode": "ro"}
    }


def test_docker_backend_omits_volumes_without_mounts() -> None:
    captured: dict = {}

    class _Containers:
        def create(self, image, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(id="cid", name=kwargs.get("name"))

    backend = DockerBackend.__new__(DockerBackend)
    backend._client = SimpleNamespace(containers=_Containers())

    backend.create_container("image:tag", "container-name")

    assert "volumes" not in captured
