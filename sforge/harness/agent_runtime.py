"""Host-provided agent runtimes mounted into work containers.

A task image describes the *task* environment (CUDA, frameworks, the staged
contract).  Which agent harness drives it — Codex, Claude Code, and at which
CLI version — is a launch-time variable.  An *agent runtime* is a
self-contained host directory (Node plus one agent CLI installed under a
single prefix) that the docker backend bind-mounts read-only into the work
container at a fixed target.  Selecting an agent or bumping its version then
never requires rebuilding any task image.

Layout of a runtime directory::

    <runtime>/
      .sforge-agent-runtime.json   # marker, see REQUIRED_MARKER_FIELDS
      env.sh                       # prepends <target>/bin to PATH (via BASH_ENV)
      bin/node, bin/<cli>          # self-contained toolchain
      lib/node_modules/...

The marker's ``agent`` field must equal the SForge agent name the run uses
(``codex``, ``claude-code``); ``cli`` names the executable under ``bin/``.
When a runtime is configured, the agent's network install commands are
replaced by an offline probe of the mounted toolchain, so restricted-network
tasks need no baked-in agent and no install shim.
"""

from __future__ import annotations

import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUNTIME_MARKER_NAME = ".sforge-agent-runtime.json"
RUNTIME_ENV_FILE = "env.sh"
RUNTIME_MOUNT_TARGET = "/opt/agent-runtime"
RUNTIME_SCHEMA_VERSION = 1

_NAME_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class AgentRuntimeError(RuntimeError):
    """A configured agent runtime directory is missing or malformed."""


@dataclass(frozen=True)
class AgentRuntime:
    root: Path
    agent: str
    cli: str
    version: str
    node_version: str | None
    tree_sha256: str | None

    @property
    def mount_target(self) -> str:
        return RUNTIME_MOUNT_TARGET

    @property
    def env_file(self) -> str:
        return f"{RUNTIME_MOUNT_TARGET}/{RUNTIME_ENV_FILE}"

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "agent": self.agent,
            "cli": self.cli,
            "version": self.version,
            "node_version": self.node_version,
            "tree_sha256": self.tree_sha256,
            "mount_target": self.mount_target,
        }


def _marker_string(marker: dict[str, Any], field: str, pattern: re.Pattern) -> str:
    value = marker.get(field)
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise AgentRuntimeError(f"agent runtime marker field is invalid: {field}")
    return value


def load_agent_runtime(path: Path) -> AgentRuntime:
    """Validate one runtime directory and return its identity."""
    requested = Path(path).expanduser()
    try:
        root = requested.resolve(strict=True)
    except OSError as exc:
        raise AgentRuntimeError(
            f"agent runtime directory does not exist: {requested}"
        ) from exc
    if not root.is_dir():
        raise AgentRuntimeError(f"agent runtime is not a directory: {root}")

    marker_path = root / RUNTIME_MARKER_NAME
    try:
        marker_stat = marker_path.lstat()
    except OSError as exc:
        raise AgentRuntimeError(
            f"agent runtime marker is missing: {marker_path}"
        ) from exc
    if not stat.S_ISREG(marker_stat.st_mode):
        raise AgentRuntimeError(
            f"agent runtime marker must be a regular file: {marker_path}"
        )
    try:
        marker = json.loads(
            marker_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise AgentRuntimeError(
            f"agent runtime marker is malformed: {marker_path}"
        ) from exc
    if (
        not isinstance(marker, dict)
        or marker.get("schema_version") != RUNTIME_SCHEMA_VERSION
        or marker.get("kind") != "sforge-agent-runtime"
    ):
        raise AgentRuntimeError(f"invalid agent runtime marker: {marker_path}")

    agent = _marker_string(marker, "agent", _NAME_RE)
    cli = _marker_string(marker, "cli", _NAME_RE)
    version = _marker_string(marker, "version", _VERSION_RE)
    node_version = marker.get("node_version")
    if node_version is not None and (
        not isinstance(node_version, str)
        or _VERSION_RE.fullmatch(node_version) is None
    ):
        raise AgentRuntimeError("agent runtime marker field is invalid: node_version")
    tree_sha256 = marker.get("tree_sha256")
    if tree_sha256 is not None and (
        not isinstance(tree_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", tree_sha256) is None
    ):
        raise AgentRuntimeError("agent runtime marker field is invalid: tree_sha256")

    env_path = root / RUNTIME_ENV_FILE
    if not env_path.is_file():
        raise AgentRuntimeError(f"agent runtime env file is missing: {env_path}")

    cli_path = root / "bin" / cli
    try:
        resolved_cli = cli_path.resolve(strict=True)
    except OSError as exc:
        raise AgentRuntimeError(
            f"agent runtime CLI is missing: {cli_path}"
        ) from exc
    if not resolved_cli.is_relative_to(root):
        raise AgentRuntimeError(
            f"agent runtime CLI escapes the runtime directory: {cli_path}"
        )

    return AgentRuntime(
        root=root,
        agent=agent,
        cli=cli,
        version=version,
        node_version=node_version if isinstance(node_version, str) else None,
        tree_sha256=tree_sha256 if isinstance(tree_sha256, str) else None,
    )


def runtime_probe_cmds(runtime: AgentRuntime) -> list[str]:
    """Offline install-step replacement: verify the mounted toolchain.

    The container gets ``BASH_ENV=<target>/env.sh`` so every ``bash -c``
    exec already has the runtime on PATH; the probe just proves the mount is
    present and the CLI reports the pinned version.
    """
    target = RUNTIME_MOUNT_TARGET
    return [
        f"test -x {target}/bin/{runtime.cli} && "
        f"{runtime.cli} --version | grep -F {json.dumps(runtime.version)}"
    ]


def runtime_environment(runtime: AgentRuntime) -> dict[str, str]:
    """Environment entries the work container needs for the mounted runtime."""
    return {
        "BASH_ENV": runtime.env_file,
        "SFORGE_AGENT_RUNTIME": f"{runtime.agent}@{runtime.version}",
    }
