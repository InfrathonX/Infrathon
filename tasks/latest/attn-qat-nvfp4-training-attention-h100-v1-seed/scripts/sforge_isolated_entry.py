#!/usr/bin/env python3
"""Run SForge Work containers on a pre-created internal Docker network.

This avoids the host-iptables dependency of SForge's standard local isolation.
The task launcher (run_fixed_budget.sh) supplies an HTTP CONNECT sidecar on the
same internal network, and the Work container has no direct external route:
the launcher passes ``--enable-internet`` so the harness attaches a network at
all, but every container created through this entry lands on the launcher's
``--internal`` network, where only the sidecar is reachable.  With the task's
``internet: true`` the sidecar runs with ``--allow-public-egress`` (public
hosts on ports 80/443; internal addresses refused), and the Codex command
and evolve prompt follow the task JSON via MLSYSFORGE_WORK_INTERNET (see
below).  Lineage: flash-kmeans ``sforge_isolated_entry.py``.
"""

from __future__ import annotations

import os
from typing import Any

import docker

from sforge.harness.agent.codex import CodexAgent
from sforge.harness.backend import (
    AcceleratorAllocation,
    HANDOFF_BINDING_ANNOTATIONS,
)
from sforge.harness.backend.docker_backend import (
    DockerBackend,
    DockerContainerHandle,
    _validated_docker_labels,
    _validated_handoff_binding_labels,
)
from sforge.harness import run_agent


NETWORK_ENV = "SFORGE_ISOLATED_DOCKER_NETWORK"


def _validated_isolated_network(backend: DockerBackend) -> str:
    """Require the launcher-created internal network; refuse anything else."""
    network_name = os.environ.get(NETWORK_ENV, "").strip()
    if not network_name:
        raise RuntimeError(f"{NETWORK_ENV} is required")
    try:
        network = backend.client.networks.get(network_name)
        network.reload()
    except Exception as exc:
        raise RuntimeError(
            f"isolated Docker network does not exist: {network_name}"
        ) from exc
    if network.attrs.get("Internal") is not True:
        raise RuntimeError(
            f"refusing non-internal Docker network: {network_name}"
        )
    return network_name


def isolated_create_container(
    self: DockerBackend,
    image: str,
    name: str,
    *,
    command: str = "tail -f /dev/null",
    environment: dict[str, str] | None = None,
    extra_hosts: dict[str, str] | None = None,
    cap_drop: list[str] | None = None,
    cpu_limit: int | None = None,
    mem_limit: str | None = None,
    user: str | None = None,
    annotations: dict[str, str] | None = None,
    mounts: list[tuple[str, str, str]] | None = None,
    accelerator_allocation: AcceleratorAllocation | None = None,
) -> DockerContainerHandle:
    kwargs: dict[str, Any] = {"network": _validated_isolated_network(self)}
    if cpu_limit is not None:
        kwargs["nano_cpus"] = int(cpu_limit * 1e9)
    if mem_limit is not None:
        kwargs["mem_limit"] = mem_limit
    if cap_drop:
        kwargs["cap_drop"] = cap_drop
    if extra_hosts:
        kwargs["extra_hosts"] = extra_hosts
    if user:
        kwargs["user"] = user
    if annotations is not None:
        labels = _validated_docker_labels(annotations)
        if HANDOFF_BINDING_ANNOTATIONS.intersection(labels):
            labels = _validated_handoff_binding_labels(labels)
        if labels:
            kwargs["labels"] = labels
    if mounts:
        # Only SForge's host-provided agent runtime uses this path: a
        # read-only bind mount into the Work container. Nothing may ever be
        # writable from inside, and Judge containers never receive
        # caller-supplied mounts.
        if ".judge." in image or ".judge." in name:
            raise RuntimeError(
                "Judge containers accept no caller-supplied mounts"
            )
        volumes: dict[str, dict[str, str]] = {}
        for host_path, container_path, mode in mounts:
            if mode != "ro":
                raise RuntimeError(
                    "Work containers accept only read-only bind mounts"
                )
            volumes[host_path] = {"bind": container_path, "mode": mode}
        kwargs["volumes"] = volumes
    if accelerator_allocation is not None:
        if accelerator_allocation.vendor != "nvidia":
            raise ValueError(
                "isolated Docker adapter supports only NVIDIA accelerator "
                f"allocations, not {accelerator_allocation.vendor!r}"
            )
        kwargs["device_requests"] = [
            docker.types.DeviceRequest(
                driver="nvidia",
                device_ids=list(accelerator_allocation.device_ids),
                capabilities=[["gpu"]],
            )
        ]
    container = self.client.containers.create(
        image,
        name=name,
        detach=True,
        command=command,
        environment=environment or {},
        **kwargs,
    )
    return DockerContainerHandle(container)


DockerBackend.create_container = isolated_create_container

# The Work image is agent-free; SForge replaces install commands with an
# offline probe of the host-mounted runtime whenever SFORGE_AGENT_RUNTIME_DIR
# is set (the launcher requires it). This fallback covers only a legacy image
# with a baked toolchain and no configured runtime.
CodexAgent.install_cmds = [
    "test \"$(node --version)\" = v20.18.0 && "
    "codex --version | grep -F \"${SFORGE_CODEX_CLI_VERSION:-0.148.0}\""
]

# The launcher always passes --enable-internet because SForge's own
# internet=false isolation needs host iptables, which this host does not
# grant; the internal network plus the CONNECT sidecar is the enforcement
# boundary.  The task JSON's ``internet`` value reaches this entry through
# MLSYSFORGE_WORK_INTERNET ("true"/"false", exported by the launcher).
# "false" keeps the offline Codex flags and prompt; "true" (or unset) passes
# the caller's value through, so Codex runs with web_search="live" and the
# prompt carries no network note.
WORK_INTERNET_ENV = "MLSYSFORGE_WORK_INTERNET"


def _task_internet(requested: Any) -> bool:
    """Resolve the Codex/prompt internet flag from MLSYSFORGE_WORK_INTERNET."""
    value = os.environ.get(WORK_INTERNET_ENV, "").strip().lower()
    if value == "false":
        return False
    if value in {"", "true"}:
        return bool(requested)
    raise RuntimeError(f"{WORK_INTERNET_ENV} must be 'true' or 'false'")


_original_format_run_cmd = CodexAgent.format_run_cmd


def task_internet_format_run_cmd(
    self: CodexAgent, *args: Any, **kwargs: Any
) -> str:
    """Build the Codex command with the task JSON's internet setting."""
    kwargs["internet"] = _task_internet(kwargs.get("internet", True))
    return _original_format_run_cmd(self, *args, **kwargs)


CodexAgent.format_run_cmd = task_internet_format_run_cmd

_original_generate_prompt = run_agent.generate_evolve_prompt


def task_internet_generate_prompt(*args: Any, **kwargs: Any) -> str:
    """Build the evolve prompt with the task JSON's internet setting."""
    kwargs["internet"] = _task_internet(kwargs.get("internet", True))
    return _original_generate_prompt(*args, **kwargs)


run_agent.generate_evolve_prompt = task_internet_generate_prompt

from sforge.cli import main


if __name__ == "__main__":
    # The launcher picks the Agent (MLSYSFORGE_AGENT=codex|claude-code); Claude
    # Code bindings: sforge/harness/agent/claude_local.py.  Codex is untouched.
    import sys

    from sforge.harness.agent.claude_local import install_from_environment

    install_from_environment(_task_internet, sys.argv)
    main()
