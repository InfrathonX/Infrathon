#!/usr/bin/env python3
"""Run this task's Work container on a launcher-created internal Docker network.

SForge's stock offline mode needs passwordless host iptables, which formal
hosts do not grant, and its Docker backend cannot name a network for the Work
container at all.  This entry wraps ``DockerBackend.create_container``:

* the Work container of this task (``sforge.run.<task_id>.<run_id>``) is
  attached to the ``--internal`` network named by
  ``SFORGE_ISOLATED_DOCKER_NETWORK``; its only reachable peer there is the
  launcher's restricted CONNECT-proxy sidecar (``scripts/restricted_proxy.py``),
  which allowlists the model-auth hosts itself and relays the Judge port;
* every other container (the interactive and final Judge) goes through the
  untouched stock path, so ``SFORGE_JUDGE_NETWORK_MODE=none`` keeps its stock
  meaning and the Judge stays offline.

The launcher passes ``--enable-internet`` to skip SForge's iptables
implementation; the network above still has no direct public egress.  With the
task's ``internet: true`` the sidecar runs with ``--allow-public-egress``
(public hosts on ports 80/443; internal addresses refused), and the Codex
command line and evolve prompt follow the task JSON via
MLSYSFORGE_WORK_INTERNET (true: live web search and no network text; false:
the old offline flags and prompt).
"""

from __future__ import annotations

import os
from typing import Any

import docker

from sforge.harness import run_agent
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


TASK_ID = "tilelang_mamba2_ssd_forward_h100_seed_v2"
WORK_NAME_PREFIX = f"sforge.run.{TASK_ID}."
WORK_IMAGE_MARKER = f".work.{TASK_ID}"
NETWORK_ENV = "SFORGE_ISOLATED_DOCKER_NETWORK"

_stock_create_container = DockerBackend.create_container


def _validated_work_network(backend: DockerBackend) -> str:
    """Require the launcher-created internal network used only by Work."""
    network_name = os.environ.get(NETWORK_ENV, "").strip()
    if not network_name:
        raise RuntimeError(f"{NETWORK_ENV} is required for the Work container")
    try:
        network = backend.client.networks.get(network_name)
        network.reload()
    except Exception as exc:
        raise RuntimeError(
            f"isolated Docker network does not exist: {network_name}"
        ) from exc
    if network.attrs.get("Internal") is not True:
        raise RuntimeError(f"refusing non-internal Work network: {network_name}")
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
    network_mode: str | None = None,
    user: str | None = None,
    annotations: dict[str, str] | None = None,
    mounts: list[tuple[str, str, str]] | None = None,
    accelerator_allocation: AcceleratorAllocation | None = None,
) -> DockerContainerHandle:
    if not name.startswith(WORK_NAME_PREFIX):
        # Judge containers keep the stock semantics (SFORGE_JUDGE_NETWORK_MODE).
        return _stock_create_container(
            self,
            image,
            name,
            command=command,
            environment=environment,
            extra_hosts=extra_hosts,
            cap_drop=cap_drop,
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
            network_mode=network_mode,
            user=user,
            annotations=annotations,
            mounts=mounts,
            accelerator_allocation=accelerator_allocation,
        )
    if WORK_IMAGE_MARKER not in image:
        raise RuntimeError(
            f"Work container {name!r} must use this task's Work image, got {image!r}"
        )
    if network_mode is not None:
        raise RuntimeError("the Work network is fixed by the launcher")

    kwargs: dict[str, Any] = {"network": _validated_work_network(self)}
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
        # read-only bind mount into the untrusted Work container.  Nothing may
        # ever be writable from inside.
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

from sforge.cli import main  # noqa: E402


if __name__ == "__main__":
    # The launcher picks the Agent (MLSYSFORGE_AGENT=codex|claude-code); Claude
    # Code bindings: sforge/harness/agent/claude_local.py.  Codex is untouched.
    import sys

    from sforge.harness.agent.claude_local import install_from_environment

    install_from_environment(_task_internet, sys.argv)
    main()
