#!/usr/bin/env python3
"""SForge on local Docker for MInference v3 (scripts/run_local_v3.sh drives ``serve``, ``run``, ``eval``).

On PyroMind the controller mounts the three run-time trees and runs the task's
bootstrap (scripts/bootstrap/bootstrap.py; deploy/pyromind/minference-v3/ in the development tree).  This entry does the
same on host Docker by wrapping ``DockerBackend``:

* Work (``sforge.run.<task_id>.<run_id>``): the launcher's ``--internal``
  network (``SFORGE_ISOLATED_DOCKER_NETWORK``; its only peer is the restricted
  CONNECT sidecar, scripts/restricted_proxy.py), one exact DeviceRequest for
  ``NVIDIA_VISIBLE_DEVICES``, and the public model and prompts read-only at
  their fixed paths;
* Judge (``<benchmark>.judge.<task_id>.<submission>``): exactly the two
  read-only trees the bootstrap admits (model, final prompts) and, once
  started, ``install_private_judge``, which binds them into the root-only
  context before SForge's own kernel bootstrap runs (judge.private_bootstrap);
* every other container (the integrity review) keeps the stock path.

The private bootstrap needs the exact Judge GPU, which SForge passes only in its
single-allocation handoff mode; ``judge_submission`` therefore gets the
allocation of ``MINFERENCE_JUDGE_GPU_UUID`` when it has none.  An ``eval`` run
with ``MINFERENCE_FINAL_EVAL=1`` is a final: phase ``final`` and round
``final-1`` (the Judge admits agent-/auto-/final-N rounds only, not manual-N).

Mount sources come from the launcher: ``MINFERENCE_MODEL_DIR`` (the Judge's
model), ``MINFERENCE_WORK_MODEL_DIR`` (the public model),
``MINFERENCE_PUBLIC_PROMPTS_DIR`` and
``MINFERENCE_PRIVATE_PROMPTS_DIR`` (run-scoped single-link copies, the final
ones root-only).  The Codex command and evolve prompt follow the task JSON's
internet flag via MLSYSFORGE_WORK_INTERNET, as in the kernel launchers.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any

import docker

from sforge import cli as sforge_cli
from sforge.harness import judge_server, run_agent, run_evaluation
from sforge.harness.agent.codex import CodexAgent
from sforge.harness.backend import AcceleratorAllocation, HANDOFF_BINDING_ANNOTATIONS
from sforge.harness.backend.docker_backend import (
    DockerBackend,
    DockerContainerHandle,
    _validated_docker_labels,
    _validated_handoff_binding_labels,
)

TASK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TASK_ROOT.parents[2]
TASK_ID = "minference_sparse_prefill_h100_seed_v3"
BENCHMARK_NAME = "minference_sparse_prefill"
WORK_NAME_PREFIX = f"sforge.run.{TASK_ID}."
JUDGE_NAME_PREFIX = f"{BENCHMARK_NAME}.judge.{TASK_ID}."
NETWORK_ENV = "SFORGE_ISOLATED_DOCKER_NETWORK"
JUDGE_GPU_ENV = "MINFERENCE_JUDGE_GPU_UUID"
FINAL_ENV = "MINFERENCE_FINAL_EVAL"
UUID = re.compile(r"GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}")
WORK_MODEL_MOUNT = "/workspace/benchmark/model"
WORK_PROMPTS_MOUNT = "/workspace/benchmark/public-prompts"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ASSETS = _module("minference_local_assets", TASK_ROOT / "scripts/external_assets_on_demand_v3.py")
# The controller bootstrap PyroMind runs (deploy/pyromind/minference-v3/); the public
# repository keeps it with the task, in scripts/bootstrap/.
_BOOTSTRAP_PATHS = (TASK_ROOT / "scripts/bootstrap/bootstrap.py",
                    REPOSITORY_ROOT / "deploy/pyromind/minference-v3/bootstrap.py")
BOOTSTRAP = _module("minference_local_bootstrap", next(p for p in _BOOTSTRAP_PATHS if p.is_file()))
_roles: dict[str, str] = {}


def _directory(variable: str) -> str:
    raw = os.environ.get(variable, "").strip()
    if not raw:
        raise RuntimeError(f"{variable} is required")
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{variable} must be an absolute, non-symlink directory: {raw}")
    return str(path)


def _gpu(raw: str, what: str) -> AcceleratorAllocation:
    raw = raw.strip()
    if UUID.fullmatch(raw) is None:
        raise RuntimeError(f"{what} must be one whole-GPU UUID, got {raw!r}")
    return AcceleratorAllocation("nvidia", (raw,))


def _validated_work_network(backend: DockerBackend) -> str:
    """Require the launcher-created internal network used only by Work."""
    network_name = os.environ.get(NETWORK_ENV, "").strip()
    if not network_name:
        raise RuntimeError(f"{NETWORK_ENV} is required for the Work container")
    try:
        network = backend.client.networks.get(network_name)
        network.reload()
    except Exception as exc:
        raise RuntimeError(f"isolated Docker network does not exist: {network_name}") from exc
    if network.attrs.get("Internal") is not True:
        raise RuntimeError(f"refusing non-internal Work network: {network_name}")
    return network_name


def work_mounts() -> list[tuple[str, str, str]]:
    return [(_directory("MINFERENCE_WORK_MODEL_DIR"), WORK_MODEL_MOUNT, "ro"),
            (_directory("MINFERENCE_PUBLIC_PROMPTS_DIR"), WORK_PROMPTS_MOUNT, "ro")]


def judge_mounts() -> list[tuple[str, str, str]]:
    return [(_directory("MINFERENCE_MODEL_DIR"), ASSETS.MOUNTS["model"], "ro"),
            (_directory("MINFERENCE_PRIVATE_PROMPTS_DIR"), ASSETS.MOUNTS["private-performance"], "ro")]


_stock_create_container = DockerBackend.create_container
_stock_start_container = DockerBackend.start_container


def local_create_container(
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
    stock = dict(command=command, environment=environment, extra_hosts=extra_hosts, cap_drop=cap_drop,
                 cpu_limit=cpu_limit, mem_limit=mem_limit, network_mode=network_mode, user=user,
                 annotations=annotations, mounts=mounts, accelerator_allocation=accelerator_allocation)
    if name.startswith(JUDGE_NAME_PREFIX):
        # The bootstrap admits exactly the two asset mounts on an offline,
        # root, one-GPU Judge; SForge sets network none and the allocation.
        if mounts:
            raise RuntimeError("the MInference Judge accepts no SForge mounts")
        if accelerator_allocation is None:
            raise RuntimeError("the MInference Judge needs its exact GPU allocation")
        stock["mounts"] = judge_mounts()
        handle = _stock_create_container(self, image, name, **stock)
        _roles[handle.id] = "judge"
        return handle
    if not name.startswith(WORK_NAME_PREFIX):
        return _stock_create_container(self, image, name, **stock)
    if network_mode is not None:
        raise RuntimeError("the Work network is fixed by the launcher")
    if accelerator_allocation is None:
        accelerator_allocation = _gpu((environment or {}).get("NVIDIA_VISIBLE_DEVICES", ""),
                                      "the Work NVIDIA_VISIBLE_DEVICES")
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
    # SForge's agent runtime plus the two public trees, all read-only.
    volumes: dict[str, dict[str, str]] = {}
    for host_path, container_path, mode in [*(mounts or []), *work_mounts()]:
        if mode != "ro":
            raise RuntimeError("Work containers accept only read-only bind mounts")
        volumes[host_path] = {"bind": container_path, "mode": mode}
    kwargs["volumes"] = volumes
    kwargs["device_requests"] = [docker.types.DeviceRequest(
        driver="nvidia", device_ids=list(accelerator_allocation.device_ids), capabilities=[["gpu"]])]
    container = self.client.containers.create(
        image, name=name, detach=True, command=command, environment=environment or {}, **kwargs)
    handle = DockerContainerHandle(container)
    _roles[handle.id] = "work"
    return handle


def local_start_container(self: DockerBackend, handle) -> None:
    _stock_start_container(self, handle)
    role = _roles.get(handle.id)
    if role == "judge":
        # Sets handle.minference_external_context for SForge's kernel bootstrap.
        BOOTSTRAP.install_private_judge(self, handle, TASK_ROOT)


DockerBackend.create_container = local_create_container
DockerBackend.start_container = local_start_container

_stock_judge_submission = run_evaluation.judge_submission


def local_judge_submission(*args: Any, **kwargs: Any):
    task_spec = kwargs.get("task_spec", args[0] if args else None)
    if getattr(task_spec, "task_id", None) == TASK_ID:
        if kwargs.get("accelerator_allocation") is None:
            kwargs["accelerator_allocation"] = _gpu(os.environ.get(JUDGE_GPU_ENV, ""), JUDGE_GPU_ENV)
        if os.environ.get(FINAL_ENV) == "1":
            kwargs["evaluation_phase"] = "final"
            kwargs["round_id"] = "final-1"
    return _stock_judge_submission(*args, **kwargs)


# The CLI and the Judge server imported the name directly.
run_evaluation.judge_submission = local_judge_submission
judge_server.judge_submission = local_judge_submission
sforge_cli.judge_submission = local_judge_submission

# The launcher always passes --enable-internet because SForge's own
# internet=false isolation needs host iptables; the internal network plus the
# CONNECT sidecar is the enforcement boundary.  The task JSON's ``internet``
# reaches this entry through MLSYSFORGE_WORK_INTERNET ("true"/"false").
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


def task_internet_format_run_cmd(self: CodexAgent, *args: Any, **kwargs: Any) -> str:
    kwargs["internet"] = _task_internet(kwargs.get("internet", True))
    return _original_format_run_cmd(self, *args, **kwargs)


CodexAgent.format_run_cmd = task_internet_format_run_cmd

_original_generate_prompt = run_agent.generate_evolve_prompt


def task_internet_generate_prompt(*args: Any, **kwargs: Any) -> str:
    kwargs["internet"] = _task_internet(kwargs.get("internet", True))
    return _original_generate_prompt(*args, **kwargs)


run_agent.generate_evolve_prompt = task_internet_generate_prompt


if __name__ == "__main__":
    # The launcher picks the Agent (MLSYSFORGE_AGENT=codex|claude-code); Claude
    # Code bindings: sforge/harness/agent/claude_local.py.  Codex is untouched.
    from sforge.harness.agent.claude_local import install_from_environment

    install_from_environment(_task_internet, sys.argv)
    sforge_cli.main()
