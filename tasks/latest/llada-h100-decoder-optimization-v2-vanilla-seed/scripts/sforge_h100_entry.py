#!/usr/bin/env python3
"""Role-aware SForge Docker adapter for the LLaDA task.

Work and Judge are separate single-GPU containers.  The launcher resolves the
role devices to physical UUIDs and exports the expected UUID for each role
(two distinct cards on the legacy path, the same scheduler-allocated whole GPU
under the single-allocation handoff).  This adapter validates that role/UUID
tuple again at the final Docker boundary and creates one explicit NVIDIA
``DeviceRequest``.  Under the handoff the host coordinator additionally passes
the attested ``accelerator_allocation`` and the trusted binding labels, and
the standalone final Judge is mounted on its own root-owned checkpoint
directory scoped to the final run identity.
"""

from __future__ import annotations

import os
import re
import stat
import math
from pathlib import Path
from typing import Any, Literal

# SForge reads this setting while its modules are imported.
os.environ.setdefault("SFORGE_CODEX_CLI_VERSION", "0.148.0")

from docker.types import DeviceRequest
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
from sforge.harness.resource_handoff import JUDGE_ATTEMPT_ANNOTATION


Role = Literal["work", "judge"]

TASK_ID = "llada_h100_decoder_optimization_v2"
NETWORK_ENV = "SFORGE_ISOLATED_DOCKER_NETWORK"
MODEL_REPO_ENV = "LLADA_MODEL_REPO"
WORK_UUID_ENV = "LLADA_WORK_GPU_UUID"
JUDGE_UUID_ENV = "LLADA_JUDGE_GPU_UUID"
ROLE_ENV = "LLADA_GPU_ROLE"
# Execution-profile ids from the task JSON `portable` block.  Each names a
# capability envelope (1 CUDA device per role, sm_80+, >= 34 GiB), never a
# GPU model.  The legacy dual-card path keeps its historical id; the
# single-allocation handoff selects the count-1 profile through
# LLADA_EXECUTION_PROFILE_ID, exported by the launcher for the whole run.
EXECUTION_PROFILE_ID = "dual-cuda-sm80-34g"
SINGLE_EXECUTION_PROFILE_ID = "single-cuda-sm80-34g"
EXECUTION_PROFILE_IDS = (SINGLE_EXECUTION_PROFILE_ID, EXECUTION_PROFILE_ID)
EXECUTION_PROFILE_ENV = "LLADA_EXECUTION_PROFILE_ID"
RUN_SCOPE_ENV = "LLADA_RUNTIME_RUN_ID"
# The standalone final Judge of the single-allocation handoff is created by
# the long-lived coordinator, whose environment names the interactive run.
# These two variables carry the final run identity and its root-owned
# checkpoint directory; the adapter binds them only to the coordinator's
# final Judge container (name suffix and attempt annotation both "final-").
FINAL_RUN_SCOPE_ENV = "LLADA_FINAL_RUNTIME_RUN_ID"
FINAL_JUDGE_CHECKPOINT_DIR_ENV = "LLADA_FINAL_JUDGE_CHECKPOINT_DIR"
FINAL_HANDOFF_SUBMISSION_RE = re.compile(r"^final-[0-9a-f]{12}$")
FINAL_HANDOFF_ATTEMPT_RE = re.compile(r"^final-[0-9a-f]{24}$")
WORK_IMAGE_ID_ENV = "SFORGE_EXPECTED_WORK_IMAGE_ID"
JUDGE_IMAGE_ID_ENV = "SFORGE_EXPECTED_JUDGE_IMAGE_ID"
JUDGE_CHECKPOINT_DIR_ENV = "LLADA_JUDGE_CHECKPOINT_DIR"
MODEL_MOUNT = "/opt/models/llada-hf"
JUDGE_CHECKPOINT_MOUNT = "/logs/verifier/checkpoints"
MODEL_REVISION = "6059b30c9531767ce47e0ca6a8161a6bb0c67693"
GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
RUN_SCOPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _image_role(image: str) -> Role | None:
    work_marker = f".work.{TASK_ID}"
    judge_marker = f".judge.{TASK_ID}"
    work = work_marker in image
    judge = judge_marker in image
    if work == judge:
        return None
    return "work" if work else "judge"


def _name_role(name: str) -> Role | None:
    work = name.startswith(f"sforge.run.{TASK_ID}.")
    judge = (
        f".judge.{TASK_ID}." in name
        or name.startswith(f"sforge.game.{TASK_ID}.")
    )
    if work == judge:
        return None
    return "work" if work else "judge"


def _classify_role(image: str, name: str) -> Role:
    """Require the image and container identities to agree on one role."""

    image_role = _image_role(image)
    name_role = _name_role(name)
    if image_role is None or name_role is None:
        raise RuntimeError(
            "cannot unambiguously classify Work/Judge container from both "
            f"image={image!r} and name={name!r}"
        )
    if image_role != name_role:
        raise RuntimeError(
            "container role mismatch between image and name: "
            f"image={image_role}, name={name_role}"
        )
    return image_role


def _single_gpu_uuid(raw: str, *, field: str) -> str:
    value = raw.strip()
    if not value:
        raise RuntimeError(f"{field} is required")
    if "," in value or not GPU_UUID_RE.fullmatch(value):
        raise RuntimeError(
            f"{field} must be exactly one explicit physical GPU UUID"
        )
    return value


def _expected_role_uuid(role: Role) -> str:
    field = WORK_UUID_ENV if role == "work" else JUDGE_UUID_ENV
    return _single_gpu_uuid(os.environ.get(field, ""), field=field)


def _run_scope(field: str = RUN_SCOPE_ENV) -> str:
    value = os.environ.get(field, "").strip()
    if RUN_SCOPE_RE.fullmatch(value) is None:
        raise RuntimeError(f"{field} must be a safe nonempty run ID")
    return value


def _execution_profile_id() -> str:
    """The execution profile this run was launched under (default: legacy)."""
    value = os.environ.get(EXECUTION_PROFILE_ENV, EXECUTION_PROFILE_ID).strip()
    if value not in EXECUTION_PROFILE_IDS:
        raise RuntimeError(
            f"{EXECUTION_PROFILE_ENV} must be one of {', '.join(EXECUTION_PROFILE_IDS)}"
        )
    return value


def _binding_labels(annotations: dict[str, str] | None) -> dict[str, str]:
    """Validate SForge annotations exactly like the stock Docker backend does.

    Handoff binding annotations (run, lease, allocation, protocol, contract)
    must arrive complete so the coordinator's exact container lookup works.
    """
    if not annotations:
        return {}
    labels = _validated_docker_labels(annotations)
    if HANDOFF_BINDING_ANNOTATIONS.intersection(labels):
        labels = _validated_handoff_binding_labels(labels)
    return labels


def _is_final_handoff_judge(role: Role, name: str, labels: dict[str, str]) -> bool:
    """Whether this Judge is the coordinator's standalone final Judge.

    The coordinator names that container ``<benchmark>.judge.<task>.final-<12
    hex>`` and annotates it with attempt ``final-<24 hex>`` derived from the
    same request digest; interactive Judges carry a random 12-hex submission
    id and the same id as attempt.  The two signals must agree, otherwise the
    container is refused: a final Judge must never be mounted on the
    interactive checkpoint directory, nor an interactive Judge on the final one.
    """
    if role != "judge":
        return False
    suffix = name.rsplit(".", 1)[-1]
    name_final = FINAL_HANDOFF_SUBMISSION_RE.fullmatch(suffix) is not None
    attempt = labels.get(JUDGE_ATTEMPT_ANNOTATION)
    attempt_final = (
        isinstance(attempt, str)
        and FINAL_HANDOFF_ATTEMPT_RE.fullmatch(attempt) is not None
    )
    if name_final != attempt_final or (
        name_final and not attempt.startswith(suffix)
    ):
        raise RuntimeError(
            "final handoff Judge identity mismatch between container name "
            f"{name!r} and attempt annotation {attempt!r}"
        )
    return name_final


def _immutable_image(backend: DockerBackend, role: Role, image: str) -> str:
    field = WORK_IMAGE_ID_ENV if role == "work" else JUDGE_IMAGE_ID_ENV
    expected = os.environ.get(field, "").strip()
    if IMAGE_ID_RE.fullmatch(expected) is None:
        raise RuntimeError(f"{field} must be an immutable Docker image ID")
    try:
        tagged = backend.client.images.get(image)
        immutable = backend.client.images.get(expected)
    except Exception as exc:
        raise RuntimeError(f"cannot resolve the attested {role} image") from exc
    if tagged.id != expected or immutable.id != expected:
        raise RuntimeError(f"{role} image tag no longer matches {field}")
    return expected


def _validated_container_environment(
    role: Role,
    environment: dict[str, str] | None,
) -> tuple[dict[str, str], str]:
    container_environment = dict(environment or {})
    requested_uuid = _single_gpu_uuid(
        container_environment.get("NVIDIA_VISIBLE_DEVICES", ""),
        field="NVIDIA_VISIBLE_DEVICES",
    )
    expected_uuid = _expected_role_uuid(role)
    if requested_uuid != expected_uuid:
        raise RuntimeError(
            f"{role} requested GPU {requested_uuid}, expected {expected_uuid}"
        )

    requested_role = container_environment.get(ROLE_ENV, "").strip().lower()
    if requested_role != role:
        raise RuntimeError(
            f"{ROLE_ENV} must identify this container as {role}, "
            f"got {requested_role or 'empty'}"
        )
    requested_cuda = container_environment.get("CUDA_VISIBLE_DEVICES", "0").strip()
    if requested_cuda != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the logical ordinal 0")
    # LLADA_TARGET names the execution profile, not a GPU model: hardware
    # admission is the capability envelope enforced by the launcher and Judge.
    execution_profile_id = _execution_profile_id()
    requested_target = container_environment.get(
        "LLADA_TARGET", execution_profile_id
    ).strip()
    if requested_target != execution_profile_id:
        raise RuntimeError(f"LLADA_TARGET must be {execution_profile_id}")
    if role == "judge":
        for field in (
            "LLADA_INTERACTIVE_JUDGE_TIMEOUT",
            "LLADA_FINAL_JUDGE_TIMEOUT",
        ):
            raw = container_environment.get(field, "").strip()
            try:
                timeout = float(raw)
            except ValueError as exc:
                raise RuntimeError(f"{field} must be explicitly positive") from exc
            if not math.isfinite(timeout) or timeout <= 0:
                raise RuntimeError(f"{field} must be explicitly positive")

    if role == "judge":
        # Complete interactive and terminal workloads share one deadline.
        container_environment["LLADA_INTERACTIVE_JUDGE_TIMEOUT"] = container_environment["LLADA_FINAL_JUDGE_TIMEOUT"]

    # These values are authoritative even if a caller omitted the optional
    # duplicates. NVIDIA_VISIBLE_DEVICES remains the physical UUID while CUDA
    # code sees the selected device as logical cuda:0.
    container_environment["NVIDIA_VISIBLE_DEVICES"] = requested_uuid
    container_environment["CUDA_VISIBLE_DEVICES"] = "0"
    container_environment["LLADA_TARGET"] = execution_profile_id
    container_environment[ROLE_ENV] = role
    container_environment["LLADA_MODEL_PATH"] = (
        f"{MODEL_MOUNT}/snapshots/{MODEL_REVISION}"
    )
    container_environment["HF_HUB_OFFLINE"] = "1"
    container_environment["TRANSFORMERS_OFFLINE"] = "1"
    container_environment["HF_DATASETS_OFFLINE"] = "1"
    container_environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return container_environment, requested_uuid


def _model_volume() -> str:
    raw = os.environ.get(MODEL_REPO_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{MODEL_REPO_ENV} is required")
    repo = Path(raw).resolve(strict=True)
    snapshot = repo / "snapshots" / MODEL_REVISION
    if not repo.is_dir() or repo.is_symlink():
        raise RuntimeError(f"model repository must be a real directory: {repo}")
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise RuntimeError(f"pinned LLaDA snapshot is unavailable below {repo}")
    return f"{repo}:{MODEL_MOUNT}:ro"


def _judge_checkpoint_volume() -> str:
    return _scoped_judge_checkpoint_volume(
        JUDGE_CHECKPOINT_DIR_ENV, _run_scope()
    )


def _final_judge_checkpoint_volume() -> str:
    """The standalone final Judge's own root-owned checkpoint directory."""
    return _scoped_judge_checkpoint_volume(
        FINAL_JUDGE_CHECKPOINT_DIR_ENV, _run_scope(FINAL_RUN_SCOPE_ENV)
    )


def _scoped_judge_checkpoint_volume(field: str, run_scope: str) -> str:
    raw = os.environ.get(field, "").strip()
    if not raw:
        raise RuntimeError(f"{field} is required for Judge")
    candidate = Path(raw)
    if not candidate.is_absolute() or candidate.is_symlink():
        raise RuntimeError("Judge checkpoint directory must be absolute and non-symlink")
    try:
        checkpoint = candidate.resolve(strict=True)
        logs_root = Path(os.environ["SFORGE_LOG_DIR"]).resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise RuntimeError("cannot resolve the Judge checkpoint/log root") from exc
    if (
        checkpoint != candidate
        or not checkpoint.is_dir()
        or logs_root not in checkpoint.parents
        or checkpoint.name != "judge-checkpoints"
        or checkpoint.parent.name != run_scope
        or checkpoint.parent.parent.name != "runs"
    ):
        raise RuntimeError("Judge checkpoint directory is outside its exact run scope")
    metadata = checkpoint.stat()
    if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise RuntimeError("Judge checkpoint directory must be root-owned mode 700")
    return f"{checkpoint}:{JUDGE_CHECKPOINT_MOUNT}:rw"


def _validated_work_network(backend: DockerBackend) -> str:
    network_name = os.environ.get(NETWORK_ENV, "").strip()
    if not network_name:
        raise RuntimeError(f"{NETWORK_ENV} is required for Work")
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


def role_bound_create_container(
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
    """Create one role-bound single-GPU container with immutable inputs.

    Under the single-allocation handoff the host coordinator passes the exact
    ``accelerator_allocation`` it attested and the trusted binding labels; the
    allocation must name the same single device the role environment pins.
    """
    role = _classify_role(image, name)
    container_environment, requested_uuid = _validated_container_environment(
        role, environment
    )
    if accelerator_allocation is not None:
        if accelerator_allocation.vendor != "nvidia":
            raise RuntimeError(
                "task adapter supports only NVIDIA accelerator allocations, "
                f"not {accelerator_allocation.vendor!r}"
            )
        if list(accelerator_allocation.device_ids) != [requested_uuid]:
            raise RuntimeError(
                f"{role} accelerator allocation does not name the single "
                f"pinned device {requested_uuid}"
            )
    binding_labels = _binding_labels(annotations)
    final_handoff_judge = _is_final_handoff_judge(role, name, binding_labels)
    run_scope = _run_scope(FINAL_RUN_SCOPE_ENV) if final_handoff_judge else _run_scope()
    immutable_image = _immutable_image(self, role, image)
    volumes = [_model_volume()]
    if final_handoff_judge:
        volumes.append(_final_judge_checkpoint_volume())
    elif role == "judge":
        volumes.append(_judge_checkpoint_volume())
    if mounts:
        # Only SForge's host-provided agent runtime uses this path: a
        # read-only bind mount into the untrusted Work container.  The
        # Judge mount plan stays exclusively task-owned, and nothing may
        # ever be writable from Work.
        if role == "judge":
            raise RuntimeError("Judge containers accept no external mounts")
        for host_path, container_path, mode in mounts:
            if mode != "ro":
                raise RuntimeError(
                    "Work containers accept only read-only bind mounts"
                )
            volumes.append(f"{host_path}:{container_path}:{mode}")
    kwargs: dict[str, Any] = {
        "volumes": volumes,
        "device_requests": [
            DeviceRequest(
                driver="nvidia",
                device_ids=[requested_uuid],
                capabilities=[["gpu"]],
            )
        ],
        "security_opt": ["no-new-privileges:true"],
        "labels": {
            **binding_labels,
            "mlsysforge.task_id": TASK_ID,
            "mlsysforge.run_id": run_scope,
            "mlsysforge.gpu_role": role,
            "mlsysforge.gpu_uuid": requested_uuid,
        },
    }
    if role == "judge":
        kwargs["network_mode"] = "none"
    else:
        kwargs["network"] = _validated_work_network(self)
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

    container = self.client.containers.create(
        immutable_image,
        name=name,
        detach=True,
        command=command,
        environment=container_environment,
        **kwargs,
    )
    return DockerContainerHandle(container)


DockerBackend.create_container = role_bound_create_container

# Node and the pinned Codex CLI are baked into Work. The task-owned sidecar
# (scripts/restricted_connect_proxy.py) is Work's only route off the internal
# network: the allowlisted ChatGPT endpoints, plus public internet hosts when
# the task JSON says ``internet: true`` (sidecar ``--allow-public-egress``).
CodexAgent.install_cmds = [
    'test "$(node --version)" = v20.18.0 && '
    'codex --version | grep -F -- "${SFORGE_CODEX_CLI_VERSION:-0.148.0}"'
]

# The launcher always passes --enable-internet because SForge's own
# internet=false isolation needs sudo iptables, which this host does not
# grant; the internal network plus the sidecar above is the enforcement
# boundary.  The task JSON's ``internet`` value reaches this entry through
# MLSYSFORGE_WORK_INTERNET ("true"/"false", exported by the launcher).
# "false" keeps the offline Codex flags and prompt; "true" (or unset) passes
# the caller's value through, so Codex runs with web_search="live" and the
# prompt carries no network note.
WORK_INTERNET_ENV = "MLSYSFORGE_WORK_INTERNET"


def _task_internet(requested: Any) -> bool:
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
    kwargs["internet"] = _task_internet(kwargs.get("internet", True))
    return _original_format_run_cmd(self, *args, **kwargs)


CodexAgent.format_run_cmd = task_internet_format_run_cmd

_original_generate_prompt = run_agent.generate_evolve_prompt


def task_internet_generate_prompt(*args: Any, **kwargs: Any) -> str:
    kwargs["internet"] = _task_internet(kwargs.get("internet", True))
    return _original_generate_prompt(*args, **kwargs)


run_agent.generate_evolve_prompt = task_internet_generate_prompt

from sforge.cli import main


if __name__ == "__main__":
    main()
