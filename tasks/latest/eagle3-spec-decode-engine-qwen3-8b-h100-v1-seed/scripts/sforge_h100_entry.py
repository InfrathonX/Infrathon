#!/usr/bin/env python3
"""Role-aware SForge Docker adapter for the two-GPU EAGLE-3 task.

Work and Judge are separate single-GPU containers.  The launcher resolves the
two numeric host ordinals to physical UUIDs and exports the expected UUID for
each role.  This adapter validates that role/UUID tuple again at the final
Docker boundary and creates one explicit NVIDIA ``DeviceRequest``.
"""

from __future__ import annotations

import math
import os
import re
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


Role = Literal["work", "judge"]

TASK_ID = "eagle3_spec_decode_engine_qwen3_8b_h100_v1"
NETWORK_ENV = "SFORGE_ISOLATED_DOCKER_NETWORK"
MODEL_REPO_ENV = "EAGLE3_TARGET_REPO"
HEAD_REPO_ENV = "EAGLE3_HEAD_DIR"
WORK_UUID_ENV = "EAGLE3_WORK_GPU_UUID"
JUDGE_UUID_ENV = "EAGLE3_JUDGE_GPU_UUID"
ROLE_ENV = "EAGLE3_GPU_ROLE"
RUN_SCOPE_ENV = "EAGLE3_RUNTIME_RUN_ID"
WORK_IMAGE_ID_ENV = "SFORGE_EXPECTED_WORK_IMAGE_ID"
JUDGE_IMAGE_ID_ENV = "SFORGE_EXPECTED_JUDGE_IMAGE_ID"
# Execution-profile id, not a GPU model: the admitted envelope is one CUDA
# device per role, compute capability >= 8.0 and >= 34 GiB of device memory.
EXECUTION_PROFILE = "single-cuda-sm80-34g"
MODEL_MOUNT = "/opt/models/qwen3-8b-hf"
HEAD_MOUNT = "/opt/models/eagle3-head"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
RUN_SCOPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _image_role(image: str) -> Role | None:
    """Classify a task image tag as Work or Judge."""
    # Replay resolves a validated task image tag to its immutable image ID
    # before creating the container.  Treat only the launcher-provided,
    # already-validated IDs as role identities; arbitrary sha256 references
    # must remain unclassifiable.
    if image == os.environ.get(WORK_IMAGE_ID_ENV):
        return "work"
    if image == os.environ.get(JUDGE_IMAGE_ID_ENV):
        return "judge"
    work_marker = f".work.{TASK_ID}"
    judge_marker = f".judge.{TASK_ID}"
    work = work_marker in image
    judge = judge_marker in image
    if work == judge:
        return None
    return "work" if work else "judge"


def _name_role(name: str) -> Role | None:
    """Classify an SForge container name as Work or Judge."""
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
    """Validate and return one explicit physical GPU UUID."""
    value = raw.strip()
    if not value:
        raise RuntimeError(f"{field} is required")
    if "," in value or not GPU_UUID_RE.fullmatch(value):
        raise RuntimeError(
            f"{field} must be exactly one explicit physical GPU UUID"
        )
    return value


def _expected_role_uuid(role: Role) -> str:
    """Read the launcher-attested UUID for one container role."""
    field = WORK_UUID_ENV if role == "work" else JUDGE_UUID_ENV
    return _single_gpu_uuid(os.environ.get(field, ""), field=field)


def _run_scope() -> str:
    """Read the safe run identifier attached to every container."""
    value = os.environ.get(RUN_SCOPE_ENV, "").strip()
    if RUN_SCOPE_RE.fullmatch(value) is None:
        raise RuntimeError(f"{RUN_SCOPE_ENV} must be a safe nonempty run ID")
    return value


def _immutable_image(backend: DockerBackend, role: Role, image: str) -> str:
    """Resolve a tag and require its ID to match the launcher attestation."""
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
    """Normalize offline container variables and bind the expected role GPU."""
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
    requested_target = container_environment.get(
        "EAGLE3_TARGET", EXECUTION_PROFILE
    ).strip()
    if requested_target != EXECUTION_PROFILE:
        raise RuntimeError(f"EAGLE3_TARGET must be {EXECUTION_PROFILE}")
    if role == "judge":
        for field in (
            "EAGLE3_INTERACTIVE_JUDGE_TIMEOUT",
            "EAGLE3_FINAL_JUDGE_TIMEOUT",
        ):
            raw = container_environment.get(field, "").strip()
            try:
                timeout = float(raw)
            except ValueError as exc:
                raise RuntimeError(f"{field} must be explicitly positive") from exc
            if not math.isfinite(timeout) or timeout <= 0:
                raise RuntimeError(f"{field} must be explicitly positive")

    # These values are authoritative even if a caller omitted the optional
    # duplicates. NVIDIA_VISIBLE_DEVICES remains the physical UUID while CUDA
    # code sees the selected device as logical cuda:0.
    container_environment["NVIDIA_VISIBLE_DEVICES"] = requested_uuid
    container_environment["CUDA_VISIBLE_DEVICES"] = "0"
    container_environment["EAGLE3_TARGET"] = EXECUTION_PROFILE
    container_environment[ROLE_ENV] = role
    container_environment["EAGLE3_TARGET_REPO"] = MODEL_MOUNT
    container_environment["EAGLE3_HEAD_DIR"] = HEAD_MOUNT
    container_environment["HF_HUB_OFFLINE"] = "1"
    container_environment["TRANSFORMERS_OFFLINE"] = "1"
    container_environment["HF_DATASETS_OFFLINE"] = "1"
    container_environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return container_environment, requested_uuid


def _model_volume() -> str:
    """Return the pinned, read-only target repository mount specification."""
    raw = os.environ.get(MODEL_REPO_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{MODEL_REPO_ENV} is required")
    repo = Path(raw).resolve(strict=True)
    snapshot = repo / "snapshots" / MODEL_REVISION
    if not repo.is_dir() or repo.is_symlink():
        raise RuntimeError(f"model repository must be a real directory: {repo}")
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise RuntimeError(f"pinned Qwen3 snapshot is unavailable below {repo}")
    return f"{repo}:{MODEL_MOUNT}:ro"


def _head_volume() -> str:
    """Return the complete, read-only EAGLE head mount specification."""
    raw = os.environ.get(HEAD_REPO_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{HEAD_REPO_ENV} is required")
    root = Path(raw).resolve(strict=True)
    if not root.is_dir() or root.is_symlink() or not (root / "model.safetensors").is_file():
        raise RuntimeError("head artifact must be a real complete directory")
    return f"{root}:{HEAD_MOUNT}:ro"


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


def _validated_work_network(backend: DockerBackend) -> str:
    """Require the launcher-created internal network used only by Work."""
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
    immutable_image = _immutable_image(self, role, image)
    volumes = [_model_volume(), _head_volume()]
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
            **_binding_labels(annotations),
            "mlsysforge.task_id": TASK_ID,
            "mlsysforge.run_id": _run_scope(),
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

# Node and the pinned Codex CLI are baked into Work. The task-owned internal
# proxy is the only permitted route to the allowlisted ChatGPT endpoints.
CodexAgent.install_cmds = [
    'test "$(node --version)" = v20.18.0 && '
    'codex --version | grep -F -- "${SFORGE_CODEX_CLI_VERSION:-0.148.0}"'
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
    main()
