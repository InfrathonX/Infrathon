#!/usr/bin/env python3
"""SForge Docker adapter for GPU pinning and read-only external model mounts.

Work gets exactly one device, pinned by ``NVIDIA_VISIBLE_DEVICES``.  The Judge
gets every whole-GPU UUID listed in ``QVG_JUDGE_GPU_UUIDS`` (spaces or
semicolons; SForge splits its Judge extra-env on commas, so the list travels
in this host variable) as one DeviceRequest, and the controller spreads the
hidden cases across them.  Every Judge GPU stays visible inside the container
(no ``CUDA_VISIBLE_DEVICES``): the controller binds each case to its device by
index in PCI bus order.  Without the variable the Judge falls back to the one
device ``NVIDIA_VISIBLE_DEVICES`` pins.  A coordinator allocation, when
present, must name exactly the same devices; there is no fallback to an index.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any

# This task owns a newer CLI pin than SForge's repository-wide compatibility
# default. Set it before importing SForge config so even direct task-local
# runtime entry calls resolve the supported version.
os.environ.setdefault("SFORGE_CODEX_CLI_VERSION", "0.148.0")

from docker.types import DeviceRequest
from sforge.harness.agent.codex import CodexAgent
from sforge.harness.backend import AcceleratorAllocation, HANDOFF_BINDING_ANNOTATIONS
from sforge.harness.backend.docker_backend import (
    DockerBackend,
    DockerContainerHandle,
    _validated_docker_labels,
    _validated_handoff_binding_labels,
)
from sforge.harness import run_agent
from codex_search import enable_native_web_search


artifact_root_value = os.environ.get("QVG_ARTIFACT_ROOT", "").strip()
if not artifact_root_value:
    raise RuntimeError("QVG_ARTIFACT_ROOT must name the external Quant-VideoGen assets")
ARTIFACT_ROOT = Path(artifact_root_value)
REPLAY_MODEL_LOCK_ENV = "QVG_REPLAY_MODEL_LOCK_PATH"
REPLAY_MODEL_LOCK_SHA_ENV = "QVG_REPLAY_MODEL_LOCK_SHA256"
REPLAY_VERIFY_MODELS_SHA_ENV = "QVG_REPLAY_VERIFY_MODELS_SHA256"
REPLAY_RUNTIME_IDENTITY_PATH_ENV = "QVG_REPLAY_RUNTIME_IDENTITY_PATH"
REPLAY_RUNTIME_IDENTITY_SHA_ENV = "QVG_REPLAY_RUNTIME_IDENTITY_SHA256"
REPLAY_RUNTIME_IDENTITY_HELPER_SHA_ENV = (
    "QVG_REPLAY_RUNTIME_IDENTITY_HELPER_SHA256"
)
# Space-separated, index i pairs with UUID i; replay_candidates.sh exports the
# Judge GPUs it attested, which must be exactly the Judge container's devices.
REPLAY_GPU_INDICES_ENV = "QVG_REPLAY_GPU_INDICES"
REPLAY_GPU_UUIDS_ENV = "QVG_REPLAY_GPU_UUIDS"
JUDGE_GPUS_ENV = "QVG_JUDGE_GPU_UUIDS"
MAX_JUDGE_GPUS = 8
VERIFY_MODELS = Path(__file__).with_name("verify_models.py")
RUNTIME_IDENTITY = Path(__file__).with_name("runtime_identity.py")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GPU_UUID_RE = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_replay_guarded_container_ids: set[str] = set()


def _uuid_list(raw: str) -> list[str]:
    return raw.replace(";", " ").split()


def _judge_gpu_uuids() -> list[str]:
    """The Judge's whole-GPU UUIDs from the host variable; empty when unset."""

    uuids = _uuid_list(os.environ.get(JUDGE_GPUS_ENV, ""))
    if any(_GPU_UUID_RE.fullmatch(value) is None for value in uuids):
        raise RuntimeError(f"{JUDGE_GPUS_ENV} must list whole-GPU UUIDs")
    if len(set(uuids)) != len(uuids) or len(uuids) > MAX_JUDGE_GPUS:
        raise RuntimeError(
            f"{JUDGE_GPUS_ENV} must name 1-{MAX_JUDGE_GPUS} distinct GPUs"
        )
    return uuids


def _replay_gpu_selection() -> tuple[list[str], list[str]]:
    """Attested Judge GPU indices and UUIDs, paired by position."""

    indices = os.environ.get(REPLAY_GPU_INDICES_ENV, "").split()
    uuids = _uuid_list(os.environ.get(REPLAY_GPU_UUIDS_ENV, ""))
    if (
        not 1 <= len(indices) <= MAX_JUDGE_GPUS
        or len(indices) != len(uuids)
        or len(set(indices)) != len(indices)
        or len(set(uuids)) != len(uuids)
        or any(re.fullmatch(r"0|[1-9][0-9]*", value) is None for value in indices)
        or any(_GPU_UUID_RE.fullmatch(value) is None for value in uuids)
    ):
        raise RuntimeError("incomplete QVG replay attestation contract")
    return indices, uuids


def _stable_stat_tuple(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_regular(path: Path, *, private: bool = False) -> bytes:
    try:
        linked_before = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError("QVG replay attestation input is unavailable") from exc
    if not stat.S_ISREG(linked_before.st_mode) or linked_before.st_uid != os.getuid():
        raise RuntimeError("QVG replay attestation input is unsafe")
    if private and stat.S_IMODE(linked_before.st_mode) & 0o077:
        raise RuntimeError("QVG replay attestation evidence is not private")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError("QVG replay attestation input is unavailable") from exc
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino)
            != (linked_before.st_dev, linked_before.st_ino)
        ):
            raise RuntimeError("QVG replay attestation input changed")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        linked_after = os.stat(path, follow_symlinks=False)
        if (
            _stable_stat_tuple(before) != _stable_stat_tuple(after)
            or (after.st_dev, after.st_ino)
            != (linked_after.st_dev, linked_after.st_ino)
            or not stat.S_ISREG(linked_after.st_mode)
        ):
            raise RuntimeError("QVG replay attestation input changed")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_stable_regular(path: Path, *, private: bool = False) -> str:
    return hashlib.sha256(_read_stable_regular(path, private=private)).hexdigest()


def _replay_attestation_enabled() -> bool:
    return any(
        os.environ.get(name, "").strip()
        for name in (
            REPLAY_MODEL_LOCK_ENV,
            REPLAY_MODEL_LOCK_SHA_ENV,
            REPLAY_VERIFY_MODELS_SHA_ENV,
            REPLAY_RUNTIME_IDENTITY_PATH_ENV,
            REPLAY_RUNTIME_IDENTITY_SHA_ENV,
            REPLAY_RUNTIME_IDENTITY_HELPER_SHA_ENV,
            REPLAY_GPU_INDICES_ENV,
            REPLAY_GPU_UUIDS_ENV,
        )
    )


def _verify_replay_live_artifacts() -> None:
    """Re-run model verification at every replay Judge boundary."""

    baseline_raw = os.environ.get(REPLAY_MODEL_LOCK_ENV, "").strip()
    expected_baseline_sha = os.environ.get(REPLAY_MODEL_LOCK_SHA_ENV, "").strip()
    expected_verifier_sha = os.environ.get(
        REPLAY_VERIFY_MODELS_SHA_ENV, ""
    ).strip()
    runtime_baseline_raw = os.environ.get(
        REPLAY_RUNTIME_IDENTITY_PATH_ENV, ""
    ).strip()
    expected_runtime_baseline_sha = os.environ.get(
        REPLAY_RUNTIME_IDENTITY_SHA_ENV, ""
    ).strip()
    expected_runtime_helper_sha = os.environ.get(
        REPLAY_RUNTIME_IDENTITY_HELPER_SHA_ENV, ""
    ).strip()
    replay_gpu_indices = os.environ.get(REPLAY_GPU_INDICES_ENV, "").strip()
    replay_gpu_uuids = os.environ.get(REPLAY_GPU_UUIDS_ENV, "").strip()
    if not any(
        (
            baseline_raw,
            expected_baseline_sha,
            expected_verifier_sha,
            runtime_baseline_raw,
            expected_runtime_baseline_sha,
            expected_runtime_helper_sha,
            replay_gpu_indices,
            replay_gpu_uuids,
        )
    ):
        return
    if (
        not baseline_raw
        or _SHA256_RE.fullmatch(expected_baseline_sha) is None
        or _SHA256_RE.fullmatch(expected_verifier_sha) is None
        or not runtime_baseline_raw
        or _SHA256_RE.fullmatch(expected_runtime_baseline_sha) is None
        or _SHA256_RE.fullmatch(expected_runtime_helper_sha) is None
    ):
        raise RuntimeError("incomplete QVG replay attestation contract")
    gpu_indices, gpu_uuids = _replay_gpu_selection()
    baseline_path = Path(baseline_raw)
    if not baseline_path.is_absolute():
        raise RuntimeError("QVG replay model lock path must be absolute")
    baseline_bytes = _read_stable_regular(baseline_path, private=True)
    if hashlib.sha256(baseline_bytes).hexdigest() != expected_baseline_sha:
        raise RuntimeError("QVG replay model lock changed")
    if _sha256_stable_regular(VERIFY_MODELS) != expected_verifier_sha:
        raise RuntimeError("QVG replay model verifier changed")
    try:
        baseline = json.loads(baseline_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("QVG replay model lock is invalid") from exc
    if not isinstance(baseline, dict) or baseline.get("status") != "verified":
        raise RuntimeError("QVG replay model lock is not verified")

    with tempfile.TemporaryDirectory(prefix="sforge-qvg-replay-attempt-") as temporary:
        output = Path(temporary) / "model-lock.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(VERIFY_MODELS),
                "--artifact-root",
                str(ARTIFACT_ROOT),
                "--output",
                str(output),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3600,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError("QVG replay live model verification failed")
        fresh_bytes = _read_stable_regular(output, private=True)

    if _sha256_stable_regular(VERIFY_MODELS) != expected_verifier_sha:
        raise RuntimeError("QVG replay model verifier changed")
    try:
        fresh = json.loads(fresh_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("QVG replay live model evidence is invalid") from exc
    if fresh != baseline:
        raise RuntimeError("QVG replay live model artifacts changed during the batch")
    if _sha256_stable_regular(baseline_path, private=True) != expected_baseline_sha:
        raise RuntimeError("QVG replay model lock changed")

    runtime_baseline_path = Path(runtime_baseline_raw)
    if not runtime_baseline_path.is_absolute():
        raise RuntimeError("QVG replay runtime identity path must be absolute")
    runtime_baseline = _read_stable_regular(runtime_baseline_path, private=True)
    if hashlib.sha256(runtime_baseline).hexdigest() != expected_runtime_baseline_sha:
        raise RuntimeError("QVG replay runtime identity evidence changed")
    if (
        not RUNTIME_IDENTITY.is_file()
        or RUNTIME_IDENTITY.is_symlink()
        or _sha256_stable_regular(RUNTIME_IDENTITY)
        != expected_runtime_helper_sha
    ):
        raise RuntimeError("QVG replay runtime identity helper changed")
    gpu_arguments: list[str] = []
    for gpu_index, gpu_uuid in zip(gpu_indices, gpu_uuids, strict=True):
        gpu_arguments += ["--gpu-index", gpu_index, "--expected-gpu-uuid", gpu_uuid]
    runtime_completed = subprocess.run(
        [
            sys.executable,
            str(RUNTIME_IDENTITY),
            *gpu_arguments,
            "--verify-baseline",
            str(runtime_baseline_path),
            "--expected-baseline-sha256",
            expected_runtime_baseline_sha,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if (
        runtime_completed.returncode != 0
        or _sha256_stable_regular(RUNTIME_IDENTITY)
        != expected_runtime_helper_sha
        or _sha256_stable_regular(runtime_baseline_path, private=True)
        != expected_runtime_baseline_sha
    ):
        raise RuntimeError("QVG replay host runtime identity changed")


def work_volumes() -> dict[str, dict[str, str]]:
    return {
        str(ARTIFACT_ROOT / "staging/self-forcing"): {"bind": "/opt/qvg-src/ckpts/Self-Forcing", "mode": "ro"},
        str(ARTIFACT_ROOT / "staging/vbench"): {"bind": "/opt/vbench", "mode": "ro"},
        str(ARTIFACT_ROOT / "staging/vbench-cache"): {"bind": "/opt/vbench-cache", "mode": "ro"},
        str(ARTIFACT_ROOT / "public-eval"): {"bind": "/eval/public", "mode": "ro"},
    }


def judge_volumes() -> dict[str, dict[str, str]]:
    hy = ARTIFACT_ROOT / "staging/hy-worldplay-8b"
    return {
        str(ARTIFACT_ROOT / "staging/longcat/LongCat-Video"): {"bind": "/models/hidden/longcat", "mode": "ro"},
        str(hy / "HunyuanVideo-1.5/transformer"): {"bind": "/models/hidden/hy-worldplay/base/transformer", "mode": "ro"},
        str(hy / "HunyuanVideo-1.5/vae"): {"bind": "/models/hidden/hy-worldplay/base/vae", "mode": "ro"},
        str(hy / "HunyuanVideo-1.5/scheduler"): {"bind": "/models/hidden/hy-worldplay/base/scheduler", "mode": "ro"},
        str(hy / "HY-WorldPlay/ar_distilled_action_model"): {"bind": "/models/hidden/hy-worldplay/action", "mode": "ro"},
        str(hy / "Qwen2.5-VL-7B-Instruct"): {"bind": "/models/hidden/hy-worldplay/base/text_encoder/llm", "mode": "ro"},
        str(hy / "byt5-small"): {"bind": "/models/hidden/hy-worldplay/base/text_encoder/byt5-small", "mode": "ro"},
        str(hy / "Glyph-SDXL-v2"): {"bind": "/models/hidden/hy-worldplay/base/text_encoder/Glyph-SDXL-v2", "mode": "ro"},
        # The official HY downloader copies the complete FLUX Redux snapshot
        # into this directory; SiglipVisionModel then selects image_encoder/.
        str(hy / "FLUX.1-Redux-dev"): {"bind": "/models/hidden/hy-worldplay/base/vision_encoder/siglip", "mode": "ro"},
        str(ARTIFACT_ROOT / "staging/vbench"): {"bind": "/opt/vbench", "mode": "ro"},
        str(ARTIFACT_ROOT / "staging/vbench-cache"): {"bind": "/opt/vbench-cache", "mode": "ro"},
        str(ARTIFACT_ROOT / "hidden"): {"bind": "/eval/hidden", "mode": "ro"},
    }


def mounted_create_container(
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
    # Under a coordinator handoff the host passes the exact allocation it
    # attested plus the trusted binding labels; the allocation must name the
    # same devices the role pins (one for Work, the listed set for the Judge).
    labels: dict[str, str] = {}
    if annotations:
        labels = _validated_docker_labels(annotations)
        if HANDOFF_BINDING_ANNOTATIONS.intersection(labels):
            labels = _validated_handoff_binding_labels(labels)
    is_judge = ".judge." in image or ".judge." in name
    env = dict(environment or {})
    judge_uuids = _judge_gpu_uuids()
    if is_judge and judge_uuids:
        pinned = env.get("NVIDIA_VISIBLE_DEVICES", "").strip()
        if pinned and pinned != ",".join(judge_uuids):
            raise RuntimeError(
                f"NVIDIA_VISIBLE_DEVICES disagrees with {JUDGE_GPUS_ENV}"
            )
        env["NVIDIA_VISIBLE_DEVICES"] = ",".join(judge_uuids)
        requested_devices = list(judge_uuids)
    else:
        visible_devices = env.get("NVIDIA_VISIBLE_DEVICES", "").strip()
        if not visible_devices or visible_devices.lower() in {"all", "void", "none"}:
            raise RuntimeError("Work/Judge containers require an explicit GPU UUID device list")
        requested_devices = [value.strip() for value in visible_devices.split(",") if value.strip()]
        if len(requested_devices) != 1 or not requested_devices[0].startswith("GPU-"):
            raise RuntimeError(
                "Work/Judge containers require exactly one explicit physical GPU UUID"
            )
        if not is_judge and requested_devices[0] in judge_uuids:
            raise RuntimeError("the Work GPU is one of the Judge GPUs")
    if is_judge:
        # Every Judge GPU stays visible; the controller binds each case to its
        # device by index (PCI bus order), so a mask would hide the others.
        env.pop("CUDA_VISIBLE_DEVICES", None)
    if accelerator_allocation is not None:
        if accelerator_allocation.vendor != "nvidia":
            raise RuntimeError(
                "task adapter supports only NVIDIA accelerator allocations, "
                f"not {accelerator_allocation.vendor!r}"
            )
        if sorted(accelerator_allocation.device_ids) != sorted(requested_devices):
            raise RuntimeError(
                "accelerator allocation does not name the pinned "
                f"devices {','.join(requested_devices)}"
            )
    replay_guard = is_judge and _replay_attestation_enabled()
    if replay_guard:
        if sorted(_replay_gpu_selection()[1]) != sorted(requested_devices):
            raise RuntimeError("QVG replay attestation does not cover the Judge GPUs")
        _verify_replay_live_artifacts()
    volumes = judge_volumes() if is_judge else work_volumes()
    for source in volumes:
        if not Path(source).exists():
            raise RuntimeError(f"required runtime mount is missing: {source}")
    kwargs: dict[str, Any] = {
        "volumes": volumes,
        "device_requests": [DeviceRequest(device_ids=requested_devices, capabilities=[["gpu"]])],
        "security_opt": ["no-new-privileges:true"],
    }
    if labels:
        kwargs["labels"] = labels
    if is_judge:
        kwargs["network_mode"] = "none"
    else:
        network_name = os.environ.get("SFORGE_ISOLATED_DOCKER_NETWORK", "").strip()
        if not network_name:
            raise RuntimeError("SFORGE_ISOLATED_DOCKER_NETWORK is required for Work")
        network = self.client.networks.get(network_name)
        if not bool(network.attrs.get("Internal")):
            raise RuntimeError(f"refusing non-internal Work network: {network_name}")
        kwargs["network"] = network_name
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
    if mounts:
        # Only SForge's host-provided agent runtime uses this path: a
        # read-only bind mount into the Work container. The Judge mount plan
        # stays exclusively task-owned, nothing is ever writable from Work,
        # and a runtime target may not shadow a task-owned mount.
        if is_judge:
            raise RuntimeError(
                "Judge containers accept only the trusted task mount plan"
            )
        for host_path, container_path, mode in mounts:
            if mode != "ro":
                raise RuntimeError(
                    "Work containers accept only read-only bind mounts"
                )
            if host_path in volumes:
                raise RuntimeError(
                    f"runtime mount collides with a task mount: {host_path}"
                )
            volumes[host_path] = {"bind": container_path, "mode": mode}
    container = self.client.containers.create(
        image,
        name=name,
        detach=True,
        command=command,
        environment=env,
        **kwargs,
    )
    handle = DockerContainerHandle(container)
    if replay_guard:
        _replay_guarded_container_ids.add(handle.id)
    return handle


_original_cleanup_container = DockerBackend.cleanup_container


def mounted_cleanup_container(
    self: DockerBackend,
    handle: DockerContainerHandle | None,
    logger: Any | None = None,
) -> None:
    container_id = handle.id if handle is not None else ""
    guarded = container_id in _replay_guarded_container_ids
    try:
        _original_cleanup_container(self, handle, logger)
    finally:
        if guarded:
            _replay_guarded_container_ids.discard(container_id)
            _verify_replay_live_artifacts()


DockerBackend.create_container = mounted_create_container
DockerBackend.cleanup_container = mounted_cleanup_container
CodexAgent.install_cmds = [
    'test "$(node --version)" = v20.18.0 && '
    'codex --version | grep -F "${SFORGE_CODEX_CLI_VERSION:-0.148.0}"'
]
_format_run_cmd = CodexAgent.format_run_cmd


def online_format_run_cmd(self: CodexAgent, *args: Any, **kwargs: Any) -> str:
    kwargs["internet"] = True
    return enable_native_web_search(_format_run_cmd(self, *args, **kwargs))


CodexAgent.format_run_cmd = online_format_run_cmd
_generate_prompt = run_agent.generate_evolve_prompt


def online_generate_prompt(*args: Any, **kwargs: Any) -> str:
    kwargs["internet"] = True
    return _generate_prompt(*args, **kwargs)


run_agent.generate_evolve_prompt = online_generate_prompt

from sforge.cli import main


if __name__ == "__main__":
    main()
