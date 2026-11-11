#!/usr/bin/env python3
"""SForge adapter for the fixed dual-H100 ORPO task.

This module applies the container resource, mount, isolation, and managed-run
lifecycle contract.  Work and Judge request the same two physical H100s, so a
managed Judge waits for idle devices, pauses the exact run's Work container,
runs evaluation, and resumes Work in ``finally``.  Standalone archive replay
still requires the caller to provide an already-idle GPU pair.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Any, Iterator

from docker.types import DeviceRequest
from sforge.harness import run_agent
from sforge.harness.agent.codex import CodexAgent
from sforge.harness.backend.docker_backend import DockerBackend, DockerContainerHandle


MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MODEL_CONFIG_SHA256 = "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
DATASET_REVISION = "7ae482b332979f8ce40d2a6b5f78f9eddf137efc"
PUBLIC_DATASET_ROOT_MANIFEST_SHA256 = (
    "a968283e5f7a78e1010ec59c47aeb5002eff87962be05e38a0c20df41e24a900"
)
PUBLIC_PARTITION_MANIFEST_SHA256 = (
    "ba9816131647aefe0d813fc7c7d3706894988943bc58e92c472667f722528c20"
)
PUBLIC_PARTITION_ARTIFACT_SHA256 = {
    "records.jsonl": "31ceaa03cd41a108c2e7c8657580baa094d3180f08d0b64128d2ecb6f8dc32f9",
    "tensors.safetensors": "e1218b35260ea9e7afd048962b53c25257d7f298e2a787628d8f5b640cfc7e79",
}
PRIVATE_DATASET_ROOT_MANIFEST_SHA256 = (
    "ac53d03376b7a874c404a1bd0f5e5f84a8b96db7fc1a2187554c2c286a91c175"
)
PRIVATE_PARTITION_MANIFEST_SHA256 = {
    "private_cross_512": "554bf1261f72a848de3ac93d21e5c24a6581a52e0e802adea2454fac6d5d1369",
    "private_cross_1024": "b731a9642512720ac116b13ff88ccc8fa8a4751f94c7a7204d792e7e65fb6672",
    "private_cross_2048": "fea2455400c3461f9c5797d4f8288728b1cdfec16d2258a431215fe47b5f429d",
}
PRIVATE_PARTITION_ARTIFACT_SHA256 = {
    "private_cross_512": {
        "records.jsonl": "f87c5fbbc22e517c79513dabe020a23f25b671ef4b856edb2509775bfbceb327",
        "tensors.safetensors": "2968c2b87f03b9fb2ee57b69d5c8a702ed994cdb16889bb4bbfe40e9d104315b",
    },
    "private_cross_1024": {
        "records.jsonl": "f87c5fbbc22e517c79513dabe020a23f25b671ef4b856edb2509775bfbceb327",
        "tensors.safetensors": "1d347aa8f5047de37ca7f257439585a7caaff44179bc44c521190328f8883368",
    },
    "private_cross_2048": {
        "records.jsonl": "f87c5fbbc22e517c79513dabe020a23f25b671ef4b856edb2509775bfbceb327",
        "tensors.safetensors": "05cc2316e4db9af9bec8f13695a07760ab30a77b4062b99b6baf020c0277a0e8",
    },
}

DEFAULT_MODEL_HOST_PATH = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B"
)
DEFAULT_DATA_HOST_ROOT = (
    Path.home() / "mlsysforge-artifacts/qwen3-orpo/tldr_preference_v1"
)
DEFAULT_PRIVATE_DATA_HOST_ROOT = (
    Path.home() / "mlsysforge-artifacts/qwen3-orpo/tldr_preference_cross_v4"
)

GPU_UUID_ENV = "ORPO_GPU_UUIDS"
MODEL_HOST_ENV = "ORPO_MODEL_HOST_PATH"
DATA_HOST_ENV = "ORPO_DATA_HOST_ROOT"
PRIVATE_DATA_HOST_ENV = "ORPO_PRIVATE_DATA_HOST_ROOT"
REPLAY_MODEL_TREE_SHA_ENV = "ORPO_REPLAY_EXPECTED_MODEL_MOUNT_SHA256"
REPLAY_PRIVATE_DATA_TREE_SHA_ENV = (
    "ORPO_REPLAY_EXPECTED_PRIVATE_DATA_MOUNT_SHA256"
)
REPLAY_MODEL_HELPER_SHA_ENV = "ORPO_REPLAY_MODEL_IDENTITY_HELPER_SHA256"
REPLAY_RUNTIME_IDENTITY_PATH_ENV = "ORPO_REPLAY_RUNTIME_IDENTITY_PATH"
REPLAY_RUNTIME_IDENTITY_SHA_ENV = "ORPO_REPLAY_RUNTIME_IDENTITY_SHA256"
REPLAY_RUNTIME_IDENTITY_HELPER_SHA_ENV = (
    "ORPO_REPLAY_RUNTIME_IDENTITY_HELPER_SHA256"
)
REPLAY_GPU_INDICES_ENV = "ORPO_REPLAY_GPU_INDICES"
NETWORK_ENV = "SFORGE_ISOLATED_DOCKER_NETWORK"
MANAGED_JUDGE_LOCK_PATH = Path("/tmp/qwen3-orpo-dual-h100-managed-judge.lock")
MANAGED_WORK_IDLE_TIMEOUT_ENV = "ORPO_MANAGED_WORK_IDLE_TIMEOUT_SECONDS"
MANAGED_WORK_IDLE_TIMEOUT_SECONDS = 900.0
MANAGED_GPU_IDLE_MAX_USED_MIB = 256
TARGET_TASK_ID = "qwen3_1p7b_orpo_training_optimizer_dual_h100_v4"
MODEL_MOUNT_ROOT = "/models/Qwen3-1.7B-repo"
MODEL_MOUNT = f"{MODEL_MOUNT_ROOT}/snapshots/{MODEL_REVISION}"
DATASET_MOUNT_ROOT = "/datasets/tldr-preference"
REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
)
_GPU_UUID_RE = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
_managed_judge_thread_lock = threading.Lock()
_replay_guarded_container_ids: set[str] = set()
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ManagedJudgeLifecycleError(RuntimeError):
    """A safe, coarse managed-run lifecycle failure."""

    def __init__(self, failure_class: str):
        super().__init__(f"managed Judge lifecycle failed: {failure_class}")
        self.failure_class = failure_class


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replay_attestation_enabled() -> bool:
    return any(
        os.environ.get(name, "").strip()
        for name in (
            REPLAY_MODEL_TREE_SHA_ENV,
            REPLAY_PRIVATE_DATA_TREE_SHA_ENV,
            REPLAY_MODEL_HELPER_SHA_ENV,
            REPLAY_RUNTIME_IDENTITY_PATH_ENV,
            REPLAY_RUNTIME_IDENTITY_SHA_ENV,
            REPLAY_RUNTIME_IDENTITY_HELPER_SHA_ENV,
            REPLAY_GPU_INDICES_ENV,
        )
    )


def _verify_replay_live_artifacts() -> None:
    """Re-hash replay-mounted model/private data before and after each Judge."""

    expected_tree = os.environ.get(REPLAY_MODEL_TREE_SHA_ENV, "").strip()
    expected_private_tree = os.environ.get(
        REPLAY_PRIVATE_DATA_TREE_SHA_ENV, ""
    ).strip()
    expected_helper = os.environ.get(REPLAY_MODEL_HELPER_SHA_ENV, "").strip()
    runtime_baseline_raw = os.environ.get(
        REPLAY_RUNTIME_IDENTITY_PATH_ENV, ""
    ).strip()
    expected_runtime_baseline = os.environ.get(
        REPLAY_RUNTIME_IDENTITY_SHA_ENV, ""
    ).strip()
    expected_runtime_helper = os.environ.get(
        REPLAY_RUNTIME_IDENTITY_HELPER_SHA_ENV, ""
    ).strip()
    gpu_indices_raw = os.environ.get(REPLAY_GPU_INDICES_ENV, "").strip()
    if not any(
        (
            expected_tree,
            expected_private_tree,
            expected_helper,
            runtime_baseline_raw,
            expected_runtime_baseline,
            expected_runtime_helper,
            gpu_indices_raw,
        )
    ):
        return
    if (
        _SHA256_RE.fullmatch(expected_tree) is None
        or _SHA256_RE.fullmatch(expected_private_tree) is None
        or _SHA256_RE.fullmatch(expected_helper) is None
        or not runtime_baseline_raw
        or _SHA256_RE.fullmatch(expected_runtime_baseline) is None
        or _SHA256_RE.fullmatch(expected_runtime_helper) is None
    ):
        raise RuntimeError("incomplete ORPO replay attestation contract")
    gpu_indices = [value.strip() for value in gpu_indices_raw.split(",")]
    if (
        len(gpu_indices) != 2
        or any(re.fullmatch(r"0|[1-9][0-9]*", value) is None for value in gpu_indices)
        or len(set(gpu_indices)) != 2
    ):
        raise RuntimeError("incomplete ORPO replay attestation contract")
    model_helper = Path(__file__).with_name("model_artifact_identity.py")
    if (
        not model_helper.is_file()
        or model_helper.is_symlink()
        or _sha256(model_helper) != expected_helper
    ):
        raise RuntimeError("ORPO replay model identity helper changed")

    # These functions validate every private dataset artifact against the
    # frozen digests and return the exact canonical mount roots.
    model_repo = _model_source()
    private_data_root = _dataset_source("private")
    completed = subprocess.run(
        [
            sys.executable,
            str(model_helper),
            "--model-repo",
            str(model_repo),
            "--private-data-root",
            str(private_data_root),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0 or _sha256(model_helper) != expected_helper:
        raise RuntimeError("ORPO replay model attestation failed")
    try:
        identity = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ORPO replay model attestation was invalid") from exc
    if identity.get("model_mount_tree_sha256") != expected_tree:
        raise RuntimeError("ORPO replay model artifacts changed during the batch")
    if identity.get("private_data_mount_tree_sha256") != expected_private_tree:
        raise RuntimeError("ORPO replay private data changed during the batch")

    runtime_helper = Path(__file__).with_name("runtime_identity.py")
    if (
        not runtime_helper.is_file()
        or runtime_helper.is_symlink()
        or _sha256(runtime_helper) != expected_runtime_helper
    ):
        raise RuntimeError("ORPO replay runtime identity helper changed")
    runtime_baseline = Path(runtime_baseline_raw)
    if not runtime_baseline.is_absolute():
        raise RuntimeError("ORPO replay runtime identity path must be absolute")
    runtime_command = [
        sys.executable,
        str(runtime_helper),
        "--include-topology",
        "--verify-baseline",
        str(runtime_baseline),
        "--expected-baseline-sha256",
        expected_runtime_baseline,
    ]
    for index in gpu_indices:
        runtime_command.extend(("--gpu-index", index))
    for uuid in _gpu_uuids():
        runtime_command.extend(("--expected-gpu-uuid", uuid))
    runtime_completed = subprocess.run(
        runtime_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if (
        runtime_completed.returncode != 0
        or _sha256(runtime_helper) != expected_runtime_helper
    ):
        raise RuntimeError("ORPO replay host runtime identity changed")


def _resolved_directory(raw_path: str | Path, *, label: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{label} must be an absolute host path: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"required {label} is missing: {path}") from exc
    if not resolved.is_dir():
        raise RuntimeError(f"required {label} is not a directory: {resolved}")
    return resolved


def _checked_file(path: Path, *, expected_sha256: str, label: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"required {label} is not a regular file: {path}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return path


def _gpu_uuids(raw: str | None = None) -> list[str]:
    if raw is None:
        raw = os.environ.get(GPU_UUID_ENV, "")
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise RuntimeError(
            f"{GPU_UUID_ENV} is unset and has no default: set it to the two "
            f"physical GPU UUIDs of the NVLink pair this run owns, comma "
            f"separated (nvidia-smi --query-gpu=index,uuid --format=csv,noheader)"
        )
    if len(values) != 2:
        raise RuntimeError(
            f"{GPU_UUID_ENV} must contain exactly two explicit physical GPU UUIDs"
        )
    if any(_GPU_UUID_RE.fullmatch(value) is None for value in values):
        raise RuntimeError(
            f"{GPU_UUID_ENV} accepts only full GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx UUIDs"
        )
    if len({value.casefold() for value in values}) != 2:
        raise RuntimeError(f"{GPU_UUID_ENV} must name two distinct physical GPUs")
    return values


def _managed_idle_timeout() -> float:
    raw = os.environ.get(
        MANAGED_WORK_IDLE_TIMEOUT_ENV,
        str(MANAGED_WORK_IDLE_TIMEOUT_SECONDS),
    )
    try:
        value = float(raw)
    except ValueError as exc:
        raise ManagedJudgeLifecycleError("invalid_idle_timeout") from exc
    if not 1.0 <= value <= 3600.0:
        raise ManagedJudgeLifecycleError("invalid_idle_timeout")
    return value


def _query_target_gpu_idle(gpu_uuids: list[str]) -> bool:
    """Return whether the physical pair has no contexts and little memory use."""

    try:
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        usage = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ManagedJudgeLifecycleError("gpu_probe_failed") from exc

    targets = set(gpu_uuids)
    for line in apps.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0] in targets:
            return False

    observed: dict[str, int] = {}
    try:
        for line in usage.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and fields[0] in targets:
                observed[fields[0]] = int(fields[1])
    except ValueError as exc:
        raise ManagedJudgeLifecycleError("gpu_probe_failed") from exc
    if set(observed) != targets:
        raise ManagedJudgeLifecycleError("gpu_probe_failed")
    return all(
        used <= MANAGED_GPU_IDLE_MAX_USED_MIB for used in observed.values()
    )


def _work_container_name(task_id: str, run_id: str) -> str:
    if task_id != TARGET_TASK_ID or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ManagedJudgeLifecycleError("invalid_run_identity")
    return f"sforge.run.{task_id}.{run_id}"


def _container_status(container: Any) -> str:
    container.reload()
    status = getattr(container, "status", None)
    if not isinstance(status, str):
        raise ManagedJudgeLifecycleError("work_state_unavailable")
    return status


def _resume_work(container: Any, *, after_judge: bool = False) -> None:
    status = _container_status(container)
    if status == "paused":
        container.unpause()
        status = _container_status(container)
    if status != "running":
        # After a completed Judge round the deadline freeze may already have
        # stopped the paused Work container (``docker stop`` on a paused
        # container leaves it ``exited``).  The Judge result is complete and
        # authoritative at that point; failing the round here would discard
        # a valid report (observed on H100 and B200: ``work_resume_failed``
        # overwriting a 10/10 verdict).  Only an unpause that leaves the
        # container in a state other than running/exited is a lifecycle error.
        if after_judge and status in {"exited", "dead", "removing"}:
            return
        raise ManagedJudgeLifecycleError("work_resume_failed")


def _managed_judge_lock_path(gpu_uuids: list[str]) -> Path:
    """One lease per GPU pair: Judges on different pairs of the same host run in parallel."""

    first, second = sorted(uuid.lower() for uuid in gpu_uuids)
    return MANAGED_JUDGE_LOCK_PATH.with_name(
        f"{MANAGED_JUDGE_LOCK_PATH.stem}-{first}-{second}.lock"
    )


@contextmanager
def _exclusive_managed_judge(
    *, backend: Any, task_id: str, run_id: str
) -> Iterator[None]:
    """Quiesce and pause the exact Work container around one Judge call."""

    if getattr(backend, "backend_name", None) != "docker":
        raise ManagedJudgeLifecycleError("unsupported_backend")
    client = getattr(backend, "client", None)
    if client is None:
        raise ManagedJudgeLifecycleError("docker_client_unavailable")
    name = _work_container_name(task_id, run_id)
    timeout = _managed_idle_timeout()
    lock_path = _managed_judge_lock_path(_gpu_uuids())
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _managed_judge_thread_lock, lock_path.open("a+") as lock_file:
        flock(lock_file.fileno(), LOCK_EX)
        # The idle wait starts once this pair is ours, so queueing is never
        # reported as gpu_busy.
        deadline = time.monotonic() + timeout
        try:
            try:
                work = client.containers.get(name)
            except Exception as exc:
                raise ManagedJudgeLifecycleError("work_not_found") from exc
            paused = False
            primary_error: BaseException | None = None
            try:
                while True:
                    if _container_status(work) != "running":
                        raise ManagedJudgeLifecycleError("work_not_running")
                    if time.monotonic() >= deadline:
                        raise ManagedJudgeLifecycleError("gpu_busy")
                    if not _query_target_gpu_idle(_gpu_uuids()):
                        time.sleep(1.0)
                        continue

                    try:
                        work.pause()
                    except Exception as exc:
                        raise ManagedJudgeLifecycleError("work_pause_failed") from exc
                    paused = True
                    if _container_status(work) != "paused":
                        raise ManagedJudgeLifecycleError("work_pause_failed")

                    # Close the idle-check/pause race.  A CUDA process that
                    # began immediately before pause retains memory, so resume
                    # and retry rather than launching a contending Judge.
                    if _query_target_gpu_idle(_gpu_uuids()):
                        break
                    _resume_work(work)
                    paused = False
                    time.sleep(1.0)

                yield
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                if paused:
                    try:
                        _resume_work(work, after_judge=True)
                    except Exception:
                        if primary_error is None:
                            raise
        finally:
            flock(lock_file.fileno(), LOCK_UN)


def _install_managed_judge_wrapper() -> None:
    """Install task-local serialization around SForge's Judge entry point."""

    from sforge.harness import judge_server

    original = judge_server.judge_submission
    if getattr(original, "_orpo_managed_serialization", False):
        return

    def serialized_judge_submission(*args: Any, **kwargs: Any):
        task_spec = kwargs.get("task_spec")
        backend = kwargs.get("backend")
        run_id = kwargs.get("run_id")
        if (
            getattr(task_spec, "task_id", None) != TARGET_TASK_ID
            or run_id is None
        ):
            return original(*args, **kwargs)
        with _exclusive_managed_judge(
            backend=backend,
            task_id=task_spec.task_id,
            run_id=run_id,
        ):
            return original(*args, **kwargs)

    serialized_judge_submission._orpo_managed_serialization = True
    judge_server.judge_submission = serialized_judge_submission


def _model_source() -> Path:
    source = _resolved_directory(
        os.environ.get(MODEL_HOST_ENV, str(DEFAULT_MODEL_HOST_PATH)),
        label="Qwen3-1.7B Hugging Face cache repository",
    )
    if source.name != "models--Qwen--Qwen3-1.7B":
        raise RuntimeError(
            f"{MODEL_HOST_ENV} must resolve to the models--Qwen--Qwen3-1.7B cache root"
        )
    _resolved_directory(source / "blobs", label="Qwen3-1.7B cache blobs directory")
    snapshot = _resolved_directory(
        source / "snapshots" / MODEL_REVISION,
        label=f"Qwen3-1.7B snapshot revision {MODEL_REVISION}",
    )
    for filename in REQUIRED_MODEL_FILES:
        artifact = snapshot / filename
        try:
            resolved_artifact = artifact.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"required Qwen3-1.7B snapshot file is missing: {artifact}") from exc
        try:
            resolved_artifact.relative_to(source)
        except ValueError as exc:
            raise RuntimeError(
                f"Qwen3-1.7B snapshot link escapes its cache repository: {artifact}"
            ) from exc
        if not resolved_artifact.is_file():
            raise RuntimeError(
                f"required Qwen3-1.7B snapshot artifact is not a file: {artifact}"
            )
    config_path = _checked_file(
        snapshot / "config.json",
        expected_sha256=MODEL_CONFIG_SHA256,
        label="Qwen3-1.7B config.json",
    )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Qwen3-1.7B config.json is not valid UTF-8 JSON") from exc
    expected = {
        "model_type": "qwen3",
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    if not isinstance(config, dict) or any(
        config.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("Qwen3-1.7B config identity differs from the frozen contract")
    return source


def _dataset_source(partition: str) -> Path:
    if partition not in {"public", "private"}:
        raise RuntimeError(f"unsupported dataset partition: {partition}")
    if partition == "private":
        root = _resolved_directory(
            os.environ.get(PRIVATE_DATA_HOST_ENV, str(DEFAULT_PRIVATE_DATA_HOST_ROOT)),
            label="TLDR V4 private cross-matrix artifact root",
        )
        _checked_file(
            root / "freeze_manifest.json",
            expected_sha256=PRIVATE_DATASET_ROOT_MANIFEST_SHA256,
            label="TLDR V4 private root freeze manifest",
        )
        for subdir, expected_manifest in PRIVATE_PARTITION_MANIFEST_SHA256.items():
            source = _resolved_directory(root / subdir, label=f"TLDR V4 {subdir}")
            _checked_file(
                source / "manifest.json",
                expected_sha256=expected_manifest,
                label=f"TLDR V4 {subdir} manifest",
            )
            for filename, expected_sha256 in PRIVATE_PARTITION_ARTIFACT_SHA256[
                subdir
            ].items():
                _checked_file(
                    source / filename,
                    expected_sha256=expected_sha256,
                    label=f"TLDR V4 {subdir} {filename}",
                )
        return root
    root = _resolved_directory(
        os.environ.get(DATA_HOST_ENV, str(DEFAULT_DATA_HOST_ROOT)),
        label="TLDR public preference artifact root",
    )
    _checked_file(
        root / "freeze_manifest.json",
        expected_sha256=PUBLIC_DATASET_ROOT_MANIFEST_SHA256,
        label="TLDR public root freeze manifest",
    )
    source = _resolved_directory(
        root / partition,
        label=f"TLDR {partition} partition",
    )
    manifest_path = _checked_file(
        source / "manifest.json",
        expected_sha256=PUBLIC_PARTITION_MANIFEST_SHA256,
        label=f"TLDR {partition} manifest",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"TLDR {partition} manifest is not valid UTF-8 JSON") from exc
    provenance = manifest.get("provenance", {}) if isinstance(manifest, dict) else {}
    dataset = provenance.get("dataset", {}) if isinstance(provenance, dict) else {}
    tokenizer = provenance.get("tokenizer", {}) if isinstance(provenance, dict) else {}
    if (
        not isinstance(manifest, dict)
        or manifest.get("partition") != partition
        or dataset.get("repo_id") != "trl-lib/tldr-preference"
        or dataset.get("revision") != DATASET_REVISION
        or dataset.get("config") != "default"
        or tokenizer.get("repo_id") != "Qwen/Qwen3-1.7B"
        or tokenizer.get("revision") != MODEL_REVISION
    ):
        raise RuntimeError(f"TLDR {partition} manifest identity differs from the frozen contract")
    for filename, expected_sha256 in PUBLIC_PARTITION_ARTIFACT_SHA256.items():
        _checked_file(
            source / filename,
            expected_sha256=expected_sha256,
            label=f"TLDR {partition} {filename}",
        )
    return source


def _is_judge(image: str, name: str) -> bool:
    return ".judge." in image or ".judge." in name


def _work_network(self: DockerBackend) -> str:
    network_name = os.environ.get(NETWORK_ENV, "").strip()
    if not network_name:
        raise RuntimeError(f"{NETWORK_ENV} is required for the Work container")
    network = self.client.networks.get(network_name)
    network.reload()
    if network.attrs.get("Internal") is not True:
        raise RuntimeError(f"refusing non-internal Work network: {network_name}")
    return network_name


def dual_h100_create_container(
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
) -> DockerContainerHandle:
    """Create one role container with exactly two attested GPU UUIDs."""

    del annotations
    is_judge = _is_judge(image, name)
    replay_guard = is_judge and _replay_attestation_enabled()
    if replay_guard:
        _verify_replay_live_artifacts()
    partition = "private" if is_judge else "public"
    gpu_uuids = _gpu_uuids()
    model_source = _model_source()
    dataset_source = _dataset_source(partition)

    container_environment = dict(environment or {})
    container_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "NVIDIA_VISIBLE_DEVICES": ",".join(gpu_uuids),
            "ORPO_MODEL_PATH": MODEL_MOUNT,
            "ORPO_DATA_PATH": DATASET_MOUNT_ROOT,
            "ORPO_DATA_PARTITION": partition,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PIP_NO_INDEX": "1",
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "offline",
        }
    )
    volumes = {
        str(model_source): {"bind": MODEL_MOUNT_ROOT, "mode": "ro"},
        str(dataset_source): {
            "bind": DATASET_MOUNT_ROOT if is_judge else f"{DATASET_MOUNT_ROOT}/public",
            "mode": "ro",
        },
    }
    if mounts:
        # Only SForge's host-provided agent runtime uses this path: a
        # read-only bind mount into the untrusted Work container.  The
        # Judge mount plan stays exclusively task-owned, and nothing may
        # ever be writable from Work.
        if is_judge:
            raise RuntimeError(
                "Judge containers accept only the trusted task mount plan"
            )
        for host_path, container_path, mode in mounts:
            if mode != "ro":
                raise RuntimeError(
                    "Work containers accept only read-only bind mounts"
                )
            volumes[host_path] = {"bind": container_path, "mode": mode}
    kwargs: dict[str, Any] = {
        "volumes": volumes,
        "device_requests": [
            DeviceRequest(
                driver="nvidia",
                device_ids=gpu_uuids,
                capabilities=[["gpu"]],
            )
        ],
        "security_opt": ["no-new-privileges:true"],
    }
    if is_judge:
        kwargs["network_mode"] = "none"
    else:
        kwargs["network"] = _work_network(self)
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
        image,
        name=name,
        detach=True,
        command=command,
        environment=container_environment,
        **kwargs,
    )
    handle = DockerContainerHandle(container)
    if replay_guard:
        _replay_guarded_container_ids.add(handle.id)
    return handle


_original_cleanup_container = DockerBackend.cleanup_container


def dual_h100_cleanup_container(
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


DockerBackend.create_container = dual_h100_create_container
DockerBackend.cleanup_container = dual_h100_cleanup_container

CodexAgent.install_cmds = [
    'test "$(node --version)" = v20.18.0 && '
    'command -v codex >/dev/null && codex --version >/dev/null'
]
# The launcher always passes --enable-internet because SForge's own
# internet=false isolation needs sudo iptables, which this host does not
# grant; the internal network plus the borrowed llada sidecar is the
# enforcement boundary.  The task JSON's ``internet`` value reaches this entry
# through MLSYSFORGE_WORK_INTERNET ("true"/"false", exported by the launcher).
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
    # The launcher picks the Agent (MLSYSFORGE_AGENT=codex|claude-code); Claude
    # Code bindings: sforge/harness/agent/claude_local.py.
    from sforge.harness.agent.claude_local import install_from_environment

    _install_managed_judge_wrapper()
    install_from_environment(_task_internet, sys.argv)
    main()
