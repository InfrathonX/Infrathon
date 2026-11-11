#!/usr/bin/env python3
"""Judge-owned paired two-rank end-to-end controller.

The controller validates the frozen formal plan before resolving model,
dataset, or CUDA paths.  It can still emit a fail-closed dry-run report for an
explicitly disabled or incomplete manifest, while the released private
manifest enables the full paired evaluation.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, IO, Mapping, Sequence


WORLD_SIZE = 2
EXPECTED_MATRIX = {
    (2, 512): "private_cross_512",
    (2, 1024): "private_cross_1024",
    (2, 2048): "private_cross_2048",
    (4, 512): "private_cross_512",
    (8, 512): "private_cross_512",
}
EXPECTED_ROLE_ORDER = ("RC", "CR", "CR", "RC")
OFFICIAL_REFERENCE_ROLE = "official_hf_trl_reference"
FULL_NLL_TARGET_CONTRACT = "separate_full_sequence_tensor"
PENDING_PLAN_STATUS = "PENDING_CROSS_MATRIX_ARTIFACTS_AND_CALIBRATION"
ACTIVE_PLAN_STATUS = "FROZEN_FORMAL_CROSS_MATRIX"
EXPECTED_FSDP = {
    "api": "FSDP1",
    "sharding_strategy": "FULL_SHARD",
    "use_orig_params": True,
    "parameter_dtype": "bfloat16",
    "reduce_dtype": "bfloat16",
}
EXPECTED_UNIQUE_PARAMETERS = 1_720_574_976
EXPECTED_MODEL_REPO_ID = "Qwen/Qwen3-1.7B"
EXPECTED_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
EXPECTED_CORRECTNESS_AUX = (
    "chosen_logps",
    "rejected_logps",
    "chosen_logits_mean",
    "rejected_logits_mean",
    "nll_loss",
    "chosen_rewards",
    "rejected_rewards",
    "logsigmoid_ratio_mean",
    "log_odds_mean",
)
VECTOR_CORRECTNESS_AUX = {
    "chosen_logps",
    "rejected_logps",
    "chosen_rewards",
    "rejected_rewards",
}
EXPECTED_CORRECTNESS_AUX_DTYPES = {
    name: (
        "torch.bfloat16"
        if name in {"chosen_logits_mean", "rejected_logits_mean"}
        else "torch.float32"
    )
    for name in EXPECTED_CORRECTNESS_AUX
}
CORRECTNESS_CAPTURE_POINT = "warmup_after_backward_before_optimizer_step"
MAX_TELEMETRY_BYTES = 8 * 1024 * 1024
# Capability envelope: what the paired training contract actually needs, not the
# GPU model it was qualified on.  Identical to the constants in the trusted
# runtime (``orpo_lab/runtime/distributed_probe.py``); tests assert the pair.
# Device name and UUID stay pure provenance and never gate admission.
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 60 * 1024**3
# Device-relative peak-memory ceiling.  The caching allocator's reserved pool has
# to leave room for what lives outside it on any card: the CUDA context, the NCCL
# communication buffers and cuBLAS workspaces, plus fragmentation slack.  That
# cost is a property of the two-rank job, not of the card, so the ceiling is
# ``min(rank total_memory) - DEVICE_MEMORY_HEADROOM_BYTES`` rather than a byte
# count measured on one GPU model.
DEVICE_MEMORY_HEADROOM_BYTES = 4 * 1024**3
DEFAULT_RANK_BOOTSTRAP = Path("/opt/orpo_candidate_worker/e2e_rank_bootstrap.py")
ALLOWED_CANDIDATE_ENVIRONMENT = {
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TOKENIZERS_PARALLELISM",
    "TRITON_CACHE_DIR",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_BLOCKING_WAIT",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "TORCH_HOME",
    "HF_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "CUDA_CACHE_PATH",
}


class PlanError(ValueError):
    """The private E2E plan is malformed or internally inconsistent."""


class WorkerTimeoutError(RuntimeError):
    """A fresh two-rank worker exceeded its private timeout."""


class WorkerNonzeroExitError(RuntimeError):
    """A fresh two-rank worker exited unsuccessfully."""


class WorkerTelemetryError(RuntimeError):
    """A worker did not produce readable bounded telemetry."""


ProcessRunner = Callable[..., int]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _require_positive_int(row: Mapping[str, Any], name: str, row_index: int) -> int:
    value = row.get(name)
    if not _is_int(value) or value <= 0:
        raise PlanError(f"e2e row {row_index} has invalid {name}")
    return value


def _require_digest(row: Mapping[str, Any], name: str, row_index: int) -> str:
    value = row.get(name)
    if not _is_sha256(value):
        raise PlanError(f"e2e row {row_index} has invalid {name}")
    assert isinstance(value, str)
    return value


def _optional_digest(row: Mapping[str, Any], name: str) -> str | None:
    value = row.get(name)
    if value is None:
        return None
    return value if _is_sha256(value) else None


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value.lower() == value
        and all(character in "0123456789abcdef" for character in value)
    )


def _inspect_row(row: Any, row_index: int) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(row, dict):
        raise PlanError(f"e2e row {row_index} is not an object")
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise PlanError(f"e2e row {row_index} has no private identifier")
    classification = row.get("classification")
    if classification not in {
        "HIDDEN_TLDR_E2E_CALIBRATION_ONLY",
        "HIDDEN_TLDR_E2E_FORMAL",
    }:
        raise PlanError(f"e2e row {row_index} has an unexpected classification")
    status = row.get("status")
    if not isinstance(status, str) or not status:
        raise PlanError(f"e2e row {row_index} has no status")
    if not isinstance(row.get("formal"), bool) or not isinstance(
        row.get("score_eligible"), bool
    ):
        raise PlanError(f"e2e row {row_index} has invalid readiness flags")
    if row.get("dataset_partition") != "private":
        raise PlanError(f"e2e row {row_index} must select the private partition")
    if row.get("world_size") != WORLD_SIZE:
        raise PlanError(f"e2e row {row_index} must use exactly two ranks")
    pair_batch = _require_positive_int(row, "pair_batch_per_rank", row_index)
    sequence_length = _require_positive_int(row, "sequence_length", row_index)
    warmup_steps = _require_positive_int(row, "warmup_steps", row_index)
    timed_steps = _require_positive_int(row, "timed_steps", row_index)
    repeats = _require_positive_int(row, "paired_repeats", row_index)
    matrix_key = (pair_batch, sequence_length)
    expected_subdir = EXPECTED_MATRIX.get(matrix_key)
    if expected_subdir is None:
        raise PlanError(f"e2e row {row_index} is outside the frozen cross matrix")
    dataset_subdir = row.get("dataset_subdir")
    if dataset_subdir != expected_subdir:
        raise PlanError(f"e2e row {row_index} selects the wrong dataset subdirectory")
    if row.get("nll_target_contract") != FULL_NLL_TARGET_CONTRACT:
        raise PlanError(f"e2e row {row_index} must use a separate full-sequence NLL target")
    if warmup_steps < 2:
        raise PlanError(
            f"e2e row {row_index} needs a final plain warm-up after correctness capture"
        )
    if warmup_steps > 10 or timed_steps > 20:
        raise PlanError(f"e2e row {row_index} exceeds the driver safety bounds")
    if repeats != len(EXPECTED_ROLE_ORDER) or tuple(row.get("paired_role_order", ())) != (
        EXPECTED_ROLE_ORDER
    ):
        raise PlanError(f"e2e row {row_index} has an invalid paired role order")

    normalized = {
        "private_id": row_id,
        "classification": classification,
        "status": status,
        "formal": row["formal"],
        "score_eligible": row["score_eligible"],
        "dataset_partition": "private",
        "dataset_subdir": dataset_subdir,
        "dataset_manifest_sha256": _optional_digest(row, "dataset_manifest_sha256"),
        "tensors_sha256": _optional_digest(row, "tensors_sha256"),
        "pair_batch_per_rank": pair_batch,
        "sequence_length": sequence_length,
        "nll_target_contract": FULL_NLL_TARGET_CONTRACT,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "paired_role_order": EXPECTED_ROLE_ORDER,
    }

    pending: list[str] = []
    prefix = f"e2e_rows[{row_index}]"
    artifact_status = row.get("dataset_artifact_status")
    if artifact_status != "FROZEN":
        pending.append(f"{prefix}.dataset_artifact_status")
    if normalized["dataset_manifest_sha256"] is None:
        pending.append(f"{prefix}.dataset_manifest_sha256")
    if normalized["tensors_sha256"] is None:
        pending.append(f"{prefix}.tensors_sha256")
    if status != ACTIVE_PLAN_STATUS:
        pending.append(f"{prefix}.status")
    if row["formal"] is not True:
        pending.append(f"{prefix}.formal")
    if classification != "HIDDEN_TLDR_E2E_FORMAL":
        pending.append(f"{prefix}.classification")
    if row["score_eligible"] is not True:
        pending.append(f"{prefix}.score_eligible")

    seed = row.get("seed")
    if not _is_int(seed) or seed < 0 or seed >= 2**63:
        pending.append(f"{prefix}.seed")
    else:
        normalized["seed"] = seed

    loss = row.get("loss_tolerance")
    if not isinstance(loss, dict) or loss.get("frozen") is not True:
        pending.append(f"{prefix}.loss_tolerance.frozen")
    else:
        rtol = _finite(loss.get("rtol"))
        atol = _finite(loss.get("atol"))
        if rtol is None or atol is None or rtol < 0.0 or atol < 0.0:
            pending.append(f"{prefix}.loss_tolerance.values")
        else:
            normalized["loss_rtol"] = rtol
            normalized["loss_atol"] = atol

    correctness = row.get("correctness_tolerance")
    if not isinstance(correctness, dict) or correctness.get("frozen") is not True:
        pending.append(f"{prefix}.correctness_tolerance.frozen")
    else:
        grouped_tolerances: dict[str, dict[str, float]] = {}
        for group in ("aux", "gradient", "optimizer"):
            tolerance = correctness.get(group)
            rtol = _finite(tolerance.get("rtol")) if isinstance(tolerance, dict) else None
            atol = _finite(tolerance.get("atol")) if isinstance(tolerance, dict) else None
            if rtol is None or atol is None or rtol < 0.0 or atol < 0.0:
                pending.append(f"{prefix}.correctness_tolerance.{group}")
            else:
                grouped_tolerances[group] = {"rtol": rtol, "atol": atol}
        if len(grouped_tolerances) == 3:
            normalized["correctness_tolerance"] = grouped_tolerances

    memory = row.get("memory_gate")
    if not isinstance(memory, dict) or memory.get("frozen") is not True:
        pending.append(f"{prefix}.memory_gate.frozen")
    else:
        ratio = _finite(
            memory.get("max_candidate_over_reference_peak_reserved_ratio")
        )
        cap = memory.get("absolute_candidate_peak_reserved_bytes_cap")
        if ratio is None or ratio <= 0.0:
            pending.append(f"{prefix}.memory_gate.ratio")
        else:
            normalized["memory_ratio_cap"] = ratio
        if not _is_int(cap) or cap <= 0:
            pending.append(f"{prefix}.memory_gate.absolute_cap")
        else:
            # Provenance, not a gate input.  This byte count was measured on the
            # qualification card; the enforced ceiling is device-relative
            # (``min rank total_memory - DEVICE_MEMORY_HEADROOM_BYTES``) and is
            # computed per run in ``_compare_pair``.  The field stays in the
            # frozen manifest schema so the qualification number remains
            # machine-readable and auditable.
            normalized["memory_absolute_cap_provenance"] = cap
    return normalized, pending


def inspect_e2e_plan(manifest: Any) -> dict[str, Any]:
    """Validate the calibration schema and identify every activation gate."""

    if not isinstance(manifest, dict) or manifest.get("schema_version") != 4:
        raise PlanError("private manifest schema_version must equal 4")
    formal_ready = manifest.get("formal_ready")
    score_enabled = manifest.get("formal_score_enabled")
    if not isinstance(formal_ready, bool) or not isinstance(score_enabled, bool):
        raise PlanError("private manifest readiness flags must be booleans")
    if manifest.get("role_legend") != {
        "R": OFFICIAL_REFERENCE_ROLE,
        "C": "candidate",
    }:
        raise PlanError("private manifest role legend is invalid")
    rows = manifest.get("e2e_rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_MATRIX):
        raise PlanError("private manifest must contain exactly five e2e rows")

    normalized_rows: list[dict[str, Any]] = []
    pending: list[str] = []
    identifiers: set[str] = set()
    for row_index, row in enumerate(rows):
        normalized, row_pending = _inspect_row(row, row_index)
        if normalized["private_id"] in identifiers:
            raise PlanError("private e2e row identifiers must be unique")
        identifiers.add(normalized["private_id"])
        normalized_rows.append(normalized)
        pending.extend(row_pending)
    actual_matrix = {
        (row["pair_batch_per_rank"], row["sequence_length"]): row["dataset_subdir"]
        for row in normalized_rows
    }
    if actual_matrix != EXPECTED_MATRIX:
        raise PlanError("private e2e rows do not match the five-row cross matrix")
    matrix_contract = manifest.get("matrix_contract")
    if not isinstance(matrix_contract, dict) or matrix_contract.get("unique_row_count") != 5:
        raise PlanError("private manifest has an invalid cross-matrix contract")

    if formal_ready is not True:
        pending.append("formal_ready")
    if score_enabled is not True:
        pending.append("formal_score_enabled")
    plan_status = manifest.get("workload_plan_status")
    if plan_status != ACTIVE_PLAN_STATUS:
        pending.append("workload_plan_status")
    if manifest.get("formal_thresholds_frozen") is not True:
        pending.append("formal_thresholds_frozen")
    if formal_ready != score_enabled:
        pending.append("readiness_flags_consistent")

    return {
        "schema_ok": True,
        "formal_requested": bool(formal_ready or score_enabled),
        "formal_ready_flag": formal_ready,
        "formal_score_enabled_flag": score_enabled,
        "activation_ready": bool(formal_ready and score_enabled and not pending),
        "workload_plan_status": plan_status,
        "reference_role": OFFICIAL_REFERENCE_ROLE,
        "pending_gates": sorted(set(pending)),
        "rows": normalized_rows,
    }


def _ensure_private_log_dir(log_dir: Path) -> None:
    if log_dir.is_symlink():
        raise PermissionError("formal log directory must not be a symlink")
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = log_dir.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise PermissionError("formal log path is not a directory")
    if os.geteuid() == 0:
        os.chown(log_dir, 0, 0, follow_symlinks=False)
    elif info.st_uid != os.geteuid():
        raise PermissionError("formal log directory is not owned by the controller")
    os.chmod(log_dir, 0o700)


def _open_private_log(path: Path) -> IO[bytes]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    if os.geteuid() == 0:
        os.fchown(descriptor, 0, 0)
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "wb")


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if os.geteuid() == 0:
            os.fchown(descriptor, 0, 0)
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, (rendered + "\n").encode("utf-8"))
    finally:
        os.close(descriptor)


def _read_worker_json(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_TELEMETRY_BYTES:
            raise RuntimeError("worker telemetry is absent or exceeds the size limit")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            payload = json.load(handle)
    finally:
        os.close(descriptor)
    if not isinstance(payload, dict):
        raise RuntimeError("worker telemetry must be a JSON object")
    return payload


def _default_process_runner(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout: IO[bytes],
    stderr: IO[bytes],
    timeout_seconds: float,
    launch_options: Mapping[str, Any],
) -> int:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        **dict(launch_options),
    )
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise WorkerTimeoutError("two-rank worker exceeded its private timeout") from exc


def _validate_candidate_launch(
    launch_options: Mapping[str, Any], environment: Mapping[str, str]
) -> None:
    required_options = {
        "user",
        "group",
        "extra_groups",
        "umask",
        "start_new_session",
        "close_fds",
    }
    if set(launch_options) != required_options:
        raise PermissionError("candidate torchrun launch options are not canonical")
    user = launch_options.get("user")
    group = launch_options.get("group")
    if not _is_int(user) or not _is_int(group) or user <= 0 or group <= 0:
        raise PermissionError("candidate torchrun requires a non-root UID and GID")
    if launch_options.get("extra_groups") != ():
        raise PermissionError("candidate torchrun supplementary groups must be empty")
    if launch_options.get("start_new_session") is not True:
        raise PermissionError("candidate torchrun must start a new session")
    if launch_options.get("close_fds") is not True:
        raise PermissionError("candidate torchrun must close unrelated descriptors")
    if launch_options.get("umask") != 0o077:
        raise PermissionError("candidate torchrun must use a private umask")
    unexpected = set(environment) - ALLOWED_CANDIDATE_ENVIRONMENT
    if unexpected:
        raise PermissionError("candidate torchrun environment contains unapproved keys")


def _build_command(
    *,
    kind: str,
    row: Mapping[str, Any],
    lab_root: Path,
    model_path: Path,
    data_path: Path,
    output_path: Path,
    rank_bootstrap: Path,
    candidate_version: int,
) -> list[str]:
    command = [
        sys.executable,
        "-I",
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--node-rank=0",
        "--nproc-per-node=2",
        "--master-addr=127.0.0.1",
        "--master-port=29500",
        "--max-restarts=0",
        str(rank_bootstrap),
        "--runtime-root",
        str(lab_root / "runtime"),
        "--",
        "--kind",
        kind,
        "--model-path",
        str(model_path),
        "--lab-root",
        str(lab_root),
        "--output",
        str(output_path),
        "--data-path",
        str(data_path),
        "--expected-data-partition",
        str(row["dataset_partition"]),
        "--expected-manifest-sha256",
        str(row["dataset_manifest_sha256"]),
        "--expected-tensors-sha256",
        str(row["tensors_sha256"]),
        "--seed",
        str(row["seed"]),
        "--pair-batch",
        str(row["pair_batch_per_rank"]),
        "--sequence-length",
        str(row["sequence_length"]),
        "--warmup-steps",
        str(row["warmup_steps"]),
        "--timed-steps",
        str(row["timed_steps"]),
        "--learning-rate",
        "1e-6",
        "--beta",
        "0.1",
    ]
    if kind == "candidate":
        command.extend(["--candidate-version", str(candidate_version)])
    return command


def _launch_cohort(
    *,
    kind: str,
    row: Mapping[str, Any],
    cohort_ordinal: int,
    lab_root: Path,
    model_path: Path,
    data_path: Path,
    log_dir: Path,
    rank_bootstrap: Path,
    candidate_version: int,
    candidate_environment: Mapping[str, str],
    reference_environment: Mapping[str, str],
    candidate_launch_options: Mapping[str, Any],
    timeout_seconds: float,
    process_runner: ProcessRunner,
    scratch_parent: Path | None,
) -> dict[str, Any]:
    scratch_guard: Path | None = None
    if scratch_parent is None:
        scratch_guard = Path(tempfile.mkdtemp(prefix="orpo-formal-controller-"))
        # Candidate ranks may traverse to their owned child but cannot rename
        # that child or create siblings in this controller-owned directory.
        os.chmod(scratch_guard, 0o711)
        scratch_parent = scratch_guard
    scratch = Path(tempfile.mkdtemp(prefix="worker-", dir=scratch_parent))
    try:
        if kind == "candidate" and os.geteuid() == 0:
            os.chown(
                scratch,
                int(candidate_launch_options["user"]),
                int(candidate_launch_options["group"]),
            )
        os.chmod(scratch, 0o700)
        output_path = scratch / "telemetry.json"
        command = _build_command(
            kind=kind,
            row=row,
            lab_root=lab_root,
            model_path=model_path,
            data_path=data_path,
            output_path=output_path,
            rank_bootstrap=rank_bootstrap,
            candidate_version=candidate_version,
        )
        private_directories = {
            name: scratch / relative
            for name, relative in {
                "TRITON_CACHE_DIR": "triton-cache",
                "HOME": "home",
                "TMPDIR": "tmp",
                "XDG_CACHE_HOME": "xdg-cache",
                "TORCH_HOME": "torch-cache",
                "HF_HOME": "hf-cache",
                "TORCHINDUCTOR_CACHE_DIR": "inductor-cache",
                "CUDA_CACHE_PATH": "cuda-cache",
            }.items()
        }
        for path in private_directories.values():
            path.mkdir(mode=0o700)
            if kind == "candidate" and os.geteuid() == 0:
                os.chown(
                    path,
                    int(candidate_launch_options["user"]),
                    int(candidate_launch_options["group"]),
                )
            os.chmod(path, 0o700)
        environment = dict(
            candidate_environment if kind == "candidate" else reference_environment
        )
        # Never share a root-created compiler cache with the low-privilege
        # cohort.  The random scratch path contains no private row identifier.
        environment.update(
            {name: str(path) for name, path in private_directories.items()}
        )
        environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        environment["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        launch_options: Mapping[str, Any] = (
            candidate_launch_options
            if kind == "candidate"
            else {"start_new_session": True, "close_fds": True, "umask": 0o077}
        )
        stdout_path = log_dir / f"cohort-{cohort_ordinal:03d}.stdout.log"
        stderr_path = log_dir / f"cohort-{cohort_ordinal:03d}.stderr.log"
        with _open_private_log(stdout_path) as stdout, _open_private_log(
            stderr_path
        ) as stderr:
            returncode = process_runner(
                command,
                cwd=rank_bootstrap.parent,
                environment=environment,
                stdout=stdout,
                stderr=stderr,
                timeout_seconds=timeout_seconds,
                launch_options=launch_options,
            )
        if returncode != 0:
            raise WorkerNonzeroExitError(
                "two-rank worker failed; consult root-only logs"
            )
        try:
            payload = _read_worker_json(output_path)
        except Exception as exc:
            raise WorkerTelemetryError("two-rank worker telemetry is unavailable") from exc
        _write_private_json(
            log_dir / f"cohort-{cohort_ordinal:03d}.telemetry.json", payload
        )
        return payload
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if scratch_guard is not None:
            shutil.rmtree(scratch_guard, ignore_errors=True)


def _positive_number(payload: Mapping[str, Any], name: str) -> float:
    value = _finite(payload.get(name))
    if value is None or value <= 0.0:
        raise RuntimeError(f"worker emitted invalid {name}")
    return value


def _positive_integer_list(payload: Mapping[str, Any], name: str, size: int) -> list[int]:
    values = payload.get(name)
    if (
        not isinstance(values, list)
        or len(values) != size
        or any(not _is_int(value) or value <= 0 for value in values)
    ):
        raise RuntimeError(f"worker emitted invalid {name}")
    return values


def _finite_number_list(payload: Mapping[str, Any], name: str, size: int) -> list[float]:
    values = payload.get(name)
    if not isinstance(values, list) or len(values) != size:
        raise RuntimeError(f"worker emitted invalid {name}")
    normalized = [_finite(value) for value in values]
    if any(value is None for value in normalized):
        raise RuntimeError(f"worker emitted non-finite {name}")
    return [float(value) for value in normalized if value is not None]


def _validate_model_optimization(payload: Mapping[str, Any], expected_kind: str) -> None:
    report = payload.get("model_optimization")
    if not isinstance(report, dict) or report.get("abi") != "Qwen3ModelPatch":
        raise RuntimeError("worker model-optimization telemetry is invalid")
    patch = report.get("patch_spec")
    if not isinstance(patch, dict):
        raise RuntimeError("worker model-optimization subreports are invalid")
    # Imperative hooks can preserve topology while mutating parameter values.
    # Formal cohorts therefore accept only no hook or the task-owned
    # declarative patch installation path.
    if "optimize_model" in report or "host_internal_optimizer" in report:
        raise RuntimeError("formal worker exported an imperative model optimizer")
    if expected_kind == "reference" or patch.get("exported") is False:
        if report != {
            "abi": "Qwen3ModelPatch",
            "exported": False,
            "invoked": False,
            "return": "not_exported",
            "invariants_validated": False,
            "patch_spec": {"exported": False, "applied": False},
        }:
            raise RuntimeError("worker no-hook model report is not canonical")
        return
    fields = patch.get("fields")
    if (
        expected_kind != "candidate"
        or patch.get("applied") is not True
        or not isinstance(fields, list)
        or not fields
        or len(fields) != len(set(fields))
        or not set(fields) <= {"rope", "rms_norm_forward", "swiglu_forward"}
        or patch.get("attribute_writes_owned_by") != "task_runtime"
        or report.get("exported") is not True
        or report.get("invoked") is not True
        or report.get("return") != "patch_spec_only"
        or report.get("invariants_validated") is not True
    ):
        raise RuntimeError("worker declarative model patch report is invalid")
    if "rms_norm_forward" in fields and patch.get("norm_module_count") != 113:
        raise RuntimeError("worker did not patch the complete Qwen3 norm set")
    if "swiglu_forward" in fields and patch.get("mlp_module_count") != 28:
        raise RuntimeError("worker did not patch the complete Qwen3 MLP set")
    invariants = report.get("invariants")
    if not isinstance(invariants, dict) or set(invariants.values()) != {True}:
        raise RuntimeError("worker model invariants were not all validated")


def _correctness_number(value: Any, name: str) -> float:
    number = _finite(value)
    if number is None:
        raise RuntimeError(f"worker correctness telemetry has invalid {name}")
    return number


def _correctness_positive_integer(value: Any, name: str) -> int:
    if not _is_int(value) or value <= 0:
        raise RuntimeError(f"worker correctness telemetry has invalid {name}")
    return value


def _validate_correctness_telemetry(
    payload: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    telemetry = payload.get("correctness_telemetry")
    if not isinstance(telemetry, dict) or telemetry.get("schema_version") != 1:
        raise RuntimeError("worker omitted correctness telemetry schema v1")
    if telemetry.get("capture_point") != CORRECTNESS_CAPTURE_POINT:
        raise RuntimeError("worker correctness capture point is invalid")
    warmup_steps = int(row["warmup_steps"])
    capture_index = warmup_steps - 2
    expected_step = capture_index + 1
    if telemetry.get("capture_warmup_step_index") != capture_index:
        raise RuntimeError("worker correctness capture did not precede the final warm-up")
    if telemetry.get("capture_expected_optimizer_step") != expected_step:
        raise RuntimeError("worker correctness capture has an invalid optimizer step")
    if (
        telemetry.get("memory_qualification") != "settled_after_final_plain_warmup"
        or telemetry.get("post_capture_allocator_settled") is not True
    ):
        raise RuntimeError("worker correctness capture did not re-settle the allocator")

    aux = telemetry.get("aux")
    if not isinstance(aux, list) or len(aux) != len(EXPECTED_CORRECTNESS_AUX):
        raise RuntimeError("worker correctness auxiliary evidence is incomplete")
    normalized_aux: list[dict[str, Any]] = []
    for expected_name, entry in zip(EXPECTED_CORRECTNESS_AUX, aux, strict=True):
        if not isinstance(entry, dict) or entry.get("name") != expected_name:
            raise RuntimeError("worker correctness auxiliary names are not canonical")
        shape = entry.get("shape_per_rank")
        dtype = entry.get("dtype_per_rank")
        expected_shape = (
            [int(row["pair_batch_per_rank"])]
            if expected_name in VECTOR_CORRECTNESS_AUX
            else []
        )
        if (
            shape != expected_shape
            or dtype != EXPECTED_CORRECTNESS_AUX_DTYPES[expected_name]
            or entry.get("finite") is not True
        ):
            raise RuntimeError("worker correctness auxiliary structure is invalid")
        global_numel = _correctness_positive_integer(
            entry.get("global_numel"), f"aux.{expected_name}.global_numel"
        )
        local_numel = math.prod(shape) if shape else 1
        if global_numel != WORLD_SIZE * local_numel:
            raise RuntimeError("worker correctness auxiliary element count is invalid")
        normalized = {
            "name": expected_name,
            "shape_per_rank": shape,
            "dtype_per_rank": dtype,
            "global_numel": global_numel,
            **{
                metric: _correctness_number(
                    entry.get(metric), f"aux.{expected_name}.{metric}"
                )
                for metric in ("global_sum", "global_l2_sq", "global_min", "global_max")
            },
        }
        if normalized["global_l2_sq"] < 0.0 or normalized["global_min"] > normalized[
            "global_max"
        ]:
            raise RuntimeError("worker correctness auxiliary statistics are invalid")
        normalized_aux.append(normalized)

    gradients = telemetry.get("gradients")
    if (
        not isinstance(gradients, dict)
        or gradients.get("finite") is not True
        or gradients.get("method")
        != "FSDP.clip_grad_norm_(max_norm=inf,norm_type=2)"
    ):
        raise RuntimeError("worker correctness gradient evidence is invalid")
    normalized_gradients = {
        "global_l2_norm": _correctness_number(
            gradients.get("global_l2_norm"), "gradients.global_l2_norm"
        ),
        **{
            name: _correctness_positive_integer(
                gradients.get(name), f"gradients.{name}"
            )
            for name in (
                "global_trainable_parameter_tensors",
                "global_trainable_parameter_elements",
                "global_gradient_tensors",
                "global_gradient_elements",
            )
        },
    }
    if (
        normalized_gradients["global_l2_norm"] <= 0.0
        or normalized_gradients["global_gradient_tensors"]
        != normalized_gradients["global_trainable_parameter_tensors"]
        or normalized_gradients["global_gradient_elements"]
        != normalized_gradients["global_trainable_parameter_elements"]
    ):
        raise RuntimeError("worker gradients do not cover every trainable shard")

    update = telemetry.get("optimizer_update")
    if (
        telemetry.get("optimizer_class") != "AdamW"
        or not isinstance(update, dict)
        or update.get("finite") is not True
        or update.get("method")
        != "trusted_AdamW_state_and_compact_parameter_signature"
    ):
        raise RuntimeError("worker optimizer-update evidence is invalid")

    def parameter_signature(name: str) -> dict[str, Any]:
        signature = update.get(name)
        if not isinstance(signature, dict):
            raise RuntimeError("worker optimizer parameter signature is missing")
        normalized_signature = {
            "global_numel": _correctness_positive_integer(
                signature.get("global_numel"), f"optimizer.{name}.global_numel"
            ),
            "global_sum": _correctness_number(
                signature.get("global_sum"), f"optimizer.{name}.global_sum"
            ),
            "global_l2_sq": _correctness_number(
                signature.get("global_l2_sq"), f"optimizer.{name}.global_l2_sq"
            ),
        }
        if normalized_signature["global_l2_sq"] <= 0.0:
            raise RuntimeError("worker optimizer parameter signature is invalid")
        return normalized_signature

    before = parameter_signature("parameter_before")
    after = parameter_signature("parameter_after")
    if before["global_numel"] != after["global_numel"]:
        raise RuntimeError("worker optimizer update changed parameter topology")
    normalized_update = {
        "expected_step": _correctness_positive_integer(
            update.get("expected_step"), "optimizer.expected_step"
        ),
        **{
            name: _correctness_positive_integer(
                update.get(name), f"optimizer.{name}"
            )
            for name in (
                "global_active_parameter_tensors",
                "global_state_entries",
                "global_entries_at_expected_step",
            )
        },
        "minimum_step": _correctness_number(
            update.get("minimum_step"), "optimizer.minimum_step"
        ),
        "maximum_step": _correctness_number(
            update.get("maximum_step"), "optimizer.maximum_step"
        ),
        "global_exp_avg_l2_norm": _correctness_number(
            update.get("global_exp_avg_l2_norm"), "optimizer.global_exp_avg_l2_norm"
        ),
        "parameter_global_sum_delta": _correctness_number(
            update.get("parameter_global_sum_delta"),
            "optimizer.parameter_global_sum_delta",
        ),
        "parameter_global_l2_sq_delta": _correctness_number(
            update.get("parameter_global_l2_sq_delta"),
            "optimizer.parameter_global_l2_sq_delta",
        ),
        "parameter_signature_changed": update.get("parameter_signature_changed"),
        "parameter_before": before,
        "parameter_after": after,
    }
    if (
        normalized_update["expected_step"] != expected_step
        or normalized_update["minimum_step"] != float(expected_step)
        or normalized_update["maximum_step"] != float(expected_step)
        or normalized_update["global_state_entries"]
        != normalized_update["global_active_parameter_tensors"]
        or normalized_update["global_entries_at_expected_step"]
        != normalized_update["global_active_parameter_tensors"]
        or normalized_update["global_exp_avg_l2_norm"] <= 0.0
        or normalized_update["parameter_signature_changed"] is not True
        or normalized_update["global_active_parameter_tensors"]
        != normalized_gradients["global_trainable_parameter_tensors"]
        or before["global_numel"]
        != normalized_gradients["global_trainable_parameter_elements"]
        or before["global_numel"] != EXPECTED_UNIQUE_PARAMETERS
    ):
        raise RuntimeError("worker AdamW state did not record every expected step")
    expected_sum_delta = after["global_sum"] - before["global_sum"]
    expected_l2_delta = after["global_l2_sq"] - before["global_l2_sq"]
    if (
        normalized_update["parameter_global_sum_delta"] != expected_sum_delta
        or normalized_update["parameter_global_l2_sq_delta"] != expected_l2_delta
        or (
            expected_sum_delta == 0.0
            and expected_l2_delta == 0.0
        )
    ):
        raise RuntimeError("worker parameter update witness is inconsistent or empty")

    final = telemetry.get("final_optimizer_update")
    if not isinstance(final, dict) or final.get("finite") is not True:
        raise RuntimeError("worker final optimizer-update witness is missing")

    def final_parameter_signature(name: str) -> dict[str, Any]:
        signature = final.get(name)
        if not isinstance(signature, dict):
            raise RuntimeError("worker final optimizer parameter signature is missing")
        normalized_signature = {
            "global_numel": _correctness_positive_integer(
                signature.get("global_numel"), f"final.{name}.global_numel"
            ),
            "global_sum": _correctness_number(
                signature.get("global_sum"), f"final.{name}.global_sum"
            ),
            "global_l2_sq": _correctness_number(
                signature.get("global_l2_sq"), f"final.{name}.global_l2_sq"
            ),
        }
        if normalized_signature["global_l2_sq"] <= 0.0:
            raise RuntimeError("worker final optimizer parameter signature is invalid")
        return normalized_signature

    final_capture = final_parameter_signature("parameter_at_capture_after")
    final_parameter = final_parameter_signature("parameter_final")
    normalized_final = {
        "expected_step": _correctness_positive_integer(
            final.get("expected_step"), "final.expected_step"
        ),
        **{
            name: _correctness_positive_integer(final.get(name), f"final.{name}")
            for name in (
                "global_active_parameter_tensors",
                "global_state_entries",
                "global_entries_at_expected_step",
            )
        },
        "minimum_step": _correctness_number(
            final.get("minimum_step"), "final.minimum_step"
        ),
        "maximum_step": _correctness_number(
            final.get("maximum_step"), "final.maximum_step"
        ),
        "global_exp_avg_l2_norm": _correctness_number(
            final.get("global_exp_avg_l2_norm"), "final.global_exp_avg_l2_norm"
        ),
        "global_exp_avg_sq_l2_norm": _correctness_number(
            final.get("global_exp_avg_sq_l2_norm"),
            "final.global_exp_avg_sq_l2_norm",
        ),
        "cumulative_parameter_global_sum_delta": _correctness_number(
            final.get("cumulative_parameter_global_sum_delta"),
            "final.cumulative_parameter_global_sum_delta",
        ),
        "cumulative_parameter_global_l2_sq_delta": _correctness_number(
            final.get("cumulative_parameter_global_l2_sq_delta"),
            "final.cumulative_parameter_global_l2_sq_delta",
        ),
        "cumulative_parameter_signature_changed": final.get(
            "cumulative_parameter_signature_changed"
        ),
        "parameter_at_capture_after": final_capture,
        "parameter_final": final_parameter,
    }
    expected_final_step = warmup_steps + int(row["timed_steps"])
    if (
        final.get("capture_point") != "after_timed_window_and_peak_reads"
        or final.get("method")
        != "trusted_AdamW_final_state_and_compact_parameter_signature"
        or normalized_final["expected_step"] != expected_final_step
        or normalized_final["minimum_step"] != float(expected_final_step)
        or normalized_final["maximum_step"] != float(expected_final_step)
        or normalized_final["global_active_parameter_tensors"]
        != normalized_update["global_active_parameter_tensors"]
        or normalized_final["global_state_entries"]
        != normalized_final["global_active_parameter_tensors"]
        or normalized_final["global_entries_at_expected_step"]
        != normalized_final["global_active_parameter_tensors"]
        or normalized_final["global_exp_avg_l2_norm"] <= 0.0
        or normalized_final["global_exp_avg_sq_l2_norm"] <= 0.0
        or final_capture != after
        or final_parameter["global_numel"] != EXPECTED_UNIQUE_PARAMETERS
        or normalized_final["cumulative_parameter_signature_changed"] is not True
    ):
        raise RuntimeError("worker final optimizer-update witness is invalid")
    expected_cumulative_sum_delta = (
        final_parameter["global_sum"] - final_capture["global_sum"]
    )
    expected_cumulative_l2_delta = (
        final_parameter["global_l2_sq"] - final_capture["global_l2_sq"]
    )
    if (
        normalized_final["cumulative_parameter_global_sum_delta"]
        != expected_cumulative_sum_delta
        or normalized_final["cumulative_parameter_global_l2_sq_delta"]
        != expected_cumulative_l2_delta
        or (
            expected_cumulative_sum_delta == 0.0
            and expected_cumulative_l2_delta == 0.0
        )
    ):
        raise RuntimeError("worker final parameter update witness is inconsistent or empty")
    return {
        "aux": normalized_aux,
        "gradients": normalized_gradients,
        "optimizer_update": normalized_update,
        "final_optimizer_update": normalized_final,
    }


def _within_tolerance(
    reference: float, candidate: float, tolerance: Mapping[str, float]
) -> bool:
    return abs(candidate - reference) <= float(tolerance["atol"]) + float(
        tolerance["rtol"]
    ) * abs(reference)


def _compare_correctness(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    for name in (
        "global_trainable_parameter_tensors",
        "global_trainable_parameter_elements",
        "global_gradient_tensors",
        "global_gradient_elements",
    ):
        if reference["gradients"][name] != candidate["gradients"][name]:
            raise RuntimeError("candidate and reference gradient topology differs")
    for name in (
        "global_active_parameter_tensors",
        "global_state_entries",
        "global_entries_at_expected_step",
        "expected_step",
    ):
        if reference["optimizer_update"][name] != candidate["optimizer_update"][name]:
            raise RuntimeError("candidate and reference optimizer topology differs")
    for phase in ("parameter_before", "parameter_after"):
        if reference["optimizer_update"][phase]["global_numel"] != candidate[
            "optimizer_update"
        ][phase]["global_numel"]:
            raise RuntimeError("candidate and reference parameter topology differs")
    for phase in ("parameter_at_capture_after", "parameter_final"):
        if reference["final_optimizer_update"][phase]["global_numel"] != candidate[
            "final_optimizer_update"
        ][phase]["global_numel"]:
            raise RuntimeError("candidate and reference final parameter topology differs")
    for name in (
        "global_active_parameter_tensors",
        "global_state_entries",
        "global_entries_at_expected_step",
        "expected_step",
    ):
        if reference["final_optimizer_update"][name] != candidate[
            "final_optimizer_update"
        ][name]:
            raise RuntimeError("candidate and reference final optimizer topology differs")

    tolerances = row["correctness_tolerance"]
    checks: list[tuple[str, float, float]] = []
    for expected, actual in zip(reference["aux"], candidate["aux"], strict=True):
        if any(
            expected[name] != actual[name]
            for name in ("name", "shape_per_rank", "dtype_per_rank", "global_numel")
        ):
            raise RuntimeError("candidate and reference auxiliary structure differs")
        checks.extend(
            ("aux", expected[metric], actual[metric])
            for metric in ("global_sum", "global_l2_sq", "global_min", "global_max")
        )
    checks.append(
        (
            "gradient",
            reference["gradients"]["global_l2_norm"],
            candidate["gradients"]["global_l2_norm"],
        )
    )
    for name in (
        "global_exp_avg_l2_norm",
        "parameter_global_sum_delta",
    ):
        checks.append(
            (
                "optimizer",
                reference["optimizer_update"][name],
                candidate["optimizer_update"][name],
            )
        )
    for name in (
        "global_exp_avg_l2_norm",
        "global_exp_avg_sq_l2_norm",
        "cumulative_parameter_global_sum_delta",
        "cumulative_parameter_global_l2_sq_delta",
    ):
        checks.append(
            (
                "optimizer",
                reference["final_optimizer_update"][name],
                candidate["final_optimizer_update"][name],
            )
        )
    passed = all(
        _within_tolerance(expected, actual, tolerances[group])
        for group, expected, actual in checks
    )
    return {"ok": passed, "numeric_checks": len(checks)}


def _validate_telemetry(
    payload: Mapping[str, Any],
    *,
    expected_kind: str,
    row: Mapping[str, Any],
    candidate_version: int,
) -> dict[str, Any]:
    if payload.get("schema_version") != 1 or payload.get("status") != "PASS":
        raise RuntimeError("worker telemetry did not report the supported PASS schema")
    if payload.get("formal") is not False or payload.get("classification") != (
        "PUBLIC_FULL_MODEL_SMOKE_DIAGNOSTIC_ONLY"
    ):
        raise RuntimeError("worker telemetry classification is invalid")
    if payload.get("objective_kind") != expected_kind or payload.get("world_size") != 2:
        raise RuntimeError("worker role or world size differs from the request")
    selected_version = payload.get("candidate_version")
    if expected_kind == "reference" and selected_version is not None:
        raise RuntimeError("reference worker selected participant code")
    if expected_kind == "candidate" and selected_version != candidate_version:
        raise RuntimeError("candidate worker selected an unexpected version")
    _validate_model_optimization(payload, expected_kind)
    if payload.get("fsdp") != EXPECTED_FSDP:
        raise RuntimeError("worker FSDP configuration differs from the formal contract")

    optimizer = payload.get("optimizer")
    if optimizer != {
        "name": "AdamW",
        "learning_rate": 1e-6,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }:
        raise RuntimeError("worker optimizer configuration differs from the formal contract")
    model = payload.get("model")
    if (
        not isinstance(model, dict)
        or model.get("repo_id") != EXPECTED_MODEL_REPO_ID
        or model.get("revision") != EXPECTED_MODEL_REVISION
        or model.get("tied_lm_head") is not True
        or model.get("unique_parameters") != EXPECTED_UNIQUE_PARAMETERS
        or model.get("parameter_dtype") != "torch.bfloat16"
    ):
        raise RuntimeError("worker model identity differs from the pinned trainable model")

    warmup_steps = int(row["warmup_steps"])
    timed_steps = int(row["timed_steps"])
    if payload.get("warmup_steps") != warmup_steps or payload.get("timed_steps") != timed_steps:
        raise RuntimeError("worker step counts differ from the private row")
    correctness = _validate_correctness_telemetry(payload, row)
    warmup_losses = _finite_number_list(
        payload, "warmup_rank_mean_losses", warmup_steps
    )
    timed_losses = _finite_number_list(payload, "timed_rank_mean_losses", timed_steps)
    durations = _finite_number_list(payload, "rank_max_step_seconds", timed_steps)
    if any(value <= 0.0 for value in durations):
        raise RuntimeError("worker emitted a non-positive measured duration")
    scored = _positive_integer_list(
        payload, "global_scored_tokens_per_step", timed_steps
    )
    non_padding = _positive_integer_list(
        payload, "global_non_padding_tokens_per_step", timed_steps
    )
    if any(a > b for a, b in zip(scored, non_padding, strict=True)):
        raise RuntimeError("worker scored tokens exceed non-padding tokens")
    if payload.get("total_global_scored_tokens") != sum(scored) or payload.get(
        "total_global_non_padding_tokens"
    ) != sum(non_padding):
        raise RuntimeError("worker token totals differ from per-step telemetry")
    total_seconds = _positive_number(payload, "total_rank_max_seconds")
    if not math.isclose(total_seconds, sum(durations), rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError("worker duration total differs from per-step telemetry")
    throughput = _positive_number(payload, "throughput_scored_tokens_per_second")
    if not math.isclose(
        throughput,
        sum(scored) / total_seconds,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError("worker throughput differs from task-owned tokens and time")

    batch = payload.get("batch")
    expected_records = (warmup_steps + timed_steps) * WORLD_SIZE * int(
        row["pair_batch_per_rank"]
    )
    if (
        not isinstance(batch, dict)
        or batch.get("source") != "frozen_tldr"
        or batch.get("partition") != row["dataset_partition"]
        or batch.get("manifest_sha256") != row["dataset_manifest_sha256"]
        or batch.get("tensors_sha256") != row["tensors_sha256"]
        or batch.get("chosen_first") is not True
        or batch.get("nll_target_contract") != row["nll_target_contract"]
        or batch.get("pair_batch_per_rank") != row["pair_batch_per_rank"]
        or batch.get("sequence_length") != row["sequence_length"]
        or batch.get("selected_unique_record_count") != expected_records
        or batch.get("task_owned_token_counting", {}).get("primary")
        != "labels[:, 1:] != -100"
    ):
        raise RuntimeError("worker batch identity differs from the private row")
    fingerprint = batch.get("fingerprint_all_ranks")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise RuntimeError("worker batch fingerprint is invalid")

    allocated = payload.get("peak_allocated_bytes_max_rank")
    reserved = payload.get("peak_reserved_bytes_max_rank")
    if (
        not _is_int(allocated)
        or not _is_int(reserved)
        or allocated <= 0
        or reserved < allocated
    ):
        raise RuntimeError("worker peak memory telemetry is invalid")
    rank_memory = payload.get("per_rank_memory")
    if not isinstance(rank_memory, list) or sorted(
        item.get("rank") for item in rank_memory if isinstance(item, dict)
    ) != [0, 1]:
        raise RuntimeError("worker per-rank memory telemetry is invalid")
    if max(item.get("peak_reserved_bytes", 0) for item in rank_memory) != reserved or max(
        item.get("peak_allocated_bytes", 0) for item in rank_memory
    ) != allocated:
        raise RuntimeError("worker rank memory does not match the reported maxima")

    identities = payload.get("rank_identities")
    completion = payload.get("rank_completion")
    if not isinstance(identities, list) or sorted(
        item.get("rank") for item in identities if isinstance(item, dict)
    ) != [0, 1]:
        raise RuntimeError("worker rank identities are incomplete")
    pids = {
        item.get("pid") for item in identities if isinstance(item, dict) and _is_int(item.get("pid"))
    }
    if len(pids) != 2 or any(pid <= 0 for pid in pids):
        raise RuntimeError("worker rank processes are invalid")
    device_total_memory = _validate_hardware_record(identities)
    if not isinstance(completion, list) or sorted(
        item.get("rank") for item in completion if isinstance(item, dict)
    ) != [0, 1] or any(item.get("status") != "PASS" for item in completion):
        raise RuntimeError("not every worker rank completed")
    run_id = payload.get("run_id")
    try:
        parsed_run_id = str(uuid.UUID(str(run_id)))
    except (ValueError, AttributeError) as exc:
        raise RuntimeError("worker run identifier is invalid") from exc
    return {
        "run_id": parsed_run_id,
        "pids": pids,
        "model": model,
        "fsdp": payload["fsdp"],
        "batch": batch,
        "scored_tokens": scored,
        "non_padding_tokens": non_padding,
        "warmup_losses": warmup_losses,
        "timed_losses": timed_losses,
        "throughput": throughput,
        "peak_reserved": reserved,
        "device_total_memory": device_total_memory,
        "correctness": correctness,
    }


def _validate_hardware_record(identities: Sequence[Mapping[str, Any]]) -> int:
    """Enforce the capability envelope on the trusted per-rank identity record.

    The runtime already refuses to start a rank whose device is below the
    envelope; re-checking here means a forged or replayed telemetry file cannot
    smuggle a below-floor device past the controller.  The device name and UUID
    in the record are provenance and are deliberately not inspected.  Returns
    the smallest per-rank ``total_memory`` so the caller can derive a
    device-relative memory ceiling.
    """

    totals: list[int] = []
    for item in identities:
        if not isinstance(item, dict):
            raise RuntimeError("worker rank identity is not a record")
        raw = str(item.get("compute_capability", ""))
        parts = raw.split(".")
        if len(parts) != 2 or not all(part.isascii() and part.isdigit() for part in parts):
            raise RuntimeError("worker rank identity has no compute capability")
        if (int(parts[0]), int(parts[1])) < MIN_COMPUTE_CAPABILITY:
            raise RuntimeError("worker rank device is below the capability envelope")
        total = item.get("total_memory_bytes")
        if not _is_int(total) or total < MIN_DEVICE_MEMORY_BYTES:
            raise RuntimeError("worker rank device is below the capability envelope")
        totals.append(int(total))
    return min(totals)


def _compare_pair(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    for name in ("model", "fsdp", "batch", "scored_tokens", "non_padding_tokens"):
        if reference[name] != candidate[name]:
            raise RuntimeError(f"candidate and reference {name} telemetry differs")
    if reference["run_id"] == candidate["run_id"] or reference["pids"] & candidate["pids"]:
        raise RuntimeError("candidate and reference did not use fresh rank processes")

    loss_ok = True
    loss_checks = 0
    for phase in ("warmup_losses", "timed_losses"):
        expected_values = reference[phase]
        actual_values = candidate[phase]
        if len(expected_values) != len(actual_values):
            raise RuntimeError("candidate and reference loss telemetry lengths differ")
        for expected, actual in zip(expected_values, actual_values, strict=True):
            allowed = float(row["loss_atol"]) + float(row["loss_rtol"]) * abs(expected)
            loss_ok = loss_ok and abs(actual - expected) <= allowed
            loss_checks += 1

    memory_ratio = candidate["peak_reserved"] / reference["peak_reserved"]
    # Both halves of the memory gate are measured on the device that ran this
    # pair: the ratio against the official-TRL reference cohort launched in the
    # same row, and a ceiling derived from the card's own capacity.  Neither is
    # a byte count frozen from one GPU model.
    memory_ceiling = (
        min(reference["device_total_memory"], candidate["device_total_memory"])
        - DEVICE_MEMORY_HEADROOM_BYTES
    )
    memory_ok = bool(
        memory_ratio <= row["memory_ratio_cap"]
        and candidate["peak_reserved"] <= memory_ceiling
    )
    correctness = _compare_correctness(
        reference["correctness"], candidate["correctness"], row
    )
    return {
        "ratio": candidate["throughput"] / reference["throughput"],
        "loss_ok": loss_ok,
        "loss_checks": loss_checks,
        "memory_ok": memory_ok,
        "memory_ratio": memory_ratio,
        "candidate_peak_reserved": candidate["peak_reserved"],
        "device_peak_reserved_ceiling": memory_ceiling,
        "correctness_evidence_ok": correctness["ok"],
        "correctness_numeric_checks": correctness["numeric_checks"],
    }


def _dry_run_report(plan: Mapping[str, Any]) -> dict[str, Any]:
    status = (
        "PENDING_QUALIFICATION"
        if plan.get("workload_plan_status") == PENDING_PLAN_STATUS
        else "DRY_RUN"
    )
    return {
        "schema_version": 1,
        "status": status,
        "schema_ok": True,
        "gpu_launched": False,
        "execution_ok": False,
        "loss_gate_ok": False,
        "memory_gate_ok": False,
        "correctness_evidence_ok": False,
        "formal_ready": False,
        "formal_score_enabled": False,
        "score": 0.0,
        "score_formula": "100 * equal_weight_geometric_mean(five_row_median_ratios)",
        "reference_role": OFFICIAL_REFERENCE_ROLE,
        "pending_gates": plan["pending_gates"],
        "rows": [
            {
                "row_ordinal": index,
                "pair_batch_per_rank": row["pair_batch_per_rank"],
                "sequence_length": row["sequence_length"],
                "paired_repeats": len(row["paired_role_order"]),
                "paired_role_order": list(row["paired_role_order"]),
            }
            for index, row in enumerate(plan["rows"])
        ],
    }


def run_e2e_controller(
    manifest: Any,
    *,
    lab_root: Path | None = None,
    model_path: Path | None = None,
    data_root: Path | None = None,
    log_dir: Path | None = None,
    rank_bootstrap: Path = DEFAULT_RANK_BOOTSTRAP,
    candidate_version: int | None = None,
    candidate_environment: Mapping[str, str] | None = None,
    reference_environment: Mapping[str, str] | None = None,
    candidate_launch_options: Mapping[str, Any] | None = None,
    timeout_seconds: float = 600.0,
    process_runner: ProcessRunner = _default_process_runner,
    scratch_parent: Path | None = None,
) -> dict[str, Any]:
    """Inspect the hidden plan, then optionally execute all fresh cohorts."""

    try:
        plan = inspect_e2e_plan(manifest)
    except PlanError:
        return {
            "schema_version": 1,
            "status": "INVALID_PLAN",
            "schema_ok": False,
            "gpu_launched": False,
            "execution_ok": False,
            "loss_gate_ok": False,
            "memory_gate_ok": False,
            "correctness_evidence_ok": False,
            "formal_ready": False,
            "formal_score_enabled": False,
            "score": 0.0,
            "pending_gates": ["e2e_schema"],
        }
    if not plan["formal_requested"]:
        return _dry_run_report(plan)
    if not plan["activation_ready"]:
        report = _dry_run_report(plan)
        report["status"] = "BLOCKED_CONFIGURATION"
        return report

    required_paths = (lab_root, model_path, data_root, log_dir)
    if any(path is None for path in required_paths):
        raise ValueError("enabled formal execution requires all trusted paths")
    if not _is_int(candidate_version) or candidate_version <= 0:
        raise ValueError("enabled formal execution requires a positive candidate version")
    if candidate_environment is None or reference_environment is None:
        raise ValueError("enabled formal execution requires explicit environments")
    if candidate_launch_options is None:
        raise ValueError("enabled formal execution requires candidate launch options")
    if timeout_seconds <= 0.0:
        raise ValueError("enabled formal execution requires a positive timeout")
    _validate_candidate_launch(candidate_launch_options, candidate_environment)

    resolved_lab = Path(lab_root).resolve(strict=True)
    resolved_model = Path(model_path).resolve(strict=True)
    resolved_data_root = Path(data_root).resolve(strict=True)
    resolved_log = Path(log_dir)
    resolved_bootstrap = Path(rank_bootstrap).absolute()
    if not resolved_lab.is_dir() or not resolved_model.is_dir() or not resolved_data_root.is_dir():
        raise ValueError("enabled formal execution paths must be directories")
    bootstrap_info = resolved_bootstrap.lstat()
    bootstrap_dir_info = resolved_bootstrap.parent.lstat()
    if (
        stat.S_ISLNK(bootstrap_info.st_mode)
        or not stat.S_ISREG(bootstrap_info.st_mode)
        or bootstrap_info.st_mode & 0o022
        or stat.S_ISLNK(bootstrap_dir_info.st_mode)
        or not stat.S_ISDIR(bootstrap_dir_info.st_mode)
        or bootstrap_dir_info.st_mode & 0o022
    ):
        raise PermissionError("rank bootstrap is not a protected regular file")
    if os.geteuid() == 0 and (
        bootstrap_info.st_uid != 0 or bootstrap_dir_info.st_uid != 0
    ):
        raise PermissionError("rank bootstrap must be owned by root")
    _ensure_private_log_dir(resolved_log)
    if scratch_parent is not None:
        scratch_parent = Path(scratch_parent).resolve(strict=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAIL",
        "schema_ok": True,
        "gpu_launched": False,
        "execution_ok": False,
        "loss_gate_ok": False,
        "memory_gate_ok": False,
        "correctness_evidence_ok": False,
        "formal_ready": True,
        "formal_score_enabled": True,
        "score": 0.0,
        "score_formula": "100 * equal_weight_geometric_mean(five_row_median_ratios)",
        "reference_role": OFFICIAL_REFERENCE_ROLE,
        "pending_gates": [],
        "rows": [],
    }
    failure_stage = "controller"
    failure_role: str | None = None
    try:
        seen_run_ids: set[str] = set()
        seen_pids: set[int] = set()
        cohort_ordinal = 0
        all_loss_ok = True
        all_memory_ok = True
        all_correctness_ok = True
        row_medians: list[float] = []
        report["gpu_launched"] = True
        for row_index, row in enumerate(plan["rows"]):
            data_path = resolved_data_root / row["dataset_subdir"]
            if not data_path.is_dir():
                raise RuntimeError("private dataset partition is unavailable")
            ratios: list[float] = []
            memory_ratios: list[float] = []
            memory_ceilings: list[int] = []
            row_loss_ok = True
            row_memory_ok = True
            row_correctness_ok = True
            for order in row["paired_role_order"]:
                runs: dict[str, dict[str, Any]] = {}
                for role_code in order:
                    kind = "reference" if role_code == "R" else "candidate"
                    failure_stage = f"{kind}_worker"
                    failure_role = None
                    payload = _launch_cohort(
                        kind=kind,
                        row=row,
                        cohort_ordinal=cohort_ordinal,
                        lab_root=resolved_lab,
                        model_path=resolved_model,
                        data_path=data_path,
                        log_dir=resolved_log,
                        rank_bootstrap=resolved_bootstrap,
                        candidate_version=candidate_version,
                        candidate_environment=candidate_environment,
                        reference_environment=reference_environment,
                        candidate_launch_options=candidate_launch_options,
                        timeout_seconds=timeout_seconds,
                        process_runner=process_runner,
                        scratch_parent=scratch_parent,
                    )
                    cohort_ordinal += 1
                    failure_stage = "telemetry_validation"
                    failure_role = kind
                    validated = _validate_telemetry(
                        payload,
                        expected_kind=kind,
                        row=row,
                        candidate_version=candidate_version,
                    )
                    if validated["run_id"] in seen_run_ids or validated["pids"] & seen_pids:
                        raise RuntimeError("a formal cohort reused a prior process identity")
                    seen_run_ids.add(validated["run_id"])
                    seen_pids.update(validated["pids"])
                    runs[kind] = validated
                failure_stage = "pair_comparison"
                failure_role = None
                comparison = _compare_pair(runs["reference"], runs["candidate"], row)
                ratios.append(comparison["ratio"])
                memory_ratios.append(comparison["memory_ratio"])
                memory_ceilings.append(comparison["device_peak_reserved_ceiling"])
                row_loss_ok = row_loss_ok and comparison["loss_ok"]
                row_memory_ok = row_memory_ok and comparison["memory_ok"]
                row_correctness_ok = (
                    row_correctness_ok and comparison["correctness_evidence_ok"]
                )
            row_median = statistics.median(ratios)
            row_medians.append(row_median)
            all_loss_ok = all_loss_ok and row_loss_ok
            all_memory_ok = all_memory_ok and row_memory_ok
            all_correctness_ok = all_correctness_ok and row_correctness_ok
            report["rows"].append(
                {
                    "row_ordinal": row_index,
                    "pair_batch_per_rank": row["pair_batch_per_rank"],
                    "sequence_length": row["sequence_length"],
                    "paired_role_order": list(row["paired_role_order"]),
                    "candidate_over_official_hf_trl_reference_ratios": ratios,
                    "median_ratio": row_median,
                    "candidate_over_official_hf_trl_reference_peak_reserved_ratios": memory_ratios,
                    "device_peak_reserved_ceiling_bytes": min(memory_ceilings),
                    "loss_gate_ok": row_loss_ok,
                    "memory_gate_ok": row_memory_ok,
                    "correctness_evidence_ok": row_correctness_ok,
                }
            )
        report["execution_ok"] = True
        report["loss_gate_ok"] = all_loss_ok
        report["memory_gate_ok"] = all_memory_ok
        report["correctness_evidence_ok"] = all_correctness_ok
        if all_loss_ok and all_memory_ok and all_correctness_ok:
            report["score"] = 100.0 * math.prod(row_medians) ** (1.0 / len(row_medians))
            report["status"] = "PASS"
        else:
            report["status"] = "GATE_FAILED"
            report["failed_gate_groups"] = [
                name
                for name, passed in (
                    ("loss", all_loss_ok),
                    ("memory", all_memory_ok),
                    ("correctness_update", all_correctness_ok),
                )
                if not passed
            ]
    except Exception as exc:
        # Candidate stderr and exception text stay in root-only cohort logs.
        # The structured result communicates only bounded stage/class/role enums.
        report["error"] = "formal e2e execution failed"
        report["failure_stage"] = failure_stage
        if failure_stage == "telemetry_validation" and failure_role in {
            "reference",
            "candidate",
        }:
            report["failure_role"] = failure_role
        if isinstance(exc, WorkerTimeoutError):
            report["failure_class"] = "timeout"
        elif isinstance(exc, WorkerNonzeroExitError):
            report["failure_class"] = "nonzero_exit"
        elif isinstance(exc, WorkerTelemetryError):
            report["failure_class"] = "missing_telemetry"
        elif failure_stage == "telemetry_validation":
            report["failure_class"] = "invalid_telemetry"
        else:
            report["failure_class"] = "infrastructure"
    return report
