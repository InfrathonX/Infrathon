#!/usr/bin/env python3
"""Trusted MLSys Forge adapter for the frozen ten-minute public diagnostic.

The adapter deliberately has two outputs:

* a complete, mode-0600 operator artifact under ``/logs/verifier``; and
* one SForge structured result containing an explicit allowlist of aggregates.

The public runner is trusted and host-integrated.  Candidate files are copied
through a no-follow directory walk before the runner is allowed to see them.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any


_COMMON_PATH = Path(__file__).resolve().with_name("run_judge.py")
_COMMON_SPEC = importlib.util.spec_from_file_location(
    "kvbench_private_judge_common", _COMMON_PATH
)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise RuntimeError("trusted Judge common module is unavailable")
COMMON = importlib.util.module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(COMMON)


PUBLIC_RUNNER_KIND = "gpu67-v2.2-public-candidate-only-vs-frozen-v2.1-direct"
PUBLIC_PROFILE = "public-swe192-distinct"
PUBLIC_SUITE = "public"
PUBLIC_WORKFLOWS = 192
PUBLIC_CONCURRENCY = 192
PUBLIC_MAX_STEPS = 100
PUBLIC_WARMUP_SEC = 300
PUBLIC_MEASUREMENT_SEC = 600
PUBLIC_NONCE_COMMITMENT_SHA256 = (
    "4902baae73726c1a2f372e35e9e104f6bb3db7dc7593822359548fba60e1b00c"
)
# Contract revisions, and the reason both are accepted, are documented on
# ``run_judge.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256``.
EXPECTED_BACKEND_SERVING_CONTRACT_SHA256 = (
    COMMON.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
)
ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256 = (
    COMMON.ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
)
# The public plane has no bounded-wait hard gate; it reports the ceiling the
# hidden gate would apply to a 600-second scored window so the same rule is
# visible at both scales.
PUBLIC_P99_GATEWAY_WAIT_CEILING_SEC = (
    COMMON.P99_GATEWAY_WAIT_WINDOW_FRACTION * PUBLIC_MEASUREMENT_SEC
)
PUBLIC_BASELINE_IDENTITY_FIELDS = frozenset(
    {
        "file_sha256",
        "valid_steps_per_min",
        "backend_serving_contract_sha256",
        "workload_nonce_sha256",
    }
)
RUNNER_BASELINE_EVIDENCE_FIELDS = frozenset(
    {
        "path",
        "file_sha256",
        "workload_nonce_sha256",
        "direct_valid_steps_per_min",
        "backend_serving_contract_sha256",
    }
)
PUBLIC_BASELINE_PATH = Path(
    os.environ.get("KVBENCH_PUBLIC_BASELINE_PATH", "/run/kvbench-baselines/public.json")
)
INSTALLED_PUBLIC_RUNNER = Path("/opt/kvbench-mlsysforge/run_public_candidate.py")

# The task-local public endpoint has a 2100-second outer deadline. Keep a
# full 100-second supervision margin so cooperative stack cleanup finishes
# before the endpoint can terminate this adapter.
PUBLIC_OUTER_TIMEOUT_SEC = 2100
PUBLIC_OUTER_MARGIN_SEC = 100
DEFAULT_PUBLIC_RUNNER_TIMEOUT_SEC = 1500
DEFAULT_PUBLIC_CLEANUP_GRACE_SEC = 450
MAX_ENTRIES = 4096
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_DEPTH = 64
MAX_RELATIVE_PATH_BYTES = 4096

PUBLIC_REASON_ALLOWLIST = frozenset(COMMON.RUNNER_REASON_ALLOWLIST) | {
    "runner_validation_failed",
}
PRESSURE_STATUS_ALLOWLIST = frozenset(
    {"pressurized", "diagnostic_only", "not_pressurized", "invalid", "unavailable"}
)


class PublicSubmissionError(ValueError):
    """Fixed-code failure at the untrusted source boundary."""


def _exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _same_finite(value: Any, expected: float) -> bool:
    observed = COMMON._finite(value)
    return observed is not None and math.isclose(
        observed, expected, rel_tol=0.0, abs_tol=1e-12
    )


def _require_profile_shape(value: Any) -> None:
    if not isinstance(value, dict) or any(
        (
            value.get("name") != PUBLIC_PROFILE,
            not _exact_int(value.get("workflows"), PUBLIC_WORKFLOWS),
            not _exact_int(value.get("concurrency"), PUBLIC_CONCURRENCY),
            not _exact_int(value.get("max_steps"), PUBLIC_MAX_STEPS),
            not _exact_int(
                value.get("measurement_warmup_sec"), PUBLIC_WARMUP_SEC
            ),
            not _exact_int(
                value.get("measurement_window_sec"), PUBLIC_MEASUREMENT_SEC
            ),
        )
    ):
        raise RuntimeError("trusted public profile shape changed")


def _valid_public_baseline_identity(identity: Any) -> bool:
    if not isinstance(identity, dict) or set(identity) != PUBLIC_BASELINE_IDENTITY_FIELDS:
        return False
    file_sha = identity.get("file_sha256")
    rate = COMMON._finite(identity.get("valid_steps_per_min"))
    return bool(
        isinstance(file_sha, str)
        and len(file_sha) == 64
        and all(character in "0123456789abcdef" for character in file_sha)
        and rate is not None
        and rate > 0.0
        and identity.get("backend_serving_contract_sha256")
        in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
        and identity.get("workload_nonce_sha256")
        == PUBLIC_NONCE_COMMITMENT_SHA256
    )


def _public_baseline_identity_from_runner_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RUNNER_BASELINE_EVIDENCE_FIELDS:
        raise RuntimeError("trusted public baseline evidence contract failed")
    identity = {
        "file_sha256": value.get("file_sha256"),
        "valid_steps_per_min": COMMON._finite(
            value.get("direct_valid_steps_per_min")
        ),
        "backend_serving_contract_sha256": value.get(
            "backend_serving_contract_sha256"
        ),
        "workload_nonce_sha256": value.get("workload_nonce_sha256"),
    }
    if (
        not isinstance(value.get("path"), str)
        or not value["path"]
        or not _valid_public_baseline_identity(identity)
    ):
        raise RuntimeError("trusted public baseline evidence contract failed")
    return identity


def _load_public_baseline_bundle() -> dict[str, Any]:
    if (
        not PUBLIC_BASELINE_PATH.is_file()
        or PUBLIC_BASELINE_PATH.is_symlink()
        or PUBLIC_BASELINE_PATH.stat().st_size > COMMON.MAX_TRUSTED_JSON_BYTES
    ):
        raise RuntimeError("frozen public Direct evidence is unavailable or changed")
    raw = PUBLIC_BASELINE_PATH.read_bytes()
    if not raw or len(raw) > COMMON.MAX_TRUSTED_JSON_BYTES:
        raise RuntimeError("frozen public Direct evidence is unavailable or changed")
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    report = COMMON._strict_json_object_bytes(
        raw, description="frozen public Direct evidence"
    )
    legs = report.get("legs")
    baseline = legs.get("baseline") if isinstance(legs, dict) else None
    if (
        report.get("schema_version") != 1
        or report.get("status") != "complete"
        or report.get("suite") != PUBLIC_SUITE
        or report.get("profile") != PUBLIC_PROFILE
        or report.get("workload_nonce_sha256")
        != PUBLIC_NONCE_COMMITMENT_SHA256
        or not isinstance(baseline, dict)
        or baseline.get("profile") != PUBLIC_PROFILE
    ):
        raise RuntimeError("frozen public Direct evidence contract changed")
    _require_profile_shape(baseline.get("profile_config"))
    direct_rate = COMMON._finite(baseline.get("valid_steps_per_min"))
    if direct_rate is None or direct_rate <= 0.0:
        raise RuntimeError("frozen public Direct serving contract changed")
    normalized_provenance = COMMON._normalize_baseline_provenance(
        baseline.get("model_provenance")
    )
    normalized = copy.deepcopy(baseline)
    normalized["model_provenance"] = normalized_provenance
    public_baseline_identity = {
        "file_sha256": observed_sha256,
        "valid_steps_per_min": direct_rate,
        "backend_serving_contract_sha256": normalized_provenance[
            "backend_serving_contract_sha256"
        ],
        "workload_nonce_sha256": report["workload_nonce_sha256"],
    }
    if not _valid_public_baseline_identity(public_baseline_identity):
        raise RuntimeError("frozen public Direct identity changed")
    return {
        "path": str(PUBLIC_BASELINE_PATH),
        "public_baseline_identity": public_baseline_identity,
        "leg": normalized,
    }


def _load_public_baseline() -> dict[str, Any]:
    return _load_public_baseline_bundle()["leg"]


def _safe_file_mode(raw_mode: int) -> int:
    # Never reproduce set-id or sticky bits from an untrusted submission.
    return stat.S_IMODE(raw_mode) & 0o777


def _copy_regular_file(
    source_fd: int,
    destination_fd: int,
    name: str,
    source_stat: os.stat_result,
    budget: dict[str, int],
) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    opened = os.open(name, flags, dir_fd=source_fd)
    try:
        verified = os.fstat(opened)
        if (
            not stat.S_ISREG(verified.st_mode)
            or (verified.st_dev, verified.st_ino)
            != (source_stat.st_dev, source_stat.st_ino)
        ):
            raise PublicSubmissionError("submission_changed_during_copy")
        budget["bytes"] += verified.st_size
        if budget["bytes"] > MAX_SOURCE_BYTES:
            raise PublicSubmissionError("submission_too_large")
        destination = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            _safe_file_mode(verified.st_mode),
            dir_fd=destination_fd,
        )
        copied = 0
        try:
            while True:
                chunk = os.read(opened, min(1024 * 1024, MAX_SOURCE_BYTES + 1))
                if not chunk:
                    break
                copied += len(chunk)
                if copied > verified.st_size or copied > MAX_SOURCE_BYTES:
                    raise PublicSubmissionError("submission_changed_during_copy")
                view = memoryview(chunk)
                while view:
                    written = os.write(destination, view)
                    if written <= 0:
                        raise OSError("short write while pinning candidate")
                    view = view[written:]
            if copied != verified.st_size:
                raise PublicSubmissionError("submission_changed_during_copy")
            os.fchmod(destination, _safe_file_mode(verified.st_mode))
            os.fsync(destination)
        finally:
            os.close(destination)
        after = os.fstat(opened)
        if (
            after.st_size != verified.st_size
            or after.st_mtime_ns != verified.st_mtime_ns
            or after.st_ctime_ns != verified.st_ctime_ns
        ):
            raise PublicSubmissionError("submission_changed_during_copy")
    finally:
        os.close(opened)


def _copy_directory(
    source_fd: int,
    destination_fd: int,
    relative: tuple[str, ...],
    budget: dict[str, int],
) -> None:
    if len(relative) > MAX_DEPTH:
        raise PublicSubmissionError("submission_too_deep")
    try:
        names = sorted(os.listdir(source_fd))
    except OSError as exc:
        raise PublicSubmissionError("submission_unreadable") from exc
    for name in names:
        if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name:
            raise PublicSubmissionError("submission_path_invalid")
        try:
            encoded_path = "/".join((*relative, name)).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise PublicSubmissionError("submission_path_invalid") from exc
        if len(encoded_path) > MAX_RELATIVE_PATH_BYTES:
            raise PublicSubmissionError("submission_path_too_long")
        budget["entries"] += 1
        if budget["entries"] > MAX_ENTRIES:
            raise PublicSubmissionError("submission_too_many_entries")
        observed = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        mode = observed.st_mode
        if stat.S_ISLNK(mode):
            raise PublicSubmissionError("submission_symlink")
        if stat.S_ISREG(mode):
            _copy_regular_file(
                source_fd, destination_fd, name, observed, budget
            )
            continue
        if not stat.S_ISDIR(mode):
            raise PublicSubmissionError("submission_special_file")

        os.mkdir(name, 0o700, dir_fd=destination_fd)
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        child_source = os.open(name, directory_flags, dir_fd=source_fd)
        child_destination = os.open(
            name, directory_flags, dir_fd=destination_fd
        )
        try:
            verified = os.fstat(child_source)
            if (
                not stat.S_ISDIR(verified.st_mode)
                or (verified.st_dev, verified.st_ino)
                != (observed.st_dev, observed.st_ino)
            ):
                raise PublicSubmissionError("submission_changed_during_copy")
            _copy_directory(
                child_source,
                child_destination,
                (*relative, name),
                budget,
            )
            os.fchmod(child_destination, _safe_file_mode(verified.st_mode))
        finally:
            os.close(child_destination)
            os.close(child_source)


def secure_copy_submission(workspace: Path, destination: Path) -> str:
    """Take a bounded snapshot without following any untrusted link."""

    if workspace.is_symlink():
        raise PublicSubmissionError("submission_symlink")
    try:
        resolved_workspace = workspace.resolve(strict=True)
    except OSError as exc:
        raise PublicSubmissionError("submission_not_directory") from exc
    root_mode = resolved_workspace.lstat().st_mode
    if not stat.S_ISDIR(root_mode):
        raise PublicSubmissionError("submission_not_directory")
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("trusted public candidate destination is not empty")
    destination.mkdir(mode=0o700)

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    source_fd = os.open(resolved_workspace, directory_flags)
    destination_fd = os.open(destination, directory_flags)
    try:
        source_root_stat = os.fstat(source_fd)
        if not stat.S_ISDIR(source_root_stat.st_mode):
            raise PublicSubmissionError("submission_not_directory")
        _copy_directory(
            source_fd,
            destination_fd,
            (),
            {"entries": 0, "bytes": 0},
        )
        # The frozen nested main container executes the bind-mounted candidate
        # as its unprivileged service user. Preserve only ordinary rwx bits on
        # the snapshot root just as we do for descendants; leaving the trusted
        # scratch's creation mode 0700 would make launch.sh fail with exit 126.
        os.fchmod(destination_fd, _safe_file_mode(source_root_stat.st_mode))
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        os.close(destination_fd)
        os.close(source_fd)

    launch = destination / "launch.sh"
    if (
        not launch.is_file()
        or launch.is_symlink()
        or not os.access(launch, os.X_OK)
    ):
        shutil.rmtree(destination, ignore_errors=True)
        raise PublicSubmissionError("launch_not_executable")
    return COMMON.tree_sha256(destination)


def _validate_runtime_paths() -> tuple[Path, Path, Path]:
    migration_raw = os.environ.get("KVBENCH_MIGRATION_ROOT", "")
    scratch_raw = os.environ.get("KVBENCH_HOST_EVAL_DIR", "")
    eval_root_raw = os.environ.get("KVBENCH_EVAL_ROOT", "")
    nonce_raw = os.environ.get(
        "KVBENCH_PUBLIC_NONCE_SOURCE",
        os.environ.get("KVBENCH_HIDDEN_NONCE_SOURCE", ""),
    )
    if not all((migration_raw, scratch_raw, eval_root_raw, nonce_raw)):
        raise RuntimeError("trusted public Judge mounts are not configured")
    migration = Path(migration_raw).resolve(strict=True)
    scratch = Path(scratch_raw).resolve(strict=True)
    eval_root = Path(eval_root_raw).resolve(strict=True)
    nonce = Path(nonce_raw).resolve(strict=True)
    if (
        not migration.is_dir()
        or migration.is_symlink()
        or not scratch.is_dir()
        or scratch.is_symlink()
        or not eval_root.is_dir()
        or eval_root.is_symlink()
        or scratch.parent != eval_root
        or not scratch.name.startswith("judge-")
        or not nonce.is_file()
        or nonce.is_symlink()
    ):
        raise RuntimeError("trusted public Judge mount contract failed")
    return migration, scratch, nonce


def _run_supervised(
    command: list[str], *, environment: dict[str, str], runner_log: Path
) -> dict[str, Any]:
    timeout = int(
        os.environ.get(
            "KVBENCH_PUBLIC_RUNNER_TIMEOUT_SEC",
            str(DEFAULT_PUBLIC_RUNNER_TIMEOUT_SEC),
        )
    )
    cleanup_grace = int(
        os.environ.get(
            "KVBENCH_PUBLIC_CLEANUP_GRACE_SEC",
            str(DEFAULT_PUBLIC_CLEANUP_GRACE_SEC),
        )
    )
    if timeout <= PUBLIC_WARMUP_SEC + PUBLIC_MEASUREMENT_SEC or cleanup_grace <= 0:
        raise RuntimeError("trusted public runner timeout budget is unsafe")
    if (
        timeout + cleanup_grace
        > PUBLIC_OUTER_TIMEOUT_SEC - PUBLIC_OUTER_MARGIN_SEC
    ):
        raise RuntimeError("trusted public runner exceeds its outer timeout budget")

    cancel_signal: int | None = None
    termination_sent = False
    timed_out = False
    forced_kill = False
    process: subprocess.Popen[bytes] | None = None
    previous_handlers: dict[int, Any] = {}

    def request_cancel(signum: int, _frame: Any) -> None:
        nonlocal cancel_signal
        cancel_signal = signum

    runner_log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        runner_log,
        os.O_CREAT
        | os.O_WRONLY
        | os.O_TRUNC
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    started = time.monotonic()
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_cancel)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = started + timeout
            cleanup_deadline: float | None = None
            while process.poll() is None:
                now = time.monotonic()
                if not termination_sent and (
                    cancel_signal is not None or now >= deadline
                ):
                    timed_out = cancel_signal is None
                    termination_sent = True
                    cleanup_deadline = now + cleanup_grace
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                if (
                    termination_sent
                    and cleanup_deadline is not None
                    and now >= cleanup_deadline
                ):
                    forced_kill = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    break
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    continue
            if process.poll() is None:
                process.wait(timeout=10)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    runner_log.chmod(0o600)
    return {
        "returncode": process.returncode if process is not None else None,
        "timed_out": timed_out,
        "cancel_signal": (
            signal.Signals(cancel_signal).name if cancel_signal is not None else None
        ),
        "termination_sent": termination_sent,
        "forced_kill": forced_kill,
        "runtime_seconds": time.monotonic() - started,
        "force_reconcile": None,
    }


def _validate_workflows(candidate: dict[str, Any], *, required: bool) -> None:
    workflows = candidate.get("workflows")
    if not isinstance(workflows, list):
        if required:
            raise RuntimeError("trusted public workflow evidence is missing")
        return
    if required and len(workflows) != PUBLIC_WORKFLOWS:
        raise RuntimeError("trusted public workflow count changed")
    task_ids: list[str] = []
    run_ids: list[str] = []
    for workflow in workflows:
        if not isinstance(workflow, dict):
            raise RuntimeError("trusted public workflow evidence is malformed")
        task_id = workflow.get("task_id")
        run_id = workflow.get("run_id")
        steps = workflow.get("total_executed_steps")
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(run_id, str)
            or not run_id
            or not isinstance(steps, int)
            or isinstance(steps, bool)
            or not 0 <= steps <= PUBLIC_MAX_STEPS
        ):
            raise RuntimeError("trusted public workflow identity is malformed")
        task_ids.append(task_id)
        run_ids.append(run_id)
    if required and (
        len(set(task_ids)) != PUBLIC_WORKFLOWS
        or len(set(run_ids)) != PUBLIC_WORKFLOWS
    ):
        raise RuntimeError("trusted public workflows are not 192 distinct assignments")


def _validate_complete_report(
    report: dict[str, Any],
    candidate_sha256: str,
    returncode: Any,
    baseline_bundle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    status = report.get("status")
    expected_returncode = {"complete": 0, "candidate_invalid": 3}.get(status)
    cleanup = report.get("cleanup")
    residue = cleanup.get("residue") if isinstance(cleanup, dict) else None
    baseline_evidence = report.get("baseline_evidence")
    expected_identity = baseline_bundle.get("public_baseline_identity")
    try:
        observed_identity = _public_baseline_identity_from_runner_evidence(
            baseline_evidence
        )
    except RuntimeError:
        observed_identity = None
    contract = report.get("workload_contract")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != PUBLIC_RUNNER_KIND
        or expected_returncode is None
        or returncode != expected_returncode
        or report.get("phase") != "quiescent"
        or report.get("direct_leg_executed") is not False
        or report.get("candidate_tree_sha256") != candidate_sha256
        or report.get("preflight_backend_serving_contract_sha256")
        not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
        or report.get("errors") != []
        or not isinstance(report.get("completed_at_utc"), str)
        or not isinstance(cleanup, dict)
        or cleanup.get("attempted") is not True
        or cleanup.get("stack_stop_succeeded") is not True
        or cleanup.get("quiescent") is not True
        or not isinstance(residue, dict)
        or residue.get("compose_container_ids") != []
        or residue.get("sandbox_container_ids") != []
        or not _valid_public_baseline_identity(expected_identity)
        or observed_identity != expected_identity
        or baseline_evidence.get("path") != baseline_bundle.get("path")
        or not isinstance(contract, dict)
        or contract.get("suite") != PUBLIC_SUITE
        or contract.get("profile") != PUBLIC_PROFILE
        or not _exact_int(contract.get("workflows"), PUBLIC_WORKFLOWS)
        or not _exact_int(contract.get("concurrency"), PUBLIC_CONCURRENCY)
        or not _exact_int(contract.get("max_steps"), PUBLIC_MAX_STEPS)
        or not _exact_int(
            contract.get("measurement_warmup_sec"), PUBLIC_WARMUP_SEC
        )
        or not _exact_int(
            contract.get("measurement_window_sec"), PUBLIC_MEASUREMENT_SEC
        )
        or contract.get("workload_nonce_sha256")
        != PUBLIC_NONCE_COMMITMENT_SHA256
    ):
        raise RuntimeError("trusted public runner terminal evidence contract failed")

    candidate = report.get("leg")
    evaluation = report.get("evaluation")
    if not isinstance(candidate, dict) or not isinstance(evaluation, dict):
        raise RuntimeError("trusted public runner has no complete candidate evidence")
    expected_valid = status == "complete"
    if evaluation.get("valid") is not expected_valid:
        raise RuntimeError("trusted public runner status and validity disagree")
    if candidate.get("profile") != PUBLIC_PROFILE:
        raise RuntimeError("trusted public candidate profile changed")
    _require_profile_shape(candidate.get("profile_config"))
    _validate_workflows(candidate, required=expected_valid)

    reasons = evaluation.get("all_invalid_reasons")
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) for reason in reasons)
        or (expected_valid and reasons != [])
    ):
        raise RuntimeError("trusted public invalid-reason contract failed")
    candidate_rate = COMMON._finite(candidate.get("valid_steps_per_min"))
    evaluation_rate = COMMON._finite(
        evaluation.get("candidate_valid_steps_per_min")
    )
    if (
        not _same_finite(
            evaluation.get("direct_valid_steps_per_min"),
            expected_identity["valid_steps_per_min"],
        )
        or candidate_rate is None
        or candidate_rate < 0.0
        or evaluation_rate is None
        or not math.isclose(
            candidate_rate, evaluation_rate, rel_tol=0.0, abs_tol=1e-12
        )
        or (expected_valid and candidate_rate <= 0.0)
    ):
        raise RuntimeError("trusted public throughput accounting failed")
    if expected_valid:
        for field in ("request_success_rate", "valid_step_ratio"):
            value = COMMON._finite(candidate.get(field))
            if value is None or not 0.0 <= value <= 1.0:
                raise RuntimeError("trusted public quality aggregate is malformed")
    return candidate, evaluation, status


def _counter_delta(candidate: dict[str, Any], metric_name: str) -> float | None:
    metrics = candidate.get("vllm_metrics")
    deltas = metrics.get("counter_deltas") if isinstance(metrics, dict) else None
    if not isinstance(deltas, list):
        return None
    total = 0.0
    found = False
    for item in deltas:
        if not isinstance(item, dict) or item.get("name") != metric_name:
            continue
        value = COMMON._finite(item.get("delta"))
        if value is None or value < 0.0:
            return None
        total += value
        found = True
    return total if found and math.isfinite(total) else None


def _nested_finite(value: Any, *keys: str) -> float | None:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return COMMON._finite(current)


def _safe_reasons(evaluation: dict[str, Any]) -> list[str]:
    raw = evaluation.get("all_invalid_reasons")
    if not isinstance(raw, list):
        return ["runner_validation_failed"]
    normalized = {
        reason if reason in PUBLIC_REASON_ALLOWLIST else "runner_validation_failed"
        for reason in raw
        if isinstance(reason, str)
    }
    return sorted(normalized)


def public_structured_result(
    candidate: dict[str, Any],
    evaluation: dict[str, Any],
    baseline: dict[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    valid = status == "complete" and evaluation.get("valid") is True
    candidate_rate = COMMON._finite(candidate.get("valid_steps_per_min"))
    direct_rate = COMMON._finite(baseline.get("valid_steps_per_min"))
    if (
        candidate_rate is None
        or candidate_rate < 0.0
        or direct_rate is None
        or direct_rate <= 0.0
    ):
        raise RuntimeError("public diagnostic throughput is malformed")
    speedup = candidate_rate / direct_rate
    change = (speedup - 1.0) * 100.0

    request_success = COMMON._finite(candidate.get("request_success_rate"))
    valid_step_ratio = COMMON._finite(candidate.get("valid_step_ratio"))
    audit = candidate.get("backend_audit")
    p95_wait = _nested_finite(audit, "p95_gateway_wait_sec")
    p99_wait = _nested_finite(audit, "p99_gateway_wait_sec")
    candidate_work = COMMON._finite(candidate.get("backend_tokens_per_valid_step"))
    direct_work = COMMON._finite(baseline.get("backend_tokens_per_valid_step"))
    backend_work_ratio = (
        candidate_work / direct_work
        if candidate_work is not None
        and direct_work is not None
        and candidate_work >= 0.0
        and direct_work > 0.0
        else None
    )

    prefix_hits = _counter_delta(candidate, "vllm:prefix_cache_hits_total")
    prefix_queries = _counter_delta(candidate, "vllm:prefix_cache_queries_total")
    prefix_hit_rate = (
        prefix_hits / prefix_queries
        if prefix_hits is not None
        and prefix_queries is not None
        and 0.0 <= prefix_hits <= prefix_queries
        and prefix_queries > 0.0
        else None
    )

    pressure = candidate.get("pressure_validation")
    raw_pressure_status = pressure.get("status") if isinstance(pressure, dict) else None
    pressure_status = (
        raw_pressure_status
        if raw_pressure_status in PRESSURE_STATUS_ALLOWLIST
        else "unavailable"
    )
    pressure_valid = (
        pressure.get("valid")
        if isinstance(pressure, dict) and isinstance(pressure.get("valid"), bool)
        else None
    )
    waiting_valid = None
    waiting_samples = None
    if isinstance(pressure, dict):
        contention = pressure.get("contention_evidence")
        waiting = (
            contention.get("waiting_at_high_kv")
            if isinstance(contention, dict)
            else None
        )
        if isinstance(waiting, dict):
            waiting_valid = (
                waiting.get("valid")
                if isinstance(waiting.get("valid"), bool)
                else None
            )
            samples = waiting.get("sample_count")
            waiting_samples = (
                samples
                if isinstance(samples, int)
                and not isinstance(samples, bool)
                and samples >= 0
                else None
            )
    invalid_reasons = _safe_reasons(evaluation)
    score = candidate_rate if valid else 0.0
    status_label = "PASSED" if valid else "FAILED"
    summary = (
        f"PUBLIC DIAGNOSTIC {'VALID' if valid else 'INVALID'}: "
        f"candidate={candidate_rate:.6f}, frozen Direct={direct_rate:.1f} "
        "valid executable agent steps/min; fixed 192-workflow/192-concurrency/"
        "100-step/300-second-warmup/600-second-scored profile."
    )
    metrics = {
        "source": "kvbench_public_candidate_only_vs_frozen_direct",
        "evaluation_status": "valid" if valid else "candidate_invalid",
        "profile": PUBLIC_PROFILE,
        "workflows": PUBLIC_WORKFLOWS,
        "concurrency": PUBLIC_CONCURRENCY,
        "max_steps": PUBLIC_MAX_STEPS,
        "measurement_warmup_sec": PUBLIC_WARMUP_SEC,
        "measurement_window_sec": PUBLIC_MEASUREMENT_SEC,
        "primary_score_name": "valid_steps_per_min",
        "primary_score_value": score,
        "primary_score_direction": "maximize",
        "correctness_gate_passed": valid,
        "frozen_direct_valid_steps_per_min": direct_rate,
        "candidate_valid_steps_per_min": candidate_rate,
        "speedup_over_frozen_direct": speedup,
        "change_percent": change,
        "request_success_rate": request_success,
        "valid_step_ratio": valid_step_ratio,
        "gateway_wait_p95_sec": p95_wait,
        "gateway_wait_p99_sec": p99_wait,
        # Advisory only on the public plane; the hidden gate applies the same
        # fraction to its own scored window.
        "gateway_wait_p99_ceiling_sec": PUBLIC_P99_GATEWAY_WAIT_CEILING_SEC,
        "prefix_cache_hit_rate": prefix_hit_rate,
        "backend_work_ratio_to_frozen_direct": backend_work_ratio,
        "pressure_status": pressure_status,
        "pressure_valid": pressure_valid,
        "kv_cache_usage_p95": _nested_finite(
            pressure,
            "occupancy_evidence",
            "observed_kv_occupancy",
            "kv_p95",
        ),
        "high_kv_time_fraction": _nested_finite(
            pressure,
            "occupancy_evidence",
            "observed_kv_occupancy",
            "time_fraction",
        ),
        "waiting_at_high_kv": waiting_valid,
        "waiting_at_high_kv_sample_count": waiting_samples,
        "preemptions_delta": _nested_finite(
            pressure, "contention_evidence", "preemption", "preemptions_delta"
        ),
        "invalid_reasons": invalid_reasons,
        "qualification_passed": evaluation.get("qualification_passed") is True,
        "strictly_better_than_frozen_direct": candidate_rate > direct_rate,
        "candidate_only": True,
        "fresh_paired_ab": False,
        "auto_feedback_safe": True,
    }
    result = {
        "valid": valid,
        "score": score,
        "pass_rate": 1.0 if valid else 0.0,
        "total_tests": 1,
        "passed": 1 if valid else 0,
        "failed": 0 if valid else 1,
        "errors": 0,
        "summary": summary,
        "details": [
            {
                "name": "public_diagnostic:fixed_profile",
                "status": status_label,
                "message": (
                    "trusted fixed public workload completed"
                    if valid
                    else "trusted fixed public workload reported a candidate-invalid result"
                ),
                "score": 1.0 if valid else 0.0,
                "weight": 1.0,
            }
        ],
        "metrics": metrics,
    }
    validate_public_output(result, "valid" if valid else "candidate_invalid")
    return result


def candidate_failure(code: str) -> dict[str, Any]:
    allowed = {
        "submission_not_directory",
        "submission_symlink",
        "submission_special_file",
        "submission_too_large",
        "submission_too_many_entries",
        "submission_too_deep",
        "submission_path_invalid",
        "submission_path_too_long",
        "submission_unreadable",
        "submission_changed_during_copy",
        "launch_not_executable",
    }
    normalized = code if code in allowed else "submission_contract_invalid"
    result = {
        "valid": False,
        "score": 0.0,
        "pass_rate": 0.0,
        "total_tests": 1,
        "passed": 0,
        "failed": 1,
        "errors": 0,
        "summary": f"PUBLIC DIAGNOSTIC INVALID: packaging gate failed ({normalized}).",
        "details": [
            {
                "name": "public_diagnostic:submission_contract",
                "status": "FAILED",
                "message": normalized,
                "score": 0.0,
                "weight": 1.0,
            }
        ],
        "metrics": {
            "source": "kvbench_public_candidate_only_vs_frozen_direct",
            "evaluation_status": "candidate_invalid",
            "profile": PUBLIC_PROFILE,
            "primary_score_name": "valid_steps_per_min",
            "primary_score_value": 0.0,
            "primary_score_direction": "maximize",
            "correctness_gate_passed": False,
            "invalid_reasons": [normalized],
            "auto_feedback_safe": True,
        },
    }
    validate_public_output(result, "candidate_invalid")
    return result


def infrastructure_failure(classification: str = "infrastructure_error") -> dict[str, Any]:
    if classification not in {"infrastructure_error", "resource_busy", "cancelled"}:
        classification = "infrastructure_error"
    result = {
        "valid": False,
        "score": 0.0,
        "pass_rate": 0.0,
        "total_tests": 1,
        "passed": 0,
        "failed": 0,
        "errors": 1,
        "summary": (
            "PUBLIC DIAGNOSTIC INFRASTRUCTURE ERROR: no authoritative candidate "
            "result is available; operator-only evidence contains the diagnostic."
        ),
        "details": [
            {
                "name": "public_diagnostic:infrastructure",
                "status": "ERROR",
                "message": "trusted public evaluation did not complete",
                "score": 0.0,
                "weight": 1.0,
            }
        ],
        "metrics": {
            "source": "kvbench_public_candidate_only_vs_frozen_direct",
            "evaluation_status": classification,
            "profile": PUBLIC_PROFILE,
            "primary_score_name": "valid_steps_per_min",
            "primary_score_value": 0.0,
            "primary_score_direction": "maximize",
            "correctness_gate_passed": False,
            "auto_feedback_safe": False,
        },
    }
    validate_public_output(result, classification)
    return result


def validate_public_output(result: dict[str, Any], classification: str) -> None:
    COMMON.validate_public_result(result, classification)
    encoded = json.dumps(
        result, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    forbidden_keys = (
        '"task_id"',
        '"run_id"',
        '"prompt"',
        '"trace"',
        '"path"',
        '"workload_nonce"',
        '"candidate_dir"',
    )
    if any(key in encoded for key in forbidden_keys):
        raise RuntimeError("public diagnostic contains a forbidden private field")


def _assert_public_baseline_snapshot_current(
    baseline_bundle: dict[str, Any],
) -> None:
    identity = baseline_bundle.get("public_baseline_identity")
    raw_path = baseline_bundle.get("path")
    if not _valid_public_baseline_identity(identity) or not isinstance(raw_path, str):
        raise RuntimeError("frozen public Direct snapshot identity is unavailable")
    path = Path(raw_path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("frozen public Direct snapshot path drifted")
    if COMMON.file_sha256(path) != identity["file_sha256"]:
        raise RuntimeError("frozen public Direct snapshot bytes drifted")


def _write_private_envelope(
    path: Path,
    *,
    started_at_utc: str,
    classification: str,
    candidate_sha256: str | None,
    runner_metadata: dict[str, Any],
    runner_evidence: dict[str, Any] | None,
    public_result: dict[str, Any],
    private_errors: list[str],
    baseline_bundle: dict[str, Any] | None = None,
) -> None:
    public_baseline_identity: dict[str, Any] | None = None
    if baseline_bundle is not None:
        candidate_identity = baseline_bundle.get("public_baseline_identity")
        if not _valid_public_baseline_identity(candidate_identity):
            raise RuntimeError("frozen public Direct snapshot identity is unavailable")
        public_baseline_identity = copy.deepcopy(candidate_identity)
    COMMON._atomic_write_json(
        path,
        {
            "schema_version": 1,
            "kind": "kvbench-mlsysforge-public-private-evaluation",
            "status": "terminal",
            "classification": classification,
            "started_at_utc": started_at_utc,
            "completed_at_utc": COMMON.utc_now(),
            "candidate_tree_sha256": candidate_sha256,
            "commitments": {
                "public_baseline_identity": public_baseline_identity,
            },
            "runner": runner_metadata,
            "runner_evidence": runner_evidence,
            "public_result": public_result,
            "public_result_sha256": COMMON.canonical_json_sha256(public_result),
            "private_errors": private_errors,
        },
    )


def _prepare_log_dir(raw: Path) -> Path:
    if raw.exists():
        if raw.is_symlink() or not raw.is_dir():
            raise RuntimeError("trusted public log directory is unsafe")
    else:
        raw.mkdir(mode=0o700, parents=True)
    resolved = raw.resolve(strict=True)
    resolved.chmod(0o700)
    return resolved


def _make_eval_scratch(host_scratch: Path) -> Path:
    name = "public-" + str(os.getpid()) + "-" + secrets.token_hex(8)
    path = host_scratch / name
    path.mkdir(mode=0o700)
    return path


def _clear_eval_scratch(path: Path | None, host_scratch: Path | None) -> None:
    if path is None or host_scratch is None:
        return
    resolved = path.resolve(strict=True)
    if (
        resolved.parent != host_scratch
        or not resolved.name.startswith("public-")
        or resolved.is_symlink()
    ):
        raise RuntimeError("refusing unsafe public scratch cleanup")
    shutil.rmtree(resolved)


def _emit(result: dict[str, Any]) -> None:
    print(">>>>> Start Structured Result")
    print(json.dumps(result, allow_nan=False, ensure_ascii=True, sort_keys=True))
    print(">>>>> End Structured Result")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at_utc = COMMON.utc_now()
    log_dir = _prepare_log_dir(args.log_dir)
    private_path = log_dir / "kvbench-public-private-evidence.json"
    classification = "infrastructure_error"
    public_result = infrastructure_failure()
    runner_evidence: dict[str, Any] | None = None
    baseline_bundle: dict[str, Any] | None = None
    candidate_sha256: str | None = None
    runner_metadata: dict[str, Any] = {
        "returncode": None,
        "timed_out": False,
        "cancel_signal": None,
        "termination_sent": False,
        "forced_kill": False,
        "runtime_seconds": 0.0,
        "force_reconcile": None,
    }
    private_errors: list[str] = []
    host_scratch: Path | None = None
    eval_scratch: Path | None = None
    candidate_dir: Path | None = None
    migration: Path | None = None

    try:
        migration, host_scratch, nonce_source = _validate_runtime_paths()
        canonical_runner = migration / "scripts/run_public_candidate.py"
        if not canonical_runner.is_file() or canonical_runner.is_symlink():
            raise RuntimeError("canonical public runner is unavailable")
        if (
            not INSTALLED_PUBLIC_RUNNER.is_file()
            or INSTALLED_PUBLIC_RUNNER.is_symlink()
            or COMMON.file_sha256(canonical_runner)
            != COMMON.file_sha256(INSTALLED_PUBLIC_RUNNER)
        ):
            raise RuntimeError("canonical and image public runner bytes disagree")
        baseline_bundle = _load_public_baseline_bundle()
        baseline = baseline_bundle["leg"]
        eval_scratch = _make_eval_scratch(host_scratch)
        candidate_dir = eval_scratch / "candidate"
        evidence_path = eval_scratch / "evidence.json"
        candidate_sha256 = secure_copy_submission(args.workspace, candidate_dir)
        command = [
            sys.executable,
            str(canonical_runner),
            "--candidate-dir",
            str(candidate_dir),
            "--expected-candidate-tree-sha256",
            candidate_sha256,
            "--public-workload-nonce-source",
            str(nonce_source),
            "--output",
            str(evidence_path),
        ]
        runner_metadata = _run_supervised(
            command,
            environment={
                **os.environ,
                "NVIDIA_VISIBLE_DEVICES": "void",
            },
            runner_log=log_dir / "kvbench-public-runner.log",
        )
        if runner_metadata["forced_kill"] is True and migration is not None:
            runner_metadata["force_reconcile"] = COMMON._force_runtime_reconcile(
                migration, candidate_dir
            )
        if evidence_path.is_file() and not evidence_path.is_symlink():
            runner_evidence = COMMON._strict_json_object(evidence_path)
        if runner_metadata["cancel_signal"] is not None:
            classification = "cancelled"
            raise RuntimeError("trusted public runner was cancelled")
        if runner_metadata["timed_out"] is True:
            raise RuntimeError("trusted public runner timed out")
        if runner_metadata["forced_kill"] is True:
            raise RuntimeError("trusted public runner exceeded cleanup grace")
        if runner_evidence is None:
            raise RuntimeError("trusted public runner exited without evidence")
        if runner_metadata["returncode"] == 4:
            if runner_evidence.get("status") == "resource_busy":
                classification = "resource_busy"
            raise RuntimeError("trusted public measurement resource is busy")
        candidate, evaluation, status = _validate_complete_report(
            runner_evidence,
            candidate_sha256,
            runner_metadata["returncode"],
            baseline_bundle,
        )
        public_result = public_structured_result(
            candidate, evaluation, baseline, status=status
        )
        classification = "valid" if status == "complete" else "candidate_invalid"
    except PublicSubmissionError as exc:
        classification = "candidate_invalid"
        private_errors.append(f"PublicSubmissionError: {exc}")
        public_result = candidate_failure(str(exc))
    except BaseException as exc:
        private_errors.append(f"{type(exc).__name__}: {exc}")
        public_result = infrastructure_failure(classification)
    finally:
        try:
            _clear_eval_scratch(eval_scratch, host_scratch)
        except BaseException as cleanup_exc:
            classification = "infrastructure_error"
            private_errors.append(
                f"scratch cleanup: {type(cleanup_exc).__name__}: {cleanup_exc}"
            )
            public_result = infrastructure_failure()
        if baseline_bundle is not None:
            try:
                _assert_public_baseline_snapshot_current(baseline_bundle)
            except BaseException as drift_exc:
                classification = "infrastructure_error"
                private_errors.append(
                    "baseline drift: "
                    f"{type(drift_exc).__name__}: {drift_exc}"
                )
                public_result = infrastructure_failure()
        try:
            validate_public_output(public_result, classification)
        except BaseException as validation_exc:
            classification = "infrastructure_error"
            private_errors.append(
                f"public validation: {type(validation_exc).__name__}: {validation_exc}"
            )
            public_result = infrastructure_failure()
        _write_private_envelope(
            private_path,
            started_at_utc=started_at_utc,
            classification=classification,
            candidate_sha256=candidate_sha256,
            runner_metadata=runner_metadata,
            runner_evidence=runner_evidence,
            public_result=public_result,
            private_errors=private_errors,
            baseline_bundle=baseline_bundle,
        )

    _emit(public_result)
    # Candidate-invalid and infrastructure outcomes are represented in the
    # structured result.  A valid invocation always leaves parsable evidence.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
