#!/usr/bin/env python3
"""Fail-closed validation for formal two-hour Flash-KMeans run artifacts.

The ``pre`` phase validates the timeout-boundary archive and the three
host-scheduled snapshots before that archive is sent to the independent judge.
The ``post`` phase binds the independent report and the exact bytes evaluated
by that judge back to the pre-validated archive.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
import sys
import tarfile
from pathlib import Path
from typing import Any


class ArtifactValidationError(ValueError):
    """Raised when formal-run artifacts do not satisfy the fixed-run contract."""


MAX_ARCHIVE_SIZE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4096
MAX_MEMBER_PATH_BYTES = 1024
MAX_SINGLE_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_FILE_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
EXPECTED_ADAPTER_SOURCE = "flash_kmeans_h100_verifier_adapter"
PRIVATE_JUDGE_ARTIFACT_NAME = "private_judge_artifacts.tar.gz"
PRIVATE_JUDGE_MEMBER_NAME = "flash_kmeans_results.json"
# The container is frozen and the boundary snapshot is recorded before the
# outer agent runner returns and records its final runtime. Those independent
# monotonic reads naturally differ by a few milliseconds. Keep this tolerance
# far below the existing 60-second fixed-budget window while accepting that
# expected reporting skew.
SNAPSHOT_RUNTIME_REPORTING_SKEW_SECONDS = 1.0
# Scheduling modes recorded by the launcher.  Under the single-GPU lease the
# Work container is frozen and strictly retired (destroy + NVML idle
# attestation) inside the deadline callback before the agent exec returns, so
# the recorded Work runtime exceeds the snapshot boundary by that bounded
# retirement time instead of preceding it.
EVALUATION_MODE = "dual_gpu_independent_final"
SINGLE_GPU_EVALUATION_MODE = (
    "single_gpu_lease_manual_judge_then_standalone_final_handoff"
)
EVALUATION_MODES = (EVALUATION_MODE, SINGLE_GPU_EVALUATION_MODE)


# --- single-GPU lease handoff -------------------------------------------------
# Under the single-GPU lease the coordinator names the standalone final Judge
# submission ``final-<12 hex>`` and records the whole final in the source run's
# durable resource lease, so the final report is bound to that lease record
# (CLOSED, this final run id, this archive digest, this report digest) instead
# of the legacy submission-id equality.  The agent's work boundary is the seal
# (admission closed, Work paused, never-resume persisted): a queued Judge may be
# drained afterwards, so the frozen snapshot can legitimately be later than the
# seal, and the Work container is strictly retired (destroy + NVML attestation)
# before the agent exec returns, so the recorded runtime exceeds the snapshot.
SINGLE_GPU_RETIREMENT_WINDOW_SECONDS = 120.0
SINGLE_GPU_MAX_JUDGE_DRAIN_SECONDS = 7200.0
_FINAL_HANDOFF_SUBMISSION_ID = __import__("re").compile(r"^final-[0-9a-f]{12}$")


def _single_gpu_seal_boundary(snapshot, runtime, budget_seconds, tolerance, finite, error):
    """Validate the seal/snapshot/runtime ordering of a single-GPU run.

    Returns the seal elapsed seconds.  ``finite`` converts a value to a float or
    raises; ``error`` builds the validator's exception type.
    """
    freeze = snapshot.get("freeze")
    if not isinstance(freeze, dict):
        raise error("final snapshot lacks its freeze record")
    seal = finite(freeze.get("seal_requested_elapsed_seconds"))
    elapsed = finite(snapshot.get("snapshot_elapsed_seconds"))
    if abs(seal - float(budget_seconds)) > float(tolerance):
        raise error(
            f"single-GPU seal is outside the fixed-budget boundary: seal={seal}, budget={budget_seconds}"
        )
    if elapsed < seal - 1.0 or elapsed - seal > SINGLE_GPU_MAX_JUDGE_DRAIN_SECONDS:
        raise error(f"final snapshot does not follow the seal within the drain ceiling: seal={seal}, snapshot={elapsed}")
    if float(runtime) < elapsed - 1.0 or float(runtime) - elapsed > SINGLE_GPU_RETIREMENT_WINDOW_SECONDS:
        raise error(f"final snapshot boundary does not bracket the retired Work runtime: snapshot={elapsed}, runtime={runtime}")
    return seal


def _validate_final_handoff_lease(task_root, *, final_eval_id, task_id, expected_archive_sha256, report_sha256, submission_id, error):
    """Bind the persisted final report to the coordinator's durable lease."""
    import json as _json
    if not isinstance(submission_id, str) or _FINAL_HANDOFF_SUBMISSION_ID.fullmatch(submission_id) is None:
        raise error("coordinator final report submission identity is invalid")
    lease_path = Path(task_root) / "resource_lease.json"
    if lease_path.is_symlink() or not lease_path.is_file():
        raise error("durable resource lease record is missing")
    try:
        lease = _json.loads(lease_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise error(f"durable resource lease record is unreadable: {exc}")
    identity = lease.get("identity") if isinstance(lease, dict) else None
    final_judge = lease.get("final_judge") if isinstance(lease, dict) else None
    if (
        not isinstance(lease, dict)
        or lease.get("state") != "CLOSED"
        or lease.get("recovery_policy") != "never_resume"
        or not isinstance(identity, dict)
        or identity.get("task_id") != task_id
        or identity.get("run_id") != Path(task_root).resolve().parent.name
        or not isinstance(final_judge, dict)
        or not isinstance(final_judge.get("request"), dict)
        or not isinstance(final_judge.get("artifacts"), dict)
    ):
        raise error("durable resource lease did not close on this final Judge")
    request = final_judge["request"]
    artifacts = final_judge["artifacts"]
    expected = expected_archive_sha256.strip().lower()
    if (
        request.get("final_run_id") != final_eval_id
        or request.get("source_run_id") != identity.get("run_id")
        or request.get("task_id") != task_id
        or request.get("archive_sha256") != "sha256:" + expected
        or not isinstance(artifacts.get("report"), dict)
        or artifacts["report"].get("sha256") != "sha256:" + report_sha256
        or not isinstance(artifacts.get("submission"), dict)
        or artifacts["submission"].get("sha256") != "sha256:" + expected
    ):
        raise error("durable resource lease does not bind this final report")


def _reject_json_constant(value: str) -> None:
    raise ArtifactValidationError(f"non-finite JSON constant is not allowed: {value}")


def _absolute_without_symlink_resolution(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.abspath(path))


def _reject_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            component_stat = current.lstat()
        except OSError as exc:
            raise ArtifactValidationError(
                f"{label} is missing at path component: {current}"
            ) from exc
        if stat.S_ISLNK(component_stat.st_mode):
            raise ArtifactValidationError(
                f"{label} has a symlink path component: {current}"
            )


def _require_directory(path: Path, label: str) -> Path:
    path = _absolute_without_symlink_resolution(path)
    _reject_symlink_components(path, label)
    try:
        directory_stat = path.lstat()
    except OSError as exc:
        raise ArtifactValidationError(f"{label} is missing: {path}") from exc
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise ArtifactValidationError(
            f"{label} must be a directory, not a symlink: {path}"
        )
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ArtifactValidationError(f"{label} cannot be resolved: {path}") from exc


def _require_nonempty_file(
    path: Path,
    label: str,
    *,
    root: Path | None = None,
) -> Path:
    path = _absolute_without_symlink_resolution(path)
    _reject_symlink_components(path, label)
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise ArtifactValidationError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ArtifactValidationError(
            f"{label} must be a regular, non-symlink file: {path}"
        )
    if file_stat.st_size <= 0:
        raise ArtifactValidationError(f"{label} is empty: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ArtifactValidationError(f"{label} cannot be resolved: {path}") from exc
    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ArtifactValidationError(
                f"{label} resolves outside its expected root {root}: {resolved}"
            ) from exc
    return resolved


def _load_json_object_with_sha256(
    path: Path,
    label: str,
    *,
    root: Path | None = None,
) -> tuple[dict[str, Any], Path, str]:
    path, raw, digest = _read_file_with_sha256(
        path,
        label,
        root=root,
        max_bytes=MAX_JSON_BYTES,
    )
    try:
        text = raw.decode("utf-8")
        value = json.loads(text, parse_constant=_reject_json_constant)
    except ArtifactValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"{label} is not readable strict JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{label} must contain a JSON object: {path}")
    return value, path, digest


def _load_json_object(
    path: Path,
    label: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    value, _, _ = _load_json_object_with_sha256(
        path,
        label,
        root=root,
    )
    return value


def _finite_number(
    value: object,
    *,
    field: str,
    nullable: bool = False,
) -> float | int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{field} must be a finite number")
    if not math.isfinite(float(value)):
        raise ArtifactValidationError(f"{field} must be a finite number")
    return value


def _exact_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactValidationError(f"{field} must be a non-negative integer")
    return value


def _read_file_with_sha256(
    path: Path,
    label: str,
    *,
    root: Path | None = None,
    max_bytes: int | None = None,
) -> tuple[Path, bytes, str]:
    path = _require_nonempty_file(path, label, root=root)
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise ArtifactValidationError(
            f"{label} exceeds the byte limit: "
            f"{path.stat().st_size} > {max_bytes}"
        )
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_size <= 0:
                raise ArtifactValidationError(
                    f"{label} changed while opening: {path}"
                )
            if max_bytes is not None and opened_stat.st_size > max_bytes:
                raise ArtifactValidationError(
                    f"{label} exceeds the byte limit: "
                    f"{opened_stat.st_size} > {max_bytes}"
                )
            raw = handle.read()
            closed_stat = os.fstat(handle.fileno())
    except OSError as exc:
        raise ArtifactValidationError(f"cannot read {label}: {path}") from exc
    if (
        len(raw) != opened_stat.st_size
        or closed_stat.st_size != opened_stat.st_size
        or closed_stat.st_mtime_ns != opened_stat.st_mtime_ns
    ):
        raise ArtifactValidationError(f"{label} changed while being read: {path}")
    return path, raw, hashlib.sha256(raw).hexdigest()


def _validated_measurements(
    report: dict[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(report.get("valid"), bool):
        raise ArtifactValidationError(f"{label}.valid must be a boolean")
    pass_rate = _finite_number(
        report.get("pass_rate"), field=f"{label}.pass_rate"
    )
    assert pass_rate is not None
    if not 0.0 <= float(pass_rate) <= 1.0:
        raise ArtifactValidationError(
            f"{label}.pass_rate must be between 0 and 1"
        )
    _finite_number(
        report.get("score"),
        field=f"{label}.score",
        nullable=True,
    )
    _finite_number(
        report.get("score_0_100"),
        field=f"{label}.score_0_100",
        nullable=True,
    )


def _validate_adapter_provenance(
    report: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Require the single-record, output-suppressed private-judge adapter."""
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ArtifactValidationError(f"{label}.metrics must be an object")

    source = metrics.get("source")
    if source != EXPECTED_ADAPTER_SOURCE:
        raise ArtifactValidationError(
            f"{label}.metrics.source must be {EXPECTED_ADAPTER_SOURCE!r}, "
            f"got {source!r}"
        )
    for field in ("candidate_output_suppressed", "verifier_completed"):
        if metrics.get(field) is not True:
            raise ArtifactValidationError(
                f"{label}.metrics.{field} must be true, "
                f"got {metrics.get(field)!r}"
            )

    for field, expected in (
        ("adapter_schema_version", 8),
        ("feedback_schema_version", 7),
        ("result_protocol_records", 1),
        ("verifier_returncode", 0),
    ):
        value = metrics.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ArtifactValidationError(
                f"{label}.metrics.{field} must be integer {expected}, got {value!r}"
            )
    feedback = metrics.get("deployment_family_feedback")
    if not isinstance(feedback, dict) or set(feedback) != {
        "large-K",
        "large-N/high-D",
    }:
        raise ArtifactValidationError(
            f"{label}.metrics.deployment_family_feedback is incomplete"
        )
    for family, entry in feedback.items():
        if not isinstance(entry, dict) or entry.get("status") != "passed":
            raise ArtifactValidationError(
                f"{label} accepted {family} feedback must have passed status"
            )
        band = entry.get("warmed_latency_band_ms")
        if not isinstance(band, dict) or set(band) != {"lower_ms", "upper_ms"}:
            raise ArtifactValidationError(
                f"{label} accepted {family} feedback lacks one coarse band"
            )
        lower = _finite_number(
            band.get("lower_ms"), field=f"{label}.{family}.lower_ms"
        )
        upper = _finite_number(
            band.get("upper_ms"), field=f"{label}.{family}.upper_ms"
        )
        assert lower is not None and upper is not None
        if lower <= 0.0 or not math.isclose(
            upper / lower, 1.25, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise ArtifactValidationError(
                f"{label} accepted {family} feedback band is malformed"
            )
    return metrics


def _validate_final_archive(
    archive_path: Path,
    *,
    expected_size: int,
    root: Path,
) -> dict[str, Any]:
    archive_path, archive_bytes, archive_sha256 = _read_file_with_sha256(
        archive_path,
        "final archive",
        root=root,
        max_bytes=MAX_ARCHIVE_SIZE_BYTES,
    )
    actual_size = len(archive_bytes)
    if actual_size != expected_size:
        raise ArtifactValidationError(
            "final archive size does not match final_result.json: "
            f"file={actual_size}, result={expected_size}"
        )

    names: set[str] = set()
    regular_files: set[str] = set()
    member_count = 0
    total_file_bytes = 0
    try:
        with tarfile.open(
            fileobj=io.BytesIO(archive_bytes),
            mode="r:gz",
        ) as bundle:
            for member in bundle:
                member_count += 1
                if member_count > MAX_ARCHIVE_MEMBERS:
                    raise ArtifactValidationError(
                        "final archive exceeds the member-count limit: "
                        f"{member_count} > {MAX_ARCHIVE_MEMBERS}"
                    )
                name = member.name
                while name.startswith("./"):
                    name = name[2:]
                if not name:
                    raise ArtifactValidationError(
                        f"final archive has an empty member name: {member.name!r}"
                    )
                try:
                    path_bytes = len(name.encode("utf-8"))
                except UnicodeEncodeError as exc:
                    raise ArtifactValidationError(
                        "final archive member path is not valid UTF-8 text: "
                        f"{member.name!r}"
                    ) from exc
                if path_bytes > MAX_MEMBER_PATH_BYTES:
                    raise ArtifactValidationError(
                        "final archive member path exceeds the byte limit: "
                        f"{path_bytes} > {MAX_MEMBER_PATH_BYTES}"
                    )
                parts = name.split("/")
                if (
                    name.startswith("/")
                    or any(part in {"", ".", ".."} for part in parts)
                ):
                    raise ArtifactValidationError(
                        f"final archive has an unsafe member: {member.name!r}"
                    )
                if name in names:
                    raise ArtifactValidationError(
                        f"final archive has a duplicate member: {name!r}"
                    )
                names.add(name)

                if not (
                    name in {"kernels", "FINAL_VERSION", "notes.md"}
                    or name.startswith("kernels/")
                ):
                    raise ArtifactValidationError(
                        f"final archive member is outside the whitelist: {name!r}"
                    )
                if member.isdir():
                    if not (name == "kernels" or name.startswith("kernels/")):
                        raise ArtifactValidationError(
                            f"final archive root artifact is not a file: {name!r}"
                        )
                elif member.isfile():
                    if name == "kernels":
                        raise ArtifactValidationError(
                            "final archive member 'kernels' must be a directory"
                        )
                    regular_files.add(name)
                    if member.size < 0:
                        raise ArtifactValidationError(
                            f"final archive member has a negative size: {name!r}"
                        )
                    if member.size > MAX_SINGLE_FILE_BYTES:
                        raise ArtifactValidationError(
                            "final archive member exceeds the file-size limit: "
                            f"{name!r} {member.size} > {MAX_SINGLE_FILE_BYTES}"
                        )
                    total_file_bytes += member.size
                    if total_file_bytes > MAX_TOTAL_FILE_BYTES:
                        raise ArtifactValidationError(
                            "final archive exceeds the total logical-size limit: "
                            f"{total_file_bytes} > {MAX_TOTAL_FILE_BYTES}"
                        )
                else:
                    raise ArtifactValidationError(
                        "final archive permits only regular files and directories; "
                        f"got {name!r} type={member.type!r}"
                    )
    except ArtifactValidationError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ArtifactValidationError(
            f"final archive is not a readable gzip tar: {archive_path}"
        ) from exc

    required = {"FINAL_VERSION", "notes.md"}
    missing = sorted(required - regular_files)
    if missing:
        raise ArtifactValidationError(
            f"final archive lacks required regular files: {missing}"
        )
    kernel_files = sorted(
        name for name in regular_files if name.startswith("kernels/")
    )
    if not kernel_files:
        raise ArtifactValidationError(
            "final archive contains no regular files below kernels/"
        )
    for regular_name in regular_files:
        descendant_prefix = regular_name + "/"
        conflicting = next(
            (
                name
                for name in names
                if name.startswith(descendant_prefix)
            ),
            None,
        )
        if conflicting is not None:
            raise ArtifactValidationError(
                "final archive has a regular-file ancestor conflict: "
                f"{regular_name!r} contains descendant {conflicting!r}"
            )

    return {
        "path": str(archive_path),
        "size_bytes": actual_size,
        "sha256": archive_sha256,
        "member_count": member_count,
        "regular_file_count": len(regular_files),
        "kernel_file_count": len(kernel_files),
        "logical_size_bytes": total_file_bytes,
    }


def _validate_private_judge_artifact(
    artifact_path: Path,
    *,
    root: Path,
    task_id: str,
    expected_score: float,
) -> dict[str, Any]:
    """Validate the host-only exact per-case result exported by SForge."""

    artifact_path, artifact_bytes, artifact_sha256 = _read_file_with_sha256(
        artifact_path,
        "private Judge artifact",
        root=root,
        max_bytes=MAX_ARCHIVE_SIZE_BYTES,
    )
    try:
        with tarfile.open(
            fileobj=io.BytesIO(artifact_bytes),
            mode="r:gz",
        ) as bundle:
            members = bundle.getmembers()
            if (
                len(members) != 1
                or members[0].name != PRIVATE_JUDGE_MEMBER_NAME
                or not members[0].isfile()
                or members[0].size <= 0
                or members[0].size > MAX_JSON_BYTES
            ):
                raise ArtifactValidationError(
                    "private Judge artifact must contain exactly one bounded "
                    f"regular {PRIVATE_JUDGE_MEMBER_NAME}"
                )
            stream = bundle.extractfile(members[0])
            if stream is None:
                raise ArtifactValidationError(
                    "private Judge artifact member is unreadable"
                )
            detail_bytes = stream.read(MAX_JSON_BYTES + 1)
    except ArtifactValidationError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ArtifactValidationError(
            "private Judge artifact is not a readable tar.gz"
        ) from exc
    if len(detail_bytes) != members[0].size:
        raise ArtifactValidationError(
            "private Judge artifact member is truncated"
        )
    try:
        detail = json.loads(
            detail_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except ArtifactValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            "private Judge result is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(detail, dict):
        raise ArtifactValidationError(
            "private Judge result must be a JSON object"
        )
    if detail.get("schema_version") != 6:
        raise ArtifactValidationError(
            "private Judge result must use schema_version=6"
        )
    if detail.get("task") != task_id:
        raise ArtifactValidationError(
            f"private Judge result task must be {task_id!r}"
        )
    if detail.get("score_profile") != "v7_family_tail60":
        raise ArtifactValidationError(
            "private Judge result must use score_profile='v7_family_tail60'"
        )
    expected_group_weights = {
        "assign": 0.10,
        "update": 0.10,
        "full": 0.20,
        "deploy_control": 0.10,
        "deploy_large_k": 0.25,
        "deploy_large_n_high_d": 0.25,
    }
    if detail.get("score_group_weights") != expected_group_weights:
        raise ArtifactValidationError(
            "private Judge result has the wrong v7 score-group weights"
        )
    statuses = detail.get("coarse_deployment_family_statuses")
    if statuses != {"large_k": "passed", "large_n_high_d": "passed"}:
        raise ArtifactValidationError(
            "accepted private Judge result lacks passed coarse family statuses"
        )
    cases = detail.get("cases")
    if not isinstance(cases, list) or len(cases) != 22:
        raise ArtifactValidationError(
            "private Judge result must contain all 22 hidden rows"
        )

    names: set[str] = set()
    performance_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for index, row in enumerate(cases):
        if not isinstance(row, dict):
            raise ArtifactValidationError(
                f"private Judge cases[{index}] must be an object"
            )
        name = row.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ArtifactValidationError(
                "private Judge case names must be unique non-empty strings"
            )
        names.add(name)
        stage = row.get("stage")
        if stage not in {"assign", "update", "full"}:
            raise ArtifactValidationError(
                f"private Judge case {name} has an invalid stage"
            )
        shape = row.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in shape
            )
        ):
            raise ArtifactValidationError(
                f"private Judge case {name} has an invalid shape"
            )
        weight = _finite_number(
            row.get("weight"),
            field=f"private Judge case {name}.weight",
        )
        assert weight is not None
        if float(weight) < 0.0:
            raise ArtifactValidationError(
                f"private Judge case {name}.weight must be non-negative"
            )
        if float(weight) > 0.0:
            latency = _finite_number(
                row.get("candidate_ms"),
                field=f"private Judge case {name}.candidate_ms",
            )
            assert latency is not None
            if float(latency) <= 0.0 or row.get("all_gates_ok") is not True:
                raise ArtifactValidationError(
                    f"private Judge performance case {name} is not a valid measurement"
                )
            if row.get("measurement_mode") == "fresh_process":
                for field in (
                    "candidate_first_run_ms",
                    "candidate_warmed_iteration_ms",
                ):
                    value = _finite_number(
                        row.get(field),
                        field=f"private Judge case {name}.{field}",
                    )
                    if value is None or float(value) <= 0.0:
                        raise ArtifactValidationError(
                            f"private Judge deployment case {name} lacks {field}"
                        )
            performance_rows.append(row)
        else:
            gate_rows.append(row)

    if len(performance_rows) != 14 or len(gate_rows) != 8:
        raise ArtifactValidationError(
            "private Judge result must contain 14 scored rows and 8 zero-weight gates"
        )
    weight_sum = sum(float(row["weight"]) for row in performance_rows)
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ArtifactValidationError(
            f"private Judge performance weights must sum to 1, got {weight_sum}"
        )
    group_totals = {name: 0.0 for name in expected_group_weights}
    group_logs = {name: 0.0 for name in expected_group_weights}
    for row in performance_rows:
        group = row.get("score_group")
        if group not in expected_group_weights:
            raise ArtifactValidationError(
                f"private Judge case {row['name']} has an invalid score group"
            )
        group_totals[str(group)] += float(row["weight"])
    for group, expected_weight in expected_group_weights.items():
        if not math.isclose(
            group_totals[group],
            expected_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ArtifactValidationError(
                "private Judge group weights do not match v7_family_tail60"
            )
    group_values: dict[str, float] = {}
    for group, group_weight in expected_group_weights.items():
        selected = [row for row in performance_rows if row["score_group"] == group]
        if not selected:
            raise ArtifactValidationError(
                f"private Judge score group {group} is empty"
            )
        group_logs[group] = sum(
            (float(row["weight"]) / group_weight)
            * math.log(float(row["candidate_ms"]))
            for row in selected
        )
        group_values[group] = math.exp(group_logs[group])
    recomputed_score = math.exp(
        sum(
            expected_group_weights[group] * math.log(group_values[group])
            for group in expected_group_weights
        )
    )
    detail_score = _finite_number(
        detail.get("score"),
        field="private Judge result.score",
    )
    assert detail_score is not None
    for label, value in (
        ("private detail", float(detail_score)),
        ("formal report", float(expected_score)),
    ):
        if not math.isclose(
            recomputed_score,
            value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ArtifactValidationError(
                f"private Judge recomputed score differs from {label}: "
                f"recomputed={recomputed_score}, recorded={value}"
            )

    return {
        "path": str(artifact_path),
        "size_bytes": len(artifact_bytes),
        "sha256": artifact_sha256,
        "member": PRIVATE_JUDGE_MEMBER_NAME,
        "member_size_bytes": len(detail_bytes),
        "member_sha256": hashlib.sha256(detail_bytes).hexdigest(),
        "case_count": len(cases),
        "performance_case_count": len(performance_rows),
        "gate_case_count": len(gate_rows),
        "weighted_latency_ms": recomputed_score,
    }


def validate_pre_final(
    task_root: Path,
    *,
    run_id: str,
    task_id: str,
    model: str,
    budget_seconds: int,
    interval_seconds: int,
    evaluation_mode: str = EVALUATION_MODE,
) -> dict[str, Any]:
    """Validate all main-run artifacts and return their integrity evidence."""
    task_root = _require_directory(task_root, "task root")
    if evaluation_mode not in EVALUATION_MODES:
        raise ArtifactValidationError(f"unknown evaluation mode: {evaluation_mode}")
    if budget_seconds <= 0 or interval_seconds <= 0:
        raise ArtifactValidationError("budget and interval must be positive")
    expected_rounds = [
        f"auto-{index}"
        for index, _ in enumerate(
            range(interval_seconds, budget_seconds, interval_seconds), start=1
        )
    ]

    result = _load_json_object(
        task_root / "final_result.json",
        "final_result.json",
        root=task_root,
    )
    snapshot = _load_json_object(
        task_root / "final_snapshot.json",
        "final_snapshot.json",
        root=task_root,
    )
    history = _load_json_object(
        task_root / "run_history.json",
        "run_history.json",
        root=task_root,
    )

    expected_identity = {
        "agent": "codex",
        "task": task_id,
        "run_id": run_id,
        "model": model,
    }
    for field, expected in expected_identity.items():
        if result.get(field) != expected:
            raise ArtifactValidationError(
                f"final_result.json.{field} must be {expected!r}, "
                f"got {result.get(field)!r}"
            )
    if result.get("timed_out") is not True:
        raise ArtifactValidationError(
            "final_result.json.timed_out must be true for the fixed run"
        )

    runtime = _finite_number(
        result.get("runtime_seconds"),
        field="final_result.json.runtime_seconds",
    )
    assert runtime is not None
    if evaluation_mode != SINGLE_GPU_EVALUATION_MODE and not budget_seconds - 5.0 <= float(runtime) <= budget_seconds + 60.0:
        raise ArtifactValidationError(
            "final_result.json.runtime_seconds does not prove the fixed budget: "
            f"{runtime}"
        )

    if result.get("auto_submissions") == 0:
        expected_rounds = []

    expected_auto_count = len(expected_rounds)
    result_agent_count = _exact_nonnegative_int(
        result.get("agent_submissions"),
        field="final_result.json.agent_submissions",
    )
    expected_counts = {
        "auto_submissions": expected_auto_count,
        "total_rounds": expected_auto_count + result_agent_count,
    }
    for field, expected in expected_counts.items():
        actual = _exact_nonnegative_int(
            result.get(field), field=f"final_result.json.{field}"
        )
        if actual != expected:
            raise ArtifactValidationError(
                f"final_result.json.{field} must be {expected}, got {actual}"
            )

    archive_size = _exact_nonnegative_int(
        result.get("archive_size_bytes"),
        field="final_result.json.archive_size_bytes",
    )
    if archive_size <= 0:
        raise ArtifactValidationError(
            "final_result.json.archive_size_bytes must be positive"
        )

    elapsed = _finite_number(
        snapshot.get("snapshot_elapsed_seconds"),
        field="final_snapshot.json.snapshot_elapsed_seconds",
    )
    captured_at = _finite_number(
        snapshot.get("captured_at"),
        field="final_snapshot.json.captured_at",
    )
    assert elapsed is not None and captured_at is not None
    if float(captured_at) <= 0:
        raise ArtifactValidationError(
            "final_snapshot.json.captured_at must be positive"
        )
    seal_elapsed = None
    if evaluation_mode == SINGLE_GPU_EVALUATION_MODE:
        seal_elapsed = _single_gpu_seal_boundary(snapshot, runtime, budget_seconds, 60.0, lambda v: float(_finite_number(v, field="single-GPU boundary")), ArtifactValidationError)
    else:
        if abs(float(elapsed) - budget_seconds) > 60.0:
            raise ArtifactValidationError(
                "final snapshot is outside the fixed-run tolerance: "
                f"elapsed={elapsed}, budget={budget_seconds}"
            )
        if (
            float(elapsed) + SNAPSHOT_RUNTIME_REPORTING_SKEW_SECONDS
            < float(runtime)
        ):
            raise ArtifactValidationError(
                "final snapshot precedes the measured agent runtime: "
                f"snapshot={elapsed}, runtime={runtime}"
            )

    if history.get("run_id") != run_id:
        raise ArtifactValidationError(
            f"run_history.json.run_id must be {run_id!r}, "
            f"got {history.get('run_id')!r}"
        )
    for field, expected in (
        ("agent_submissions", result_agent_count),
        ("auto_submissions", expected_auto_count),
    ):
        actual = _exact_nonnegative_int(
            history.get(field), field=f"run_history.json.{field}"
        )
        if actual != expected:
            raise ArtifactValidationError(
                f"run_history.json.{field} must be {expected}, got {actual}"
            )

    entries = history.get("entries")
    if not isinstance(entries, list):
        raise ArtifactValidationError("run_history.json.entries must be a list")
    expected_total_count = expected_auto_count + result_agent_count
    if len(entries) != expected_total_count:
        raise ArtifactValidationError(
            "run_history.json must contain exactly "
            f"{expected_auto_count} auto submissions and "
            f"{result_agent_count} agent submissions; "
            f"got {len(entries)} entries"
        )
    rounds = [
        entry.get("round") if isinstance(entry, dict) else None
        for entry in entries
    ]
    if len(set(rounds)) != len(rounds):
        raise ArtifactValidationError(
            f"run_history.json contains duplicate rounds: {rounds}"
        )
    auto_entries = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("round"), str)
        and entry["round"].startswith("auto-")
    ]
    agent_entries = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("round"), str)
        and entry["round"].startswith("agent-")
    ]
    auto_rounds = [entry["round"] for entry in auto_entries]
    expected_agent_rounds = [
        f"agent-{sequence}"
        for sequence in range(1, result_agent_count + 1)
    ]
    agent_rounds = [entry["round"] for entry in agent_entries]
    if auto_rounds != expected_rounds:
        raise ArtifactValidationError(
            f"unexpected run_history.json auto rounds: {auto_rounds}; "
            f"expected {expected_rounds}"
        )
    if agent_rounds != expected_agent_rounds:
        raise ArtifactValidationError(
            f"unexpected run_history.json agent rounds: {agent_rounds}; "
            f"expected {expected_agent_rounds}"
        )
    if len(auto_entries) + len(agent_entries) != len(entries):
        raise ArtifactValidationError(
            f"run_history.json contains an unexpected round kind: {rounds}"
        )

    auto_reports: list[dict[str, Any]] = []
    agent_reports: list[dict[str, Any]] = []
    for round_id, entry in [
        *zip(expected_rounds, auto_entries, strict=True),
        *zip(expected_agent_rounds, agent_entries, strict=True),
    ]:
        if not isinstance(entry, dict):
            raise ArtifactValidationError(
                f"run_history.json entry for {round_id} must be an object"
            )
        expected_entry = {
            "type": "submission",
            "status": "completed",
            "task_id": task_id,
        }
        for field, expected in expected_entry.items():
            if entry.get(field) != expected:
                raise ArtifactValidationError(
                    f"run_history.json {round_id}.{field} must be "
                    f"{expected!r}, got {entry.get(field)!r}"
                )
        submission_id = entry.get("submission_id")
        if not isinstance(submission_id, str) or not submission_id:
            raise ArtifactValidationError(
                f"run_history.json {round_id}.submission_id must be non-empty"
            )

        round_root = task_root / "submissions" / round_id
        report_path = round_root / "report.json"
        archive_path, _, archive_sha256 = _read_file_with_sha256(
            round_root / "submission.tar.gz",
            f"{round_id} submission archive",
            root=task_root,
            max_bytes=MAX_ARCHIVE_SIZE_BYTES,
        )
        report, report_path, report_sha256 = _load_json_object_with_sha256(
            report_path,
            f"{round_id} report",
            root=task_root,
        )
        if report.get("task_id") != task_id:
            raise ArtifactValidationError(
                f"{round_id} report task_id must be {task_id!r}"
            )
        if report.get("submission_id") != submission_id:
            raise ArtifactValidationError(
                f"{round_id} submission_id differs between history and report"
            )
        _validated_measurements(report, label=f"{round_id} report")
        report_evidence = (
            {
                "round_id": round_id,
                "submission_id": submission_id,
                "report_path": str(report_path),
                "report_sha256": report_sha256,
                "archive_path": str(archive_path),
                "archive_sha256": archive_sha256,
            }
        )
        if round_id.startswith("auto-"):
            auto_reports.append(report_evidence)
        else:
            agent_reports.append(report_evidence)

    archive = _validate_final_archive(
        task_root / "final_archive.tar.gz",
        expected_size=archive_size,
        root=task_root,
    )
    return {
        "schema_version": 1,
        "phase": "pre",
        "run_id": run_id,
        "task_id": task_id,
        "model": model,
        "runtime_seconds": runtime,
        "snapshot_elapsed_seconds": elapsed,
        "snapshot_captured_at": captured_at,
        # Single-GPU mode: the agent's work boundary (admission closed, Work
        # paused never-resume). The frozen snapshot may follow it by a drained
        # queued Judge; curve points are scheduled against this seal.
        "seal_requested_elapsed_seconds": seal_elapsed,
        "auto_rounds": auto_reports,
        "agent_rounds": agent_reports,
        "final_archive": archive,
    }


def validate_final_eval(
    task_root: Path,
    final_task_root: Path,
    *,
    final_eval_id: str,
    task_id: str,
    expected_archive_sha256: str,
    evaluation_mode: str = EVALUATION_MODE,
) -> dict[str, Any]:
    """Bind a valid independent report to the exact pre-validated archive."""
    task_root = _require_directory(task_root, "task root")
    if evaluation_mode not in EVALUATION_MODES:
        raise ArtifactValidationError(f"unknown evaluation mode: {evaluation_mode}")
    final_task_root = _require_directory(
        final_task_root,
        "independent final task root",
    )
    expected_digest = expected_archive_sha256.strip().lower()
    if (
        len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise ArtifactValidationError(
            "expected archive SHA-256 must be 64 lowercase hexadecimal characters"
        )

    final_archive, final_archive_bytes, current_digest = _read_file_with_sha256(
        task_root / "final_archive.tar.gz",
        "final archive",
        root=task_root,
        max_bytes=MAX_ARCHIVE_SIZE_BYTES,
    )
    if current_digest != expected_digest:
        raise ArtifactValidationError(
            "final archive changed after pre-validation: "
            f"expected={expected_digest}, actual={current_digest}"
        )

    submission_root = final_task_root / "submissions" / "manual-1"
    (
        evaluated_archive,
        evaluated_archive_bytes,
        evaluated_digest,
    ) = _read_file_with_sha256(
        submission_root / "submission.tar.gz",
        "independent final evaluation archive",
        root=final_task_root,
        max_bytes=MAX_ARCHIVE_SIZE_BYTES,
    )
    if evaluated_digest != expected_digest:
        raise ArtifactValidationError(
            "independent final evaluation did not use the pre-validated archive: "
            f"expected={expected_digest}, evaluated={evaluated_digest}"
        )

    report_path = submission_root / "report.json"
    report, report_path, report_sha256 = _load_json_object_with_sha256(
        report_path,
        "independent final report",
        root=final_task_root,
    )
    if report.get("task_id") != task_id:
        raise ArtifactValidationError(
            f"independent final report task_id must be {task_id!r}, "
            f"got {report.get('task_id')!r}"
        )
    if evaluation_mode == SINGLE_GPU_EVALUATION_MODE:
        _validate_final_handoff_lease(
            task_root,
            final_eval_id=final_eval_id,
            task_id=task_id,
            expected_archive_sha256=expected_digest,
            report_sha256=report_sha256,
            submission_id=report.get("submission_id"),
            error=ArtifactValidationError,
        )
    elif report.get("submission_id") != final_eval_id:
        raise ArtifactValidationError(
            f"independent final report submission_id must be {final_eval_id!r}, "
            f"got {report.get('submission_id')!r}"
        )
    if report.get("valid") is not True:
        raise ArtifactValidationError(
            "independent final report must have valid=true"
        )
    if report.get("timed_out") is not False:
        raise ArtifactValidationError(
            "independent final report must have timed_out=false"
        )
    pass_rate = _finite_number(
        report.get("pass_rate"),
        field="independent final report.pass_rate",
    )
    assert pass_rate is not None
    if float(pass_rate) != 1.0:
        raise ArtifactValidationError(
            f"independent final report.pass_rate must be 1.0, got {pass_rate}"
        )
    score = _finite_number(
        report.get("score"),
        field="independent final report.score",
    )
    score_0_100 = _finite_number(
        report.get("score_0_100"),
        field="independent final report.score_0_100",
        nullable=True,
    )
    assert score is not None
    if float(score) <= 0.0:
        raise ArtifactValidationError(
            f"independent final report.score must be positive, got {score}"
        )
    if (
        score_0_100 is not None
        and not 0.0 <= float(score_0_100) <= 100.0
    ):
        raise ArtifactValidationError(
            "independent final report.score_0_100 must be between 0 and 100"
        )
    adapter_metrics = _validate_adapter_provenance(
        report,
        label="independent final report",
    )
    private_judge_artifact = _validate_private_judge_artifact(
        submission_root / PRIVATE_JUDGE_ARTIFACT_NAME,
        root=final_task_root,
        task_id=task_id,
        expected_score=float(score),
    )

    return {
        "schema_version": 1,
        "phase": "post",
        "task_id": task_id,
        "final_eval_id": final_eval_id,
        "final_archive": {
            "path": str(final_archive),
            "size_bytes": len(final_archive_bytes),
            "sha256": current_digest,
        },
        "evaluated_archive": {
            "path": str(evaluated_archive),
            "size_bytes": len(evaluated_archive_bytes),
            "sha256": evaluated_digest,
        },
        "final_report": {
            "path": str(report_path),
            "sha256": report_sha256,
            "submission_id": final_eval_id,
            "valid": True,
            "timed_out": False,
            "pass_rate": pass_rate,
            "score": score,
            "score_0_100": score_0_100,
            "adapter_source": adapter_metrics["source"],
            "adapter_schema_version": adapter_metrics["adapter_schema_version"],
            "candidate_output_suppressed": True,
            "result_protocol_records": adapter_metrics["result_protocol_records"],
            "verifier_completed": True,
            "verifier_returncode": adapter_metrics["verifier_returncode"],
        },
        "private_judge_artifact": private_judge_artifact,
        "validated_report": report,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    pre = subparsers.add_parser(
        "pre", help="validate main-run artifacts before independent final eval"
    )
    pre.add_argument("--task-root", required=True, type=Path)
    pre.add_argument("--run-id", required=True)
    pre.add_argument("--task-id", required=True)
    pre.add_argument("--model", required=True)
    pre.add_argument("--budget-seconds", required=True, type=int)
    pre.add_argument("--interval-seconds", required=True, type=int)
    pre.add_argument(
        "--evaluation-mode",
        choices=EVALUATION_MODES,
        default=EVALUATION_MODE,
        help="scheduling mode recorded by the launcher (legacy dual GPU or single-GPU lease)",
    )

    post = subparsers.add_parser(
        "post", help="validate and bind the independent final evaluation"
    )
    post.add_argument("--task-root", required=True, type=Path)
    post.add_argument("--final-task-root", required=True, type=Path)
    post.add_argument("--final-eval-id", required=True)
    post.add_argument("--task-id", required=True)
    post.add_argument("--expected-archive-sha256", required=True)
    post.add_argument(
        "--evaluation-mode",
        choices=EVALUATION_MODES,
        default=EVALUATION_MODE,
        help="scheduling mode recorded by the launcher (legacy dual GPU or single-GPU lease)",
    )
    return parser


INVALID_FINAL_STATUS = "invalid_final"
INVALID_FINAL_EXIT_CODE = 3


def invalid_final_record(args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    """Describe a rejected final as a durable, machine-readable attestation.

    The record deliberately carries no `evaluated_archive`, `final_report` or
    `private_judge_artifact` binding: nothing downstream may treat it as a
    pass.  It only preserves *why* the fail-closed validator refused, so the
    reason survives the launcher's stdout.
    """
    return {
        "schema_version": 1,
        "evaluation_mode": args.evaluation_mode,
        "status": INVALID_FINAL_STATUS,
        "phase": args.phase,
        "final_eval_id": args.final_eval_id,
        "task_id": args.task_id,
        "expected_archive_sha256": args.expected_archive_sha256,
        "final_task_root": str(args.final_task_root),
        "reason": str(error) or type(error).__name__,
        "error_type": type(error).__name__,
    }


def main(argv: list[str] | None = None) -> int:
    """Validate one phase and emit its JSON attestation.

    A rejected final (post phase) still fails closed: the process exits
    non-zero, but it first prints a complete `invalid_final` record on stdout
    so the launcher can persist `formal-postcheck.json` instead of discarding
    the temporary file.
    """
    args = _build_parser().parse_args(argv)
    try:
        if args.phase == "pre":
            result = validate_pre_final(
                args.task_root,
                run_id=args.run_id,
                task_id=args.task_id,
                model=args.model,
                budget_seconds=args.budget_seconds,
                interval_seconds=args.interval_seconds,
                evaluation_mode=args.evaluation_mode,
            )
        else:
            try:
                result = validate_final_eval(
                    args.task_root,
                    args.final_task_root,
                    final_eval_id=args.final_eval_id,
                    task_id=args.task_id,
                    expected_archive_sha256=args.expected_archive_sha256,
                    evaluation_mode=args.evaluation_mode,
                )
            except Exception as exc:  # noqa: BLE001 - every rejection is recorded, none is accepted
                record = invalid_final_record(args, exc)
                print(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False))
                print(f"ERROR: {record['reason']}", file=sys.stderr)
                return INVALID_FINAL_EXIT_CODE
    except ArtifactValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
