#!/usr/bin/env python3
"""Validate periodic hidden evaluations and the independent final result."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


class ArtifactValidationError(ValueError):
    pass


MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 1024
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_FILE_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
PRIVATE_ARTIFACT = "private_judge_artifacts.tar.gz"
PRIVATE_MEMBER = "standalone_sparse_mla_prefill_v2_results.json"
FINAL_SUITE_ID = "final-holdout-v4"
EDGE_SUITE_ID = "edge-holdout-v3"
EDGE_CASE_COUNT = 470
EVALUATION_POLICY = "full-final-shared-v3"
SELECTION_PROTOCOL_ID = "sparse-mla-shared-final-v3"
ADAPTER_SOURCE = "standalone_sparse_mla_prefill_v2_verifier_adapter"
EVALUATION_MODE = "dual_h100_shared_final_v3"
SINGLE_GPU_EVALUATION_MODE = (
    "single_gpu_lease_shared_final_v3_handoff"
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
CANDIDATE_FILE = re.compile(r"candidates/kernel_v(\d{3})/submission\.py")
CANDIDATE_DIR = re.compile(r"candidates/kernel_v\d{3}")
FAMILY_KEYS = ("v32_tflops", "glm5_tflops", "v4_flash_tflops", "v4_pro_tflops")


def _reject_json_constant(value: str) -> None:
    raise ArtifactValidationError(f"non-finite JSON constant: {value}")


def _absolute(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(expanded))


def _reject_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ArtifactValidationError(f"{label} is missing: {current}") from exc
        if stat.S_ISLNK(mode):
            raise ArtifactValidationError(
                f"{label} has a symlink path component: {current}"
            )


def _directory(path: Path, label: str) -> Path:
    path = _absolute(path)
    _reject_symlink_components(path, label)
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ArtifactValidationError(f"{label} is not a directory: {path}")
    return path.resolve(strict=True)


def _read_file(
    path: Path,
    label: str,
    *,
    root: Path,
    max_bytes: int,
) -> tuple[Path, bytes, str]:
    path = _absolute(path)
    _reject_symlink_components(path, label)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(
            f"{label} is outside its expected root: {path}"
        ) from exc
    file_stat = path.lstat()
    if not stat.S_ISREG(file_stat.st_mode) or stat.S_ISLNK(file_stat.st_mode):
        raise ArtifactValidationError(
            f"{label} must be a regular, non-symlink file: {path}"
        )
    if file_stat.st_size <= 0 or file_stat.st_size > max_bytes:
        raise ArtifactValidationError(
            f"{label} has invalid size {file_stat.st_size}: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            raw = stream.read(max_bytes + 1)
            closed = os.fstat(stream.fileno())
    except OSError as exc:
        raise ArtifactValidationError(f"cannot read {label}: {path}") from exc
    if (
        len(raw) != opened.st_size
        or opened.st_size != closed.st_size
        or opened.st_mtime_ns != closed.st_mtime_ns
        or len(raw) > max_bytes
    ):
        raise ArtifactValidationError(f"{label} changed while being read: {path}")
    return resolved, raw, hashlib.sha256(raw).hexdigest()


def _json_file(
    path: Path,
    label: str,
    *,
    root: Path,
) -> tuple[dict[str, Any], Path, str]:
    resolved, raw, digest = _read_file(
        path, label, root=root, max_bytes=MAX_JSON_BYTES
    )
    try:
        value = json.loads(
            raw.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{label} must contain a JSON object")
    return value, resolved, digest


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactValidationError(f"{field} must be a finite number")
    return number


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactValidationError(f"{field} must be a non-negative integer")
    return value


def _canonical_member_name(member: tarfile.TarInfo) -> str:
    name = member.name
    while name.startswith("./"):
        name = name[2:]
    name = name.rstrip("/")
    if not name or "\\" in name:
        raise ArtifactValidationError(
            f"final archive has an unsafe member: {member.name!r}"
        )
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ArtifactValidationError(
            f"final archive has an unsafe member: {member.name!r}"
        )
    if pure.as_posix() != name:
        raise ArtifactValidationError(
            f"final archive member is not canonical: {member.name!r}"
        )
    return name


def _validate_archive(
    archive_path: Path,
    *,
    root: Path,
    expected_size: int,
) -> dict[str, Any]:
    path, archive_bytes, digest = _read_file(
        archive_path,
        "final archive",
        root=root,
        max_bytes=MAX_ARCHIVE_BYTES,
    )
    if len(archive_bytes) != expected_size:
        raise ArtifactValidationError(
            "final archive size differs from final_result.json: "
            f"{len(archive_bytes)} != {expected_size}"
        )

    names: set[str] = set()
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    total_file_bytes = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as bundle:
            for count, member in enumerate(bundle, 1):
                if count > MAX_ARCHIVE_MEMBERS:
                    raise ArtifactValidationError(
                        "final archive exceeds its member-count limit"
                    )
                name = _canonical_member_name(member)
                if name in names:
                    raise ArtifactValidationError(
                        f"final archive has duplicate member {name!r}"
                    )
                names.add(name)

                allowed_file = (
                    name in {"FINAL_VERSION", "notes.md"}
                    or CANDIDATE_FILE.fullmatch(name) is not None
                )
                allowed_directory = (
                    name == "candidates"
                    or CANDIDATE_DIR.fullmatch(name) is not None
                )
                if member.isdir():
                    if not allowed_directory:
                        raise ArtifactValidationError(
                            f"unexpected archive directory: {name!r}"
                        )
                    directories.add(name)
                    continue
                if not member.isfile() or not allowed_file:
                    raise ArtifactValidationError(
                        "final archive permits only expected regular files and "
                        f"directories; got {name!r}"
                    )
                if member.size < 0 or member.size > MAX_FILE_BYTES:
                    raise ArtifactValidationError(
                        f"archive member has invalid size: {name!r}"
                    )
                total_file_bytes += member.size
                if total_file_bytes > MAX_TOTAL_FILE_BYTES:
                    raise ArtifactValidationError(
                        "final archive exceeds its logical-size limit"
                    )
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ArtifactValidationError(
                        f"archive member is unreadable: {name!r}"
                    )
                payload = stream.read(MAX_FILE_BYTES + 1)
                if len(payload) != member.size:
                    raise ArtifactValidationError(
                        f"archive member is truncated: {name!r}"
                    )
                files[name] = payload
    except ArtifactValidationError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ArtifactValidationError("final archive is not a readable tar.gz") from exc

    for required in ("FINAL_VERSION", "notes.md"):
        if required not in files:
            raise ArtifactValidationError(
                f"final archive lacks required file {required!r}"
            )
    candidate_paths = sorted(
        name for name in files if CANDIDATE_FILE.fullmatch(name)
    )
    if not candidate_paths:
        raise ArtifactValidationError("final archive contains no candidate sources")
    version_bytes = files["FINAL_VERSION"]
    if not re.fullmatch(rb"[1-9][0-9]{0,2}\n?", version_bytes):
        raise ArtifactValidationError(
            "FINAL_VERSION must be an ASCII integer from 1 to 999 with only an "
            "optional trailing newline"
        )
    version = int(version_bytes)
    selected = f"candidates/kernel_v{version:03d}/submission.py"
    if selected not in files:
        raise ArtifactValidationError(
            f"FINAL_VERSION selects missing candidate {selected!r}"
        )
    if not files[selected]:
        raise ArtifactValidationError("selected candidate source must be nonempty")

    for regular_name in files:
        prefix = regular_name + "/"
        if any(name.startswith(prefix) for name in names):
            raise ArtifactValidationError(
                f"regular-file ancestor conflict below {regular_name!r}"
            )

    return {
        "path": str(path),
        "sha256": digest,
        "size_bytes": len(archive_bytes),
        "member_count": len(names),
        "candidate_count": len(candidate_paths),
        "selected_version": version,
        "selected_path": selected,
        "logical_size_bytes": total_file_bytes,
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
    root = _directory(task_root, "task root")
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
    result, _, _ = _json_file(
        root / "final_result.json", "final_result.json", root=root
    )
    snapshot, _, _ = _json_file(
        root / "final_snapshot.json", "final_snapshot.json", root=root
    )
    history, _, _ = _json_file(
        root / "run_history.json", "run_history.json", root=root
    )

    expected = {
        "agent": "codex",
        "task": task_id,
        "run_id": run_id,
        "model": model,
    }
    for field, wanted in expected.items():
        if result.get(field) != wanted:
            raise ArtifactValidationError(
                f"final_result.json.{field} must be {wanted!r}"
            )
    if result.get("timed_out") is not True:
        raise ArtifactValidationError("fixed Work run must end at its timeout")
    runtime = _finite(result.get("runtime_seconds"), "runtime_seconds")
    if evaluation_mode != SINGLE_GPU_EVALUATION_MODE and not budget_seconds - 5 <= runtime <= budget_seconds + 60:
        raise ArtifactValidationError(
            f"runtime {runtime} does not prove budget {budget_seconds}"
        )
    if result.get("auto_submissions") == 0:
        expected_rounds = []

    expected_auto_count = len(expected_rounds)
    agent_count = _nonnegative_int(result.get("agent_submissions"), "agent_submissions")
    for field, wanted in (
        ("auto_submissions", expected_auto_count),
        ("total_rounds", agent_count + expected_auto_count),
    ):
        actual = _nonnegative_int(result.get(field), field)
        if actual != wanted:
            raise ArtifactValidationError(f"{field} must be {wanted}, got {actual}")

    archive_size = _nonnegative_int(
        result.get("archive_size_bytes"), "archive_size_bytes"
    )
    if archive_size <= 0:
        raise ArtifactValidationError("archive_size_bytes must be positive")
    archive = _validate_archive(
        root / "final_archive.tar.gz",
        root=root,
        expected_size=archive_size,
    )

    elapsed = _finite(
        snapshot.get("snapshot_elapsed_seconds"), "snapshot_elapsed_seconds"
    )
    if evaluation_mode == SINGLE_GPU_EVALUATION_MODE:
        _single_gpu_seal_boundary(snapshot, runtime, budget_seconds, 60.0, lambda v: _finite(v, "single-GPU boundary"), ArtifactValidationError)
    else:
        if abs(elapsed - budget_seconds) > 60:
            raise ArtifactValidationError("final snapshot is outside budget tolerance")
        if elapsed + 10.0 < runtime:
            raise ArtifactValidationError("final snapshot precedes Work runtime")
    recorded_digest = snapshot.get("archive_sha256")
    if isinstance(recorded_digest, str) and recorded_digest != archive["sha256"]:
        raise ArtifactValidationError("final snapshot archive digest is inconsistent")

    if history.get("run_id") != run_id:
        raise ArtifactValidationError("run_history.json has the wrong run_id")
    for field, wanted in (
        ("agent_submissions", agent_count),
        ("auto_submissions", expected_auto_count),
    ):
        actual = _nonnegative_int(history.get(field), f"history.{field}")
        if actual != wanted:
            raise ArtifactValidationError(
                f"history.{field} must be {wanted}, got {actual}"
            )
    if "public_submissions" in history and _nonnegative_int(
        history.get("public_submissions"), "history.public_submissions"
    ) != 0:
        raise ArtifactValidationError("public tests must remain local to Work")
    entries = history.get("entries")
    if not isinstance(entries, list) or len(entries) != agent_count + expected_auto_count:
        raise ArtifactValidationError(
            f"run history must contain {agent_count} manual and {expected_auto_count} auto entries"
        )
    expected_agent_rounds = [f"agent-{index}" for index in range(1, agent_count + 1)]
    expected_all_rounds = expected_rounds + expected_agent_rounds
    rounds = [entry.get("round") if isinstance(entry, dict) else None for entry in entries]
    if rounds != expected_all_rounds:
        raise ArtifactValidationError(
            f"unexpected rounds {rounds!r}; expected {expected_all_rounds!r}"
        )

    experiment, _, _ = _json_file(
        root / "experiment-manifest.json", "experiment manifest", root=root
    )
    frozen = experiment.get("final_holdout")
    if (experiment.get("evaluation_policy") != EVALUATION_POLICY
            or experiment.get("selection_protocol_id") != SELECTION_PROTOCOL_ID
            or not isinstance(frozen, dict)
            or frozen.get("suite_id") != FINAL_SUITE_ID):
        raise ArtifactValidationError("experiment does not declare the shared final policy")
    frozen_digest = frozen.get("manifest_sha256")
    frozen_count = frozen.get("case_count")
    edge_holdout = experiment.get("edge_holdout")
    edge_digest = edge_holdout.get("manifest_sha256") if isinstance(edge_holdout, dict) else None
    if (not isinstance(edge_holdout, dict)
            or edge_holdout.get("suite_id") != EDGE_SUITE_ID
            or not isinstance(edge_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", edge_digest)):
        raise ArtifactValidationError("experiment edge-case manifest identity is malformed")
    if (not isinstance(frozen_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", frozen_digest)
            or isinstance(frozen_count, bool) or not isinstance(frozen_count, int)
            or frozen_count <= 0):
        raise ArtifactValidationError("experiment final manifest identity is malformed")
    agent_reports: list[dict[str, Any]] = []
    for round_id, entry in zip(expected_all_rounds, entries, strict=True):
        assert isinstance(entry, dict)
        for field, wanted in (
            ("type", "submission"),
            ("status", "completed"),
            ("task_id", task_id),
        ):
            if entry.get(field) != wanted:
                raise ArtifactValidationError(
                    f"history {round_id}.{field} must be {wanted!r}"
                )
        submission_id = entry.get("submission_id")
        if not isinstance(submission_id, str) or not submission_id:
            raise ArtifactValidationError(f"history {round_id} lacks submission_id")
        round_root = root / "submissions" / round_id
        report, report_path, report_digest = _json_file(
            round_root / "report.json", f"{round_id} report", root=root
        )
        if report.get("task_id") != task_id or report.get("submission_id") != submission_id:
            raise ArtifactValidationError(f"{round_id} report identity mismatch")
        if not isinstance(report.get("valid"), bool) or not isinstance(
            report.get("timed_out"), bool
        ):
            raise ArtifactValidationError(f"{round_id} report status is malformed")
        score = _finite(report.get("score"), f"{round_id}.score")
        pass_rate = _finite(report.get("pass_rate"), f"{round_id}.pass_rate")
        if score < 0 or not 0 <= pass_rate <= 1:
            raise ArtifactValidationError(f"{round_id} report metrics are out of range")
        round_archive_path, round_archive_bytes, round_archive_digest = _read_file(
            round_root / "submission.tar.gz",
            f"{round_id} submission archive",
            root=root,
            max_bytes=MAX_ARCHIVE_BYTES,
        )
        private = _validate_private_artifact(
            round_root / PRIVATE_ARTIFACT,
            root=root,
            expected_score=score,
            expected_suite_id=FINAL_SUITE_ID,
            expected_manifest_sha256=frozen_digest,
            expected_case_count=frozen_count,
            expected_edge_manifest_sha256=edge_digest,
        )
        agent_reports.append(
            {
                "round_id": round_id,
                "submission_id": submission_id,
                "valid": report["valid"],
                "score": score,
                "report_path": str(report_path),
                "report_sha256": report_digest,
                "archive": {
                    "path": str(round_archive_path),
                    "sha256": round_archive_digest,
                    "size_bytes": len(round_archive_bytes),
                },
                "private_judge_artifact": private,
            }
        )

    return {
        "schema_version": 1,
        "phase": "pre",
        "evaluation_mode": evaluation_mode,
        "evaluation_policy": EVALUATION_POLICY,
        "selection_protocol_id": SELECTION_PROTOCOL_ID,
        "run_id": run_id,
        "task_id": task_id,
        "model": model,
        "runtime_seconds": runtime,
        "snapshot_elapsed_seconds": elapsed,
        "active_run_hidden_evaluations": expected_auto_count,
        "auto_rounds": agent_reports if expected_auto_count else [],
        "manual_rounds": agent_reports if not expected_auto_count else [],
        "final_archive": archive,
    }


def _validate_private_artifact(
    path: Path,
    *,
    root: Path,
    expected_score: float,
    expected_suite_id: str,
    expected_manifest_sha256: str | None = None,
    expected_case_count: int | None = None,
    expected_edge_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    artifact_path, artifact_bytes, digest = _read_file(
        path,
        "private Judge artifact",
        root=root,
        max_bytes=MAX_ARCHIVE_BYTES,
    )
    if artifact_path.stat().st_mode & 0o077:
        raise ArtifactValidationError("private Judge artifact must be mode 0600")
    try:
        with tarfile.open(fileobj=io.BytesIO(artifact_bytes), mode="r:gz") as bundle:
            members = bundle.getmembers()
            if (
                len(members) != 1
                or members[0].name != PRIVATE_MEMBER
                or not members[0].isfile()
                or members[0].size <= 0
                or members[0].size > MAX_JSON_BYTES
            ):
                raise ArtifactValidationError(
                    f"private artifact must contain exactly one {PRIVATE_MEMBER}"
                )
            stream = bundle.extractfile(members[0])
            if stream is None:
                raise ArtifactValidationError("private artifact member is unreadable")
            detail_bytes = stream.read(MAX_JSON_BYTES + 1)
    except ArtifactValidationError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ArtifactValidationError("private artifact is not a readable tar.gz") from exc
    if len(detail_bytes) != members[0].size:
        raise ArtifactValidationError("private artifact member is truncated")
    try:
        detail = json.loads(
            detail_bytes.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("private Judge result is not strict JSON") from exc
    if not isinstance(detail, dict):
        raise ArtifactValidationError("private Judge result must be an object")
    private_score = _finite(detail.get("score_tflops"), "private.score_tflops")
    if not math.isclose(private_score, expected_score, rel_tol=1e-9, abs_tol=1e-9):
        raise ArtifactValidationError(
            "private score_tflops differs from the structured raw score"
        )
    if (detail.get("evaluation_policy") != EVALUATION_POLICY
            or detail.get("selection_protocol_id") != SELECTION_PROTOCOL_ID):
        raise ArtifactValidationError("private Judge selection policy differs")
    suite_id = detail.get("suite_id")
    if suite_id != expected_suite_id:
        raise ArtifactValidationError(
            f"private suite_id must be {expected_suite_id!r}, got {suite_id!r}"
        )
    manifest_sha256 = detail.get("manifest_sha256")
    if expected_manifest_sha256 is not None:
        expected_manifest_sha256 = expected_manifest_sha256.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
            raise ArtifactValidationError("expected manifest SHA-256 is malformed")
        if manifest_sha256 != expected_manifest_sha256:
            raise ArtifactValidationError(
                "private manifest_sha256 differs from the built final manifest"
            )
    case_count = detail.get("case_count")
    if expected_case_count is not None:
        if case_count != expected_case_count:
            raise ArtifactValidationError(
                "private case_count differs from the built final manifest"
            )
    edge_manifest_sha256 = detail.get("edge_manifest_sha256")
    if private_score > 0.0:
        if (not isinstance(edge_manifest_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", edge_manifest_sha256)
                or detail.get("edge_cases_total") != float(EDGE_CASE_COUNT)
                or detail.get("edge_cases_failed") != 0.0):
            raise ArtifactValidationError(
                "a scored private result must record the complete passing edge suite"
            )
    if expected_edge_manifest_sha256 is not None and edge_manifest_sha256 is not None:
        if edge_manifest_sha256 != expected_edge_manifest_sha256:
            raise ArtifactValidationError(
                "private edge_manifest_sha256 differs from the built edge manifest"
            )
    return {
        "path": str(artifact_path),
        "sha256": digest,
        "size_bytes": len(artifact_bytes),
        "member": PRIVATE_MEMBER,
        "member_sha256": hashlib.sha256(detail_bytes).hexdigest(),
        "suite_id": suite_id,
        "manifest_sha256": manifest_sha256,
        "case_count": case_count,
        "edge_manifest_sha256": edge_manifest_sha256,
    }


def validate_final_eval(
    task_root: Path,
    final_task_root: Path,
    *,
    final_eval_id: str,
    task_id: str,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
    expected_case_count: int,
    evaluation_mode: str = EVALUATION_MODE,
) -> dict[str, Any]:
    root = _directory(task_root, "task root")
    if evaluation_mode not in EVALUATION_MODES:
        raise ArtifactValidationError(f"unknown evaluation mode: {evaluation_mode}")
    final_root = _directory(final_task_root, "independent final task root")
    expected_digest = expected_archive_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise ArtifactValidationError("expected archive SHA-256 is malformed")

    final_archive, final_bytes, current_digest = _read_file(
        root / "final_archive.tar.gz",
        "final archive",
        root=root,
        max_bytes=MAX_ARCHIVE_BYTES,
    )
    if current_digest != expected_digest:
        raise ArtifactValidationError("final archive changed after pre-validation")

    submissions = _directory(final_root / "submissions", "final submissions")
    names = sorted(path.name for path in submissions.iterdir())
    if names != ["manual-1"]:
        raise ArtifactValidationError(
            f"independent final Judge must have only manual-1, got {names}"
        )
    submission = _directory(submissions / "manual-1", "final submission")
    evaluated, evaluated_bytes, evaluated_digest = _read_file(
        submission / "submission.tar.gz",
        "evaluated archive",
        root=final_root,
        max_bytes=MAX_ARCHIVE_BYTES,
    )
    if evaluated_digest != expected_digest or evaluated_bytes != final_bytes:
        raise ArtifactValidationError(
            "independent final Judge did not use the pre-validated archive"
        )

    report, report_path, report_digest = _json_file(
        submission / "report.json", "final report", root=final_root
    )
    if report.get("task_id") != task_id:
        raise ArtifactValidationError("final report has the wrong task_id")
    if evaluation_mode == SINGLE_GPU_EVALUATION_MODE:
        _validate_final_handoff_lease(
            root,
            final_eval_id=final_eval_id,
            task_id=task_id,
            expected_archive_sha256=expected_digest,
            report_sha256=report_digest,
            submission_id=report.get("submission_id"),
            error=ArtifactValidationError,
        )
    elif report.get("submission_id") != final_eval_id:
        raise ArtifactValidationError("final report has the wrong submission_id")
    if report.get("valid") is not True or report.get("timed_out") is not False:
        raise ArtifactValidationError(
            "independent final report must be valid and not timed out"
        )
    if _finite(report.get("pass_rate"), "report.pass_rate") != 1.0:
        raise ArtifactValidationError("final report.pass_rate must be 1.0")
    score = _finite(report.get("score"), "report.score")
    if score <= 0:
        raise ArtifactValidationError("final raw TFLOPS score must be positive")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ArtifactValidationError("final report.metrics must be an object")
    score_tflops = _finite(metrics.get("score_tflops"), "metrics.score_tflops")
    if not math.isclose(score, score_tflops, rel_tol=1e-9, abs_tol=1e-9):
        raise ArtifactValidationError(
            "report.score must equal metrics.score_tflops without ratio remapping"
        )
    family_values = [
        _finite(metrics.get(name), f"metrics.{name}") for name in FAMILY_KEYS
    ]
    if any(value <= 0.0 for value in family_values):
        raise ArtifactValidationError("every family TFLOPS value must be positive")
    family_geomean = math.exp(
        math.fsum(math.log(value) for value in family_values) / len(family_values)
    )
    if not math.isclose(score, family_geomean, rel_tol=1e-9, abs_tol=1e-9):
        raise ArtifactValidationError(
            "report.score must equal the geometric mean of the three family TFLOPS"
        )
    expected_metrics = {
        "source": ADAPTER_SOURCE,
        "candidate_output_suppressed": True,
        "verifier_completed": True,
        "verifier_returncode": 0,
        "result_protocol_records": 1,
        "primary_score_name": "score_tflops",
    }
    for name, expected in expected_metrics.items():
        if metrics.get(name) != expected:
            raise ArtifactValidationError(
                f"metrics.{name} must be {expected!r}, got {metrics.get(name)!r}"
            )

    private = _validate_private_artifact(
        submission / PRIVATE_ARTIFACT,
        root=final_root,
        expected_score=score,
        expected_suite_id=FINAL_SUITE_ID,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_case_count=expected_case_count,
    )
    return {
        "schema_version": 1,
        "phase": "post",
        "evaluation_mode": evaluation_mode,
        "evaluation_policy": EVALUATION_POLICY,
        "selection_protocol_id": SELECTION_PROTOCOL_ID,
        "independent_final_evaluations": 1,
        "task_id": task_id,
        "final_eval_id": final_eval_id,
        "final_archive": {
            "path": str(final_archive),
            "sha256": current_digest,
            "size_bytes": len(final_bytes),
        },
        "evaluated_archive": {
            "path": str(evaluated),
            "sha256": evaluated_digest,
            "size_bytes": len(evaluated_bytes),
        },
        "final_report": {
            "path": str(report_path),
            "sha256": report_digest,
            "valid": True,
            "pass_rate": 1.0,
            "score_tflops": score,
        },
        "private_judge_artifact": private,
        "validated_report": report,
    }


def stage_experiment_manifest(task_root: Path, source: Path, expected_sha256: str) -> dict[str, Any]:
    """Install the launcher's exact run-level manifest inside the validator root."""
    root = _directory(task_root, "task root")
    source = _absolute(source)
    source_root = _directory(source.parent, "experiment source directory")
    _, payload, digest = _read_file(
        source, "experiment source manifest", root=source_root, max_bytes=MAX_JSON_BYTES
    )
    expected = expected_sha256.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or digest != expected:
        raise ArtifactValidationError("experiment source manifest differs from the launcher digest")
    destination = root / "experiment-manifest.json"
    # Exclusive creation rejects existing files and symlinks, so staging cannot
    # overwrite another attestation or redirect the write outside the task root.
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _, installed, installed_digest = _read_file(
        destination, "task experiment manifest", root=root, max_bytes=MAX_JSON_BYTES
    )
    if installed != payload or installed_digest != expected:
        raise ArtifactValidationError("task experiment manifest copy differs")
    return {"schema_version": 1, "phase": "stage", "path": str(destination), "sha256": digest}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    stage = phases.add_parser("stage")
    stage.add_argument("--task-root", required=True, type=Path)
    stage.add_argument("--source-manifest", required=True, type=Path)
    stage.add_argument("--expected-sha256", required=True)
    pre = phases.add_parser("pre")
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
    post = phases.add_parser("post")
    post.add_argument("--task-root", required=True, type=Path)
    post.add_argument("--final-task-root", required=True, type=Path)
    post.add_argument("--final-eval-id", required=True)
    post.add_argument("--task-id", required=True)
    post.add_argument("--expected-archive-sha256", required=True)
    post.add_argument("--expected-manifest-sha256", required=True)
    post.add_argument("--expected-case-count", required=True, type=int)
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
        "expected_manifest_sha256": args.expected_manifest_sha256,
        "expected_case_count": args.expected_case_count,
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
    args = _parser().parse_args(argv)
    try:
        if args.phase == "stage":
            result = stage_experiment_manifest(args.task_root, args.source_manifest, args.expected_sha256)
        elif args.phase == "pre":
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
                    expected_manifest_sha256=args.expected_manifest_sha256,
                    expected_case_count=args.expected_case_count,
                    evaluation_mode=args.evaluation_mode,
                )
            except Exception as exc:  # noqa: BLE001 - every rejection is recorded, none is accepted
                record = invalid_final_record(args, exc)
                print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
                print(f"formal artifact validation failed: {record['reason']}", file=sys.stderr)
                return INVALID_FINAL_EXIT_CODE
    except (ArtifactValidationError, OSError) as exc:
        print(f"formal artifact validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
