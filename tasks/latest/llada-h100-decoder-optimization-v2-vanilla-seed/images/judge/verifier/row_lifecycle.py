"""Fail-closed row-atomic final evaluation lifecycle.

Only ``InfrastructureFailure`` is retryable, and then at most once. Candidate
timeouts, crashes, signals, OOM, invalid outputs, and quality failures must be
raised as ``CandidateFailure`` (or any other exception) and are never retried.
Every attempt remains an append-only root-controller artifact. A completed row
is committed atomically and binds the complete evaluation identity tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Callable, Mapping, Sequence


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MEMORY_FIELDS = (
    "torch_peak_allocated_bytes",
    "torch_peak_reserved_bytes",
    "nvml_peak_device_used_bytes",
)


class RowLifecycleError(RuntimeError):
    pass


class InfrastructureFailure(RowLifecycleError):
    """A trusted controller-classified failure outside candidate control."""


class CandidateFailure(RowLifecycleError):
    """A timeout/crash/OOM/invalid output or other candidate-controlled failure."""


def evaluation_checkpoint_root(
    root: Path, metadata: Mapping[str, Any], *, expected_uid: int = 0,
) -> Path:
    """Keep each interactive archive's deadline and completed rows independent.

    Terminal evaluation retains its existing resumable root. Interactive rounds
    use the original round and host submission ID below the root-only mount;
    even submitting the same archive twice must start a fresh measurement.
    """
    def safe_directory(path: Path, *, create: bool = False) -> None:
        if create:
            path.mkdir(mode=0o700, exist_ok=True)
        observed = path.lstat()
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != expected_uid
            or stat.S_IMODE(observed.st_mode) != 0o700
        ):
            raise RowLifecycleError("checkpoint directory ownership/type is invalid")

    safe_directory(root)
    round_id = metadata.get("round_id", "")
    if not isinstance(round_id, str):
        raise RowLifecycleError("checkpoint round identity is invalid")
    if not round_id.startswith(("agent-", "auto-")):
        return root
    if re.fullmatch(r"(?:agent|auto)-[1-9][0-9]*", round_id) is None:
        raise RowLifecycleError("interactive checkpoint round identity is invalid")
    submission_id = metadata.get("submission_id")
    if (
        not isinstance(submission_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", submission_id) is None
    ):
        raise RowLifecycleError("interactive checkpoint submission identity is invalid")
    round_root = root / round_id
    safe_directory(round_root, create=True)
    selected = round_root / submission_id
    safe_directory(selected, create=True)
    return selected


@dataclass(frozen=True, slots=True)
class EvaluationIdentity:
    run_id: str
    archive_sha256: str
    work_image_id: str
    judge_image_id: str
    model_validation_sha256: str
    runtime_entry_sha256: str
    source_digest: str
    workload_policy_sha256: str
    memory_policy_sha256: str
    quality_controls_sha256: str
    runtime_provenance_sha256: str
    sforge_runtime_sha256: str
    sforge_runtime_file_set_sha256: str
    verify_decoder_sha256: str
    candidate_worker_sha256: str
    cohort_sha256: str
    quality_sha256: str
    source_admission_sha256: str
    hidden_manifest_sha256: str
    row_order_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationIdentity":
        try:
            identity = cls(**{field: value[field] for field in cls.__dataclass_fields__})
        except (KeyError, TypeError) as exc:
            raise RowLifecycleError("completed-row identity tuple is incomplete") from exc
        if not isinstance(identity.run_id, str) or not identity.run_id:
            raise RowLifecycleError("completed-row run identity is invalid")
        for field in (
            "archive_sha256",
            "model_validation_sha256",
            "runtime_entry_sha256",
            "source_digest",
            "workload_policy_sha256",
            "memory_policy_sha256",
            "quality_controls_sha256",
            "runtime_provenance_sha256",
            "sforge_runtime_sha256",
            "sforge_runtime_file_set_sha256",
            "verify_decoder_sha256",
            "candidate_worker_sha256",
            "cohort_sha256",
            "quality_sha256",
            "source_admission_sha256",
            "hidden_manifest_sha256",
            "row_order_sha256",
        ):
            if _SHA256.fullmatch(str(getattr(identity, field))) is None:
                raise RowLifecycleError(f"completed-row {field} is invalid")
        for field in ("work_image_id", "judge_image_id"):
            if _IMAGE_ID.fullmatch(str(getattr(identity, field))) is None:
                raise RowLifecycleError(f"completed-row {field} is invalid")
        return identity

    def as_dict(self) -> dict[str, str]:
        return {
            field: str(getattr(self, field)) for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class PersistentDeadline:
    identity: EvaluationIdentity
    started_at_unix_seconds: float
    outer_deadline_unix_seconds: float
    outer_timeout_seconds: float
    commitment_sha256: str

    def remaining_seconds(self) -> float:
        return self.outer_deadline_unix_seconds - time.time()

    def report_summary(self) -> dict[str, float | str]:
        return {
            "schema_version": 1,
            "identity": self.identity.as_dict(),
            "started_at_unix_seconds": self.started_at_unix_seconds,
            "outer_deadline_unix_seconds": self.outer_deadline_unix_seconds,
            "outer_timeout_seconds": self.outer_timeout_seconds,
            "commitment_sha256": self.commitment_sha256,
        }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _safe_row_name(row_id: str) -> str:
    if re.fullmatch(r"[a-z0-9_]+", row_id) is None:
        raise RowLifecycleError("row ID is unsafe")
    return row_id


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    data = _canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _read_owned(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RowLifecycleError("row artifact is missing or unsafe")
    stat = path.stat()
    if stat.st_uid != os.geteuid() or stat.st_mode & 0o077:
        raise RowLifecycleError("row artifact ownership or mode is invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RowLifecycleError("row artifact is not an object")
    return value


def _commitment_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded report view; full records stay in root-owned storage."""
    return {key: item for key, item in value.items() if key != "records"}


def _validate_row_memory_sentinel(value: object) -> None:
    if not isinstance(value, Mapping):
        raise RowLifecycleError("row memory sentinel is missing")
    start = value.get("start")
    end = value.get("end")
    repeat = value.get("repeat")
    deltas = value.get("deltas_bytes")
    repeat_deltas = value.get("repeat_deltas_bytes")
    bound = value.get("max_material_growth_bytes")
    repeat_bound = value.get("max_repeat_growth_bytes")
    if (
        value.get("schema_version") != 2
        or not isinstance(value.get("stable"), bool)
        or not isinstance(value.get("repeat_stable"), bool)
        or not isinstance(value.get("case"), str)
        or not value["case"]
        or not isinstance(start, Mapping)
        or not isinstance(end, Mapping)
        or not isinstance(repeat, Mapping)
        or not isinstance(deltas, Mapping)
        or not isinstance(repeat_deltas, Mapping)
        or set(start) != set(_MEMORY_FIELDS)
        or set(end) != set(_MEMORY_FIELDS)
        or set(repeat) != set(_MEMORY_FIELDS)
        or set(deltas) != set(_MEMORY_FIELDS)
        or set(repeat_deltas) != set(_MEMORY_FIELDS)
        or isinstance(bound, bool)
        or not isinstance(bound, int)
        or bound <= 0
        or isinstance(repeat_bound, bool)
        or not isinstance(repeat_bound, int)
        or repeat_bound != 16 * 1024 * 1024
        or any(
            _SHA256.fullmatch(str(value.get(field, ""))) is None
            for field in (
                "start_output_sha256",
                "end_output_sha256",
                "repeat_output_sha256",
            )
        )
    ):
        raise RowLifecycleError("row memory sentinel schema is invalid")
    expected_deltas: dict[str, int] = {}
    expected_repeat_deltas: dict[str, int] = {}
    for field in _MEMORY_FIELDS:
        before = start[field]
        after = end[field]
        repeated = repeat[field]
        delta = deltas[field]
        repeat_delta = repeat_deltas[field]
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (before, after, repeated)
        ) or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (delta, repeat_delta)
        ):
            raise RowLifecycleError("row memory sentinel metrics are invalid")
        expected_deltas[field] = after - before
        expected_repeat_deltas[field] = repeated - after
    observed = max(0, *expected_deltas.values())
    observed_repeat = max(0, *expected_repeat_deltas.values())
    if (
        dict(deltas) != expected_deltas
        or dict(repeat_deltas) != expected_repeat_deltas
        or value.get("observed_max_positive_growth_bytes") != observed
        or value.get("observed_max_positive_repeat_growth_bytes")
        != observed_repeat
        or value.get("repeat_stable") is not (observed_repeat <= repeat_bound)
        or value.get("stable")
        is not (
            observed <= bound
            and observed_repeat <= repeat_bound
            and value.get("start_output_sha256")
            == value.get("end_output_sha256")
            == value.get("repeat_output_sha256")
        )
    ):
        raise RowLifecycleError("row memory sentinel evidence is internally inconsistent")


def _validate_deadline(
    value: Mapping[str, Any],
    *,
    identity: EvaluationIdentity,
    outer_timeout_seconds: float,
) -> PersistentDeadline:
    frozen = dict(value)
    commitment = frozen.pop("commitment_sha256", None)
    if (
        value.get("schema_version") != 1
        or value.get("identity") != identity.as_dict()
        or not math.isfinite(outer_timeout_seconds)
        or outer_timeout_seconds <= 0
        or value.get("outer_timeout_seconds") != outer_timeout_seconds
        or not isinstance(value.get("started_at_unix_seconds"), (int, float))
        or isinstance(value.get("started_at_unix_seconds"), bool)
        or not isinstance(value.get("outer_deadline_unix_seconds"), (int, float))
        or isinstance(value.get("outer_deadline_unix_seconds"), bool)
        or not math.isfinite(float(value["started_at_unix_seconds"]))
        or not math.isfinite(float(value["outer_deadline_unix_seconds"]))
        or not math.isclose(
            float(value["outer_deadline_unix_seconds"])
            - float(value["started_at_unix_seconds"]),
            outer_timeout_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or _SHA256.fullmatch(str(commitment)) is None
        or commitment != hashlib.sha256(_canonical_bytes(frozen)).hexdigest()
    ):
        raise RowLifecycleError("persistent outer deadline failed identity validation")
    return PersistentDeadline(
        identity=identity,
        started_at_unix_seconds=float(value["started_at_unix_seconds"]),
        outer_deadline_unix_seconds=float(value["outer_deadline_unix_seconds"]),
        outer_timeout_seconds=outer_timeout_seconds,
        commitment_sha256=str(commitment),
    )


def load_or_create_absolute_deadline(
    artifact_root: Path,
    *,
    identity: EvaluationIdentity,
    outer_timeout_seconds: float,
) -> PersistentDeadline:
    """Persist the first UTC deadline; process/container resumes cannot reset it."""
    if (
        not math.isfinite(outer_timeout_seconds)
        or outer_timeout_seconds <= 0
        or outer_timeout_seconds > 7_200
    ):
        raise RowLifecycleError("outer timeout is outside the task ceiling")
    path = artifact_root / "run-deadline.json"
    if not path.exists() and not path.is_symlink():
        started = time.time()
        payload: dict[str, Any] = {
            "schema_version": 1,
            "identity": identity.as_dict(),
            "started_at_unix_seconds": started,
            "outer_deadline_unix_seconds": started + outer_timeout_seconds,
            "outer_timeout_seconds": outer_timeout_seconds,
        }
        payload["commitment_sha256"] = hashlib.sha256(
            _canonical_bytes(payload)
        ).hexdigest()
        try:
            _write_once(path, payload)
        except FileExistsError:
            # A concurrent trusted controller may have won O_EXCL. Its bytes
            # still undergo the exact identity/deadline validation below.
            pass
    deadline = _validate_deadline(
        _read_owned(path),
        identity=identity,
        outer_timeout_seconds=outer_timeout_seconds,
    )
    if deadline.remaining_seconds() <= 0:
        raise RowLifecycleError("persistent outer deadline has expired")
    return deadline


def _validate_commitment(
    value: Mapping[str, Any],
    *,
    identity: EvaluationIdentity,
    row_id: str,
    row_index: int,
    deadline_commitment_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        value.get("schema_version") != 2
        or value.get("status") != "COMPLETED"
        or value.get("identity") != identity.as_dict()
        or value.get("row_id") != row_id
        or value.get("row_index") != row_index
        or value.get("attempt") not in {0, 1}
        or value.get("deadline_commitment_sha256")
        != deadline_commitment_sha256
    ):
        raise RowLifecycleError("completed-row commitment failed identity revalidation")
    records = value.get("records")
    if not isinstance(records, list) or not records or any(
        not isinstance(record, dict) for record in records
    ):
        raise RowLifecycleError("completed-row record payload is missing or malformed")
    if value.get("record_count") != len(records):
        raise RowLifecycleError("completed-row record count mismatch")
    _validate_row_memory_sentinel(value.get("row_memory_sentinel"))
    case_hashes = [str(record.get("case_sha256", "")) for record in records]
    if (
        any(_SHA256.fullmatch(case_hash) is None for case_hash in case_hashes)
        or len(case_hashes) != len(set(case_hashes))
    ):
        raise RowLifecycleError("completed-row case commitments are invalid")
    if value.get("case_set_sha256") != hashlib.sha256(
        ("\n".join(sorted(case_hashes)) + "\n").encode()
    ).hexdigest():
        raise RowLifecycleError("completed-row case-set hash mismatch")
    if value.get("records_sha256") != hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest():
        raise RowLifecycleError("completed-row records hash mismatch")
    frozen = dict(value)
    commitment = frozen.pop("commitment_sha256", None)
    if commitment != hashlib.sha256(_canonical_bytes(frozen)).hexdigest():
        raise RowLifecycleError("completed-row commitment hash mismatch")
    return records, _commitment_summary(value)


def load_completed_row(
    artifact_root: Path,
    *,
    identity: EvaluationIdentity,
    row_id: str,
    row_index: int,
    deadline_commitment_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    path = artifact_root / "completed_rows" / f"{_safe_row_name(row_id)}.json"
    if not path.exists() and not path.is_symlink():
        return None
    return _validate_commitment(
        _read_owned(path),
        identity=identity,
        row_id=row_id,
        row_index=row_index,
        deadline_commitment_sha256=deadline_commitment_sha256,
    )


def record_attempt(
    artifact_root: Path,
    *,
    identity: EvaluationIdentity,
    row_id: str,
    row_index: int,
    attempt: int,
    status: str,
    error_class: str | None,
    elapsed_seconds: float,
    deadline_commitment_sha256: str,
) -> Path:
    if status not in {"PASS", "FAILED_INFRASTRUCTURE", "FAILED_CANDIDATE"}:
        raise RowLifecycleError("row attempt status is invalid")
    if attempt not in {0, 1} or row_index < 0:
        raise RowLifecycleError("row attempt position is invalid")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise RowLifecycleError("row attempt elapsed time is invalid")
    path = (
        artifact_root
        / "row_attempts"
        / _safe_row_name(row_id)
        / f"attempt-{attempt}-outcome.json"
    )
    _write_once(
        path,
        {
            "schema_version": 3,
            "identity": identity.as_dict(),
            "row_id": row_id,
            "row_index": row_index,
            "attempt": attempt,
            "status": status,
            "error_class": error_class,
            "elapsed_seconds": elapsed_seconds,
            "deadline_commitment_sha256": deadline_commitment_sha256,
        },
    )
    return path


def record_attempt_started(
    artifact_root: Path,
    *,
    identity: EvaluationIdentity,
    row_id: str,
    row_index: int,
    attempt: int,
    deadline_commitment_sha256: str,
) -> Path:
    if attempt not in {0, 1} or row_index < 0:
        raise RowLifecycleError("row attempt position is invalid")
    path = (
        artifact_root
        / "row_attempts"
        / _safe_row_name(row_id)
        / f"attempt-{attempt}-started.json"
    )
    _write_once(
        path,
        {
            "schema_version": 3,
            "identity": identity.as_dict(),
            "row_id": row_id,
            "row_index": row_index,
            "attempt": attempt,
            "status": "STARTED",
            "started_at_unix_seconds": time.time(),
            "deadline_commitment_sha256": deadline_commitment_sha256,
        },
    )
    return path


def commit_completed_row(
    artifact_root: Path,
    *,
    identity: EvaluationIdentity,
    row_id: str,
    row_index: int,
    attempt: int,
    records: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
    deadline_commitment_sha256: str,
    row_memory_sentinel: Mapping[str, Any],
) -> dict[str, Any]:
    if not records or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise RowLifecycleError("completed row payload is empty or untimed")
    _validate_row_memory_sentinel(row_memory_sentinel)
    case_hashes = [str(record.get("case_sha256", "")) for record in records]
    if any(_SHA256.fullmatch(value) is None for value in case_hashes):
        raise RowLifecycleError("completed row has invalid case commitments")
    if len(case_hashes) != len(set(case_hashes)):
        raise RowLifecycleError("completed row repeats a case commitment")
    record_bytes = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    payload: dict[str, Any] = {
        "schema_version": 2,
        "status": "COMPLETED",
        "identity": identity.as_dict(),
        "row_id": _safe_row_name(row_id),
        "row_index": row_index,
        "attempt": attempt,
        "record_count": len(records),
        "case_set_sha256": hashlib.sha256(
            ("\n".join(sorted(case_hashes)) + "\n").encode()
        ).hexdigest(),
        "records_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "records": [dict(record) for record in records],
        "elapsed_seconds": elapsed_seconds,
        "deadline_commitment_sha256": deadline_commitment_sha256,
        "row_memory_sentinel": dict(row_memory_sentinel),
    }
    payload["commitment_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    path = artifact_root / "completed_rows" / f"{row_id}.json"
    if path.exists() or path.is_symlink():
        existing = _read_owned(path)
        if existing != payload:
            raise RowLifecycleError("existing completed-row commitment identity differs")
        return existing
    _write_once(path, payload)
    return payload


def run_row_with_retry(
    *,
    artifact_root: Path,
    identity: EvaluationIdentity,
    row_id: str,
    row_index: int,
    run_attempt: Callable[
        [int], tuple[list[dict[str, Any]], float, Mapping[str, Any]]
    ],
    deadline: PersistentDeadline,
    max_infrastructure_retries: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_infrastructure_retries != 1:
        raise RowLifecycleError("final policy permits exactly one infrastructure retry")
    completed = load_completed_row(
        artifact_root,
        identity=identity,
        row_id=row_id,
        row_index=row_index,
        deadline_commitment_sha256=deadline.commitment_sha256,
    )
    attempts_root = artifact_root / "row_attempts" / _safe_row_name(row_id)
    existing_attempts: dict[int, dict[str, Any]] = {}
    for attempt in (0, 1):
        started_path = attempts_root / f"attempt-{attempt}-started.json"
        outcome_path = attempts_root / f"attempt-{attempt}-outcome.json"
        started = (
            _read_owned(started_path)
            if started_path.exists() or started_path.is_symlink()
            else None
        )
        outcome = (
            _read_owned(outcome_path)
            if outcome_path.exists() or outcome_path.is_symlink()
            else None
        )
        if outcome is not None and started is None:
            raise RowLifecycleError("row attempt outcome has no STARTED commitment")
        if started is not None:
            if (
                started.get("schema_version") != 3
                or started.get("status") != "STARTED"
                or started.get("identity") != identity.as_dict()
                or started.get("row_id") != row_id
                or started.get("row_index") != row_index
                or started.get("attempt") != attempt
                or started.get("deadline_commitment_sha256")
                != deadline.commitment_sha256
            ):
                raise RowLifecycleError("prior row attempt identity differs")
            if outcome is None:
                started_at = started.get("started_at_unix_seconds")
                if (
                    isinstance(started_at, bool)
                    or not isinstance(started_at, (int, float))
                    or not math.isfinite(float(started_at))
                ):
                    raise RowLifecycleError("orphan row STARTED time is invalid")
                if completed is not None and completed[1].get("attempt") == attempt:
                    record_attempt(
                        artifact_root,
                        identity=identity,
                        row_id=row_id,
                        row_index=row_index,
                        attempt=attempt,
                        status="PASS",
                        error_class=None,
                        elapsed_seconds=float(completed[1]["elapsed_seconds"]),
                        deadline_commitment_sha256=deadline.commitment_sha256,
                    )
                    outcome = _read_owned(outcome_path)
                else:
                    record_attempt(
                        artifact_root,
                        identity=identity,
                        row_id=row_id,
                        row_index=row_index,
                        attempt=attempt,
                        status="FAILED_INFRASTRUCTURE",
                        error_class="ControllerInterrupted",
                        elapsed_seconds=max(0.0, time.time() - float(started_at)),
                        deadline_commitment_sha256=deadline.commitment_sha256,
                    )
                    outcome = _read_owned(outcome_path)
            assert outcome is not None
            elapsed = outcome.get("elapsed_seconds")
            status = outcome.get("status")
            error_class = outcome.get("error_class")
            if (
                outcome.get("schema_version") != 3
                or outcome.get("identity") != identity.as_dict()
                or outcome.get("row_id") != row_id
                or outcome.get("row_index") != row_index
                or outcome.get("attempt") != attempt
                or outcome.get("deadline_commitment_sha256")
                != deadline.commitment_sha256
                or status not in {"PASS", "FAILED_INFRASTRUCTURE", "FAILED_CANDIDATE"}
                or isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0
                or (status == "PASS" and error_class is not None)
                or (
                    status in {"FAILED_INFRASTRUCTURE", "FAILED_CANDIDATE"}
                    and (not isinstance(error_class, str) or not error_class)
                )
            ):
                raise RowLifecycleError("prior row attempt outcome is invalid")
            existing_attempts[attempt] = outcome
    if completed is not None:
        return completed
    if 1 in existing_attempts:
        raise RowLifecycleError("row exhausted its only infrastructure retry")
    if 0 in existing_attempts:
        prior_status = existing_attempts[0].get("status")
        if prior_status == "FAILED_CANDIDATE":
            raise RowLifecycleError("row has a prior non-retriable candidate failure")
        if prior_status != "FAILED_INFRASTRUCTURE":
            raise RowLifecycleError("row has an incomplete or invalid prior attempt")
        attempt_order = (1,)
    else:
        attempt_order = (0, 1)

    for attempt in attempt_order:
        if deadline.remaining_seconds() <= 0:
            raise RowLifecycleError("outer deadline expired before row admission")
        started = time.monotonic()
        record_attempt_started(
            artifact_root,
            identity=identity,
            row_id=row_id,
            row_index=row_index,
            attempt=attempt,
            deadline_commitment_sha256=deadline.commitment_sha256,
        )
        try:
            records, elapsed_seconds, row_memory_sentinel = run_attempt(attempt)
        except Exception as exc:
            elapsed = max(0.0, time.monotonic() - started)
            infrastructure = isinstance(exc, InfrastructureFailure)
            record_attempt(
                artifact_root,
                identity=identity,
                row_id=row_id,
                row_index=row_index,
                attempt=attempt,
                status=("FAILED_INFRASTRUCTURE" if infrastructure else "FAILED_CANDIDATE"),
                error_class=type(exc).__name__,
                elapsed_seconds=elapsed,
                deadline_commitment_sha256=deadline.commitment_sha256,
            )
            if infrastructure and attempt == 0 and deadline.remaining_seconds() > 0:
                continue
            raise
        commitment = commit_completed_row(
            artifact_root,
            identity=identity,
            row_id=row_id,
            row_index=row_index,
            attempt=attempt,
            records=records,
            elapsed_seconds=elapsed_seconds,
            deadline_commitment_sha256=deadline.commitment_sha256,
            row_memory_sentinel=row_memory_sentinel,
        )
        # Commit the complete recoverable row first. A controller death after
        # this atomic write can resume without reusing partial timing.
        record_attempt(
            artifact_root,
            identity=identity,
            row_id=row_id,
            row_index=row_index,
            attempt=attempt,
            status="PASS",
            error_class=None,
            elapsed_seconds=elapsed_seconds,
            deadline_commitment_sha256=deadline.commitment_sha256,
        )
        return records, _commitment_summary(commitment)
    raise AssertionError("bounded row retry loop fell through")


def validate_completed_rows(
    artifact_root: Path,
    *,
    identity: EvaluationIdentity,
    row_order: Sequence[str],
    deadline_commitment_sha256: str,
) -> list[dict[str, Any]]:
    commits = []
    for row_index, row_id in enumerate(row_order):
        completed = load_completed_row(
            artifact_root,
            identity=identity,
            row_id=row_id,
            row_index=row_index,
            deadline_commitment_sha256=deadline_commitment_sha256,
        )
        if completed is None:
            raise RowLifecycleError("completed-row commitment is missing")
        _records, summary = completed
        commits.append(summary)
    return commits
