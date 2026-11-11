#!/usr/bin/env python3
"""Fail-closed validation for a formal Qwen SWE-192 MLSys Forge evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tarfile
import tempfile
from typing import Any


TASK_ID = "concurrent_agent_serving_optimizer_qwen36_swe192_v24"
TASK_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "tasks" / f"{TASK_ID}.json"
)
PRIVATE_MEMBER = "kvbench-private-evidence.json"
# Contract revision v2.5 states the backend requirement as a capability
# predicate instead of an "NVIDIA H100 NVL" name whitelist.  Evidence recorded
# under the retained v1 revision describes the same serving system and stays
# acceptable; see judge/run_judge.py for the full rationale.
EXPECTED_BACKEND_SERVING_CONTRACT_SHA256 = (
    "1cf6d2e6ff7af8448d6b7735590621d083d6a5071252bc60b40c9756ffe328c4"
)
LEGACY_BACKEND_SERVING_CONTRACT_SHA256 = (
    "44e4a3408b298a5fdc53d7bb84f0739a22f047da1974212754a98b5755777ba8"
)
PREVIOUS_BACKEND_SERVING_CONTRACT_SHA256 = (
    "18d27d97c736f90e1d74822e80ce1f212685e26627b31a08e47569a1457f52c6"
)
ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256 = frozenset(
    {
        EXPECTED_BACKEND_SERVING_CONTRACT_SHA256,
        LEGACY_BACKEND_SERVING_CONTRACT_SHA256,
        PREVIOUS_BACKEND_SERVING_CONTRACT_SHA256,
    }
)
LEGACY_MODEL_PROVENANCE = {
    "profile_id": "qwen3.6-27b-6a9e13bd-h100-nvl-tp2-kv30g-v2",
    "profile_sha256": (
        "7289ed315324b9b5165d4d680efb93004c59e0d34f0f98ef9d85406ed5a9153a"
    ),
    "model_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    "runtime_image_id": (
        "sha256:1d8369b9e3c91aa8cba5c7aeeef26e3c4afd455e5d2ce9aebe6f30f6dd9344bd"
    ),
    "tensor_parallel_size": 2,
    "kv_cache_dtype": "fp8",
}
EXPECTED_NONCE_SHA256 = (
    "c525ae31a2bb1d3ed0232c345f9fc978fd52574d415aaa2777bead17f2fac20a"
)
EXPECTED_BASELINE_SHA256: str | None = None
DIRECT_RATE: float | None = None
HIDDEN_BASELINE_IDENTITY: dict[str, Any] | None = None
DISPLAY_RESCALE_LOWER: float | None = None
DISPLAY_RESCALE_UPPER: float | None = None
HARD_GATES = frozenset(
    {
        "protocol_identity_fixed_window_lifecycle",
        "backend_one_to_one_audit",
        "paired_request_success",
        "paired_valid_step_ratio",
        "candidate_p99_gateway_wait",
        "paired_backend_work",
        "throughput_accounting",
    }
)
TOP_LEVEL_FIELDS = frozenset(
    {
        "task_id",
        "submission_id",
        "total_tests",
        "passed",
        "failed",
        "errors",
        "pass_rate",
        "score",
        "score_0_100",
        "timed_out",
        "runtime_seconds",
        "test_details",
        "valid",
        "summary",
        "details",
        "metrics",
        "submitted_at",
    }
)
STRUCTURED_KEYS = (
    "valid",
    "score",
    "pass_rate",
    "total_tests",
    "passed",
    "failed",
    "errors",
    "summary",
    "details",
    "metrics",
)
EXIT_CODES = {
    "candidate_valid": 0,
    "candidate_invalid": 0,
    "measurement_lock_contention": 75,
    "infrastructure_error": 20,
    "cancelled": 25,
    "timeout": 21,
    "malformed": 22,
    "artifact_missing": 23,
    "evaluation_process_error": 24,
}


class ValidationError(RuntimeError):
    """The report or its private evidence violates the frozen contract."""


HIDDEN_BASELINE_IDENTITY_FIELDS = frozenset(
    {
        "file_sha256",
        "valid_steps_per_min",
        "backend_serving_contract_sha256",
        "workload_nonce_sha256",
    }
)


def _canonical_hidden_baseline_identity(
    value: object, label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != HIDDEN_BASELINE_IDENTITY_FIELDS:
        raise ValidationError(f"{label} schema is not exact")
    file_sha256 = value.get("file_sha256")
    rate = value.get("valid_steps_per_min")
    backend_sha256 = value.get("backend_serving_contract_sha256")
    nonce_sha256 = value.get("workload_nonce_sha256")
    if not isinstance(file_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", file_sha256
    ) is None:
        raise ValidationError(f"{label} file_sha256 is invalid")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(float(rate))
        or float(rate) <= 0.0
    ):
        raise ValidationError(f"{label} valid_steps_per_min is invalid")
    if backend_sha256 not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256:
        raise ValidationError(f"{label} backend serving contract changed")
    if nonce_sha256 != EXPECTED_NONCE_SHA256:
        raise ValidationError(f"{label} workload nonce changed")
    return {
        "file_sha256": file_sha256,
        "valid_steps_per_min": float(rate),
        "backend_serving_contract_sha256": backend_sha256,
        "workload_nonce_sha256": nonce_sha256,
    }


def baseline_contract_sha256() -> str:
    """The contract revision the selected Direct anchor was measured under."""

    if HIDDEN_BASELINE_IDENTITY is not None:
        return str(HIDDEN_BASELINE_IDENTITY["backend_serving_contract_sha256"])
    return EXPECTED_BACKEND_SERVING_CONTRACT_SHA256


def hidden_baseline_identity() -> dict[str, Any]:
    if HIDDEN_BASELINE_IDENTITY is not None:
        return dict(HIDDEN_BASELINE_IDENTITY)
    return {
        "file_sha256": None,
        "valid_steps_per_min": None,
        "backend_serving_contract_sha256": None,
        "workload_nonce_sha256": None,
    }


def load_expected_preflight(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValidationError("expected preflight is unavailable or unsafe")
    try:
        raw = path.read_bytes()
        payload = _strict_json_loads(raw.decode("utf-8"), "expected preflight")
    except UnicodeDecodeError as exc:
        raise ValidationError("expected preflight is not valid UTF-8 JSON") from exc
    if (
        not isinstance(payload, dict)
        or isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 1
        or payload.get("status") != "passed"
    ):
        raise ValidationError("expected preflight is not a passed schema-v1 report")
    return _canonical_hidden_baseline_identity(
        {
            "file_sha256": payload.get("baseline_file_sha256"),
            "valid_steps_per_min": payload.get("baseline_valid_steps_per_min"),
            "backend_serving_contract_sha256": payload.get(
                "backend_serving_contract_sha256"
            ),
            "workload_nonce_sha256": payload.get("nonce_commitment_sha256"),
        },
        "expected preflight hidden_baseline_identity",
    )


def _validate_baseline_provenance(provenance: object) -> str:
    if not isinstance(provenance, dict):
        raise ValidationError("selected hidden baseline provenance is invalid")
    projected = {
        key: provenance.get(key) for key in LEGACY_MODEL_PROVENANCE
    }
    if (
        not isinstance(projected["profile_id"], str)
        or not projected["profile_id"]
        or re.fullmatch(r"[0-9a-f]{64}", str(projected["profile_sha256"]))
        is None
        or projected["model_revision"]
        != LEGACY_MODEL_PROVENANCE["model_revision"]
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(projected["runtime_image_id"])
        )
        is None
        or projected["tensor_parallel_size"] != 2
        or projected["kv_cache_dtype"] != "fp8"
    ):
        raise ValidationError("selected hidden baseline provenance is invalid")
    digest = provenance.get("backend_serving_contract_sha256")
    if digest is None:
        if projected != LEGACY_MODEL_PROVENANCE:
            raise ValidationError(
                "selected hidden legacy baseline provenance is unknown"
            )
        digest = LEGACY_BACKEND_SERVING_CONTRACT_SHA256
    elif digest not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256:
        raise ValidationError("selected hidden baseline serving contract changed")
    # Report the revision the anchor was measured under, not the one shipped.
    return digest


def configure_baseline_contract(
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the selected Direct evidence instead of compiling a host path/rate."""

    global EXPECTED_BASELINE_SHA256, DIRECT_RATE, HIDDEN_BASELINE_IDENTITY
    raw_path = os.environ.get("KVBENCH_HIDDEN_BASELINE_PATH", "").strip()
    if not raw_path:
        raise ValidationError("KVBENCH_HIDDEN_BASELINE_PATH is required")
    path = Path(raw_path)
    if (
        not path.is_absolute()
        or path != Path(os.path.normpath(raw_path))
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 128 * 1024 * 1024
    ):
        raise ValidationError("selected hidden baseline is unavailable or unsafe")
    raw = path.read_bytes()
    try:
        payload = _strict_json_loads(raw.decode("utf-8"), "selected hidden baseline")
    except UnicodeDecodeError as exc:
        raise ValidationError("selected hidden baseline is not valid UTF-8 JSON") from exc
    baseline = (payload.get("legs") or {}).get("baseline") if isinstance(payload, dict) else None
    rate = baseline.get("valid_steps_per_min") if isinstance(baseline, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("status") != "complete"
        or payload.get("suite") != "hidden"
        or payload.get("profile") != "hidden-pressure-192"
        or payload.get("workload_nonce_sha256") != EXPECTED_NONCE_SHA256
        or not isinstance(baseline, dict)
        or baseline.get("mode") != "baseline"
        or baseline.get("profile") != "hidden-pressure-192"
        or baseline.get("workload_nonce_sha256") != EXPECTED_NONCE_SHA256
        or not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not math.isfinite(float(rate))
        or float(rate) <= 0.0
    ):
        raise ValidationError("selected hidden baseline contract is invalid")
    backend_serving_contract_sha256 = _validate_baseline_provenance(
        baseline.get("model_provenance")
    )
    actual_identity = _canonical_hidden_baseline_identity(
        {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "valid_steps_per_min": float(rate),
            "backend_serving_contract_sha256": backend_serving_contract_sha256,
            "workload_nonce_sha256": payload.get("workload_nonce_sha256"),
        },
        "selected hidden baseline identity",
    )
    if expected_identity is not None and actual_identity != expected_identity:
        raise ValidationError(
            "selected hidden baseline does not match expected preflight identity"
        )
    EXPECTED_BASELINE_SHA256 = actual_identity["file_sha256"]
    DIRECT_RATE = actual_identity["valid_steps_per_min"]
    HIDDEN_BASELINE_IDENTITY = actual_identity
    return dict(actual_identity)


def configure_hidden_baseline_contract(expected_preflight: Path) -> None:
    global HIDDEN_BASELINE_IDENTITY
    expected_identity = load_expected_preflight(expected_preflight)
    # Preserve the authoritative expected identity in fail-closed diagnostics even
    # when the selected baseline snapshot no longer matches it.
    HIDDEN_BASELINE_IDENTITY = expected_identity
    configure_baseline_contract(expected_identity)


def configure_display_rescale_contract() -> None:
    """Load the display-only rescale from the task spec used by SForge."""

    global DISPLAY_RESCALE_LOWER, DISPLAY_RESCALE_UPPER
    path = TASK_SPEC_PATH
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 1024 * 1024
    ):
        raise ValidationError("task spec is unavailable or unsafe")
    try:
        payload = _strict_json_loads(path.read_text(encoding="utf-8"), "task spec")
    except UnicodeDecodeError as exc:
        raise ValidationError("task spec is not valid UTF-8 JSON") from exc
    judge = payload.get("judge") if isinstance(payload, dict) else None
    rescale = judge.get("rescale") if isinstance(judge, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("task_id") != TASK_ID
        or not isinstance(judge, dict)
        or judge.get("score_direction") != "maximize"
        or judge.get("selection") != "valid_then_score"
        or not isinstance(rescale, dict)
        or set(rescale) != {"kind", "lower", "upper"}
        or rescale.get("kind") != "linear"
    ):
        raise ValidationError("task spec display rescale contract is invalid")
    lower = rescale.get("lower")
    upper = rescale.get("upper")
    if (
        isinstance(lower, bool)
        or not isinstance(lower, (int, float))
        or not math.isfinite(float(lower))
        or isinstance(upper, bool)
        or not isinstance(upper, (int, float))
        or not math.isfinite(float(upper))
        or float(upper) <= float(lower)
    ):
        raise ValidationError("task spec display rescale bounds are invalid")
    DISPLAY_RESCALE_LOWER = float(lower)
    DISPLAY_RESCALE_UPPER = float(upper)


def display_rescale_score(raw_score: float) -> float:
    """Reproduce SForge's clipped linear display score exactly."""

    if DISPLAY_RESCALE_LOWER is None or DISPLAY_RESCALE_UPPER is None:
        raise ValidationError("task spec display rescale is not configured")
    return max(
        0.0,
        min(
            100.0,
            100.0
            * (raw_score - DISPLAY_RESCALE_LOWER)
            / (DISPLAY_RESCALE_UPPER - DISPLAY_RESCALE_LOWER),
        ),
    )


def direct_rate() -> float:
    if DIRECT_RATE is None:
        raise ValidationError("hidden baseline contract is not configured")
    return DIRECT_RATE


def baseline_sha256() -> str:
    if EXPECTED_BASELINE_SHA256 is None:
        raise ValidationError("hidden baseline contract is not configured")
    return EXPECTED_BASELINE_SHA256


def _strict_json_loads(raw: str, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValidationError(f"{label} contains non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{label} is not valid JSON") from exc


def _canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValidationError("public result cannot be canonically encoded") from exc
    return hashlib.sha256(encoded).hexdigest()


def _terminal_utc_pair(started: Any, completed: Any, label: str) -> None:
    if (
        not isinstance(started, str)
        or not started.endswith("Z")
        or not isinstance(completed, str)
        or not completed.endswith("Z")
    ):
        raise ValidationError(f"{label} timestamps are not terminal UTC values")
    try:
        start_value = datetime.fromisoformat(started[:-1] + "+00:00")
        completed_value = datetime.fromisoformat(completed[:-1] + "+00:00")
    except ValueError as exc:
        raise ValidationError(f"{label} timestamps are malformed") from exc
    if completed_value < start_value:
        raise ValidationError(f"{label} completion precedes start")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    parent = path.parent.resolve(strict=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ValidationError("validation output target is unsafe")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = -1
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"{label} is missing or is not a regular file")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValidationError(f"{label} is empty or not regular")
    return path.resolve(strict=True)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{label} must be finite")
    return number


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{label} must be a non-negative integer")
    return value


def _close(left: Any, right: Any, label: str) -> None:
    lhs = _finite(left, f"{label}.left")
    rhs = _finite(right, f"{label}.right")
    if not math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=1e-9):
        raise ValidationError(f"{label} does not match")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    resolved = _regular_file(path, label)
    try:
        value = _strict_json_loads(resolved.read_text(encoding="utf-8"), label)
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{label} root must be an object")
    return value


def _load_private_artifact(path: Path) -> dict[str, Any]:
    resolved = _regular_file(path, "private artifact")
    try:
        with tarfile.open(resolved, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) != 1:
                raise ValidationError("private artifact must contain exactly one member")
            member = members[0]
            if member.name != PRIVATE_MEMBER or not member.isfile():
                raise ValidationError("private artifact member contract is invalid")
            if member.size <= 0 or member.size > 32 * 1024 * 1024:
                raise ValidationError("private evidence has an invalid size")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValidationError("private evidence could not be read")
            raw = stream.read(member.size + 1)
            if len(raw) != member.size:
                raise ValidationError("private evidence size changed while reading")
    except (tarfile.TarError, OSError) as exc:
        raise ValidationError("private artifact is not a valid gzip tar") from exc
    try:
        value = _strict_json_loads(raw.decode("utf-8"), "private evidence")
    except UnicodeDecodeError as exc:
        raise ValidationError("private evidence is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValidationError("private evidence root must be an object")
    return value


def _optional_file_record(path: Path) -> dict[str, Any]:
    exists = path.is_file() and not path.is_symlink()
    return {
        "path": str(path.resolve(strict=False)),
        "sha256": sha256_file(path) if exists else None,
    }


def _validate_archive_pair(
    *,
    report_path: Path,
    evaluated_archive_path: Path,
    expected_archive_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluated = _regular_file(evaluated_archive_path, "evaluated submission archive")
    expected = _regular_file(expected_archive_path, "expected submission archive")
    report = _regular_file(report_path, "final report")
    required_evaluated = report.parent / "submission.tar.gz"
    if evaluated != required_evaluated.resolve(strict=True):
        raise ValidationError(
            "evaluated archive is not the submission.tar.gz beside the final report"
        )
    evaluated_hash = sha256_file(evaluated)
    expected_hash = sha256_file(expected)
    if evaluated_hash != expected_hash:
        raise ValidationError("evaluated submission archive differs from expected archive")
    return (
        {"path": str(evaluated), "sha256": evaluated_hash},
        {"path": str(expected), "sha256": expected_hash},
    )


def _validate_public_shape(
    report: dict[str, Any], *, expected_submission_id: str, max_runtime: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(report) != TOP_LEVEL_FIELDS:
        raise ValidationError("report top-level schema fields are not exact")
    if report.get("task_id") != TASK_ID:
        raise ValidationError("report task_id mismatch")
    if report.get("submission_id") != expected_submission_id:
        raise ValidationError("report submission_id mismatch")
    if not isinstance(report.get("valid"), bool):
        raise ValidationError("report valid must be boolean")
    if not isinstance(report.get("timed_out"), bool):
        raise ValidationError("report timed_out must be boolean")
    runtime = _finite(report.get("runtime_seconds"), "runtime_seconds")
    if runtime < 0.0 or runtime > max_runtime:
        raise ValidationError("runtime_seconds is outside the formal bound")
    if _finite(report.get("submitted_at"), "submitted_at") <= 0.0:
        raise ValidationError("submitted_at must be positive")
    raw_score = _finite(report.get("score"), "score")
    normalized_score = _finite(report.get("score_0_100"), "score_0_100")
    if not 0.0 <= normalized_score <= 100.0:
        raise ValidationError("score_0_100 is outside [0, 100]")
    expected_normalized = display_rescale_score(raw_score)
    _close(normalized_score, expected_normalized, "score_0_100 rescale")
    details = report.get("details")
    metrics = report.get("metrics")
    if not isinstance(details, list) or not details:
        raise ValidationError("report details must be a non-empty list")
    if not isinstance(metrics, dict):
        raise ValidationError("report metrics must be an object")
    if report.get("test_details") != []:
        raise ValidationError("structured report test_details must be empty")
    if not isinstance(report.get("summary"), str) or not report["summary"]:
        raise ValidationError("report summary must be non-empty")
    if len(report["summary"]) > 4096:
        raise ValidationError("report summary exceeds the frozen bound")
    statuses: list[str] = []
    for index, detail in enumerate(details):
        if not isinstance(detail, dict) or set(detail) != {
            "name",
            "status",
            "message",
            "score",
            "weight",
        }:
            raise ValidationError(f"detail[{index}] schema is invalid")
        if not isinstance(detail["name"], str) or not detail["name"]:
            raise ValidationError(f"detail[{index}].name is invalid")
        if (
            not isinstance(detail["status"], str)
            or detail["status"] not in {"PASSED", "FAILED", "ERROR"}
        ):
            raise ValidationError(f"detail[{index}].status is invalid")
        if not isinstance(detail["message"], str) or not detail["message"]:
            raise ValidationError(f"detail[{index}].message is invalid")
        _finite(detail["score"], f"detail[{index}].score")
        _finite(detail["weight"], f"detail[{index}].weight")
        statuses.append(detail["status"])
    counts = {
        "passed": statuses.count("PASSED"),
        "failed": statuses.count("FAILED"),
        "errors": statuses.count("ERROR"),
    }
    total = _count(report.get("total_tests"), "total_tests")
    if total != len(details):
        raise ValidationError("total_tests does not match details")
    for key, expected in counts.items():
        if _count(report.get(key), key) != expected:
            raise ValidationError(f"{key} does not match detail statuses")
    expected_rate = counts["passed"] / total
    _close(report.get("pass_rate"), expected_rate, "pass_rate")
    return details, metrics


def _validate_common_metrics(metrics: dict[str, Any], score: float) -> None:
    if metrics.get("primary_score_name") != "valid_steps_per_min":
        raise ValidationError("primary score name mismatch")
    if metrics.get("primary_score_direction") != "maximize":
        raise ValidationError("primary score direction mismatch")
    _close(metrics.get("primary_score_value"), score, "metrics.primary_score_value")
    if not isinstance(metrics.get("correctness_gate_passed"), bool):
        raise ValidationError("correctness_gate_passed must be boolean")
    if not isinstance(metrics.get("auto_feedback_safe"), bool):
        raise ValidationError("auto_feedback_safe must be boolean")


def _structured_projection(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report[key] for key in STRUCTURED_KEYS}


def _require_private_projection(
    private: dict[str, Any], report: dict[str, Any]
) -> None:
    public_result = private.get("public_result")
    if not isinstance(public_result, dict):
        raise ValidationError("private envelope lacks public_result")
    report_projection = _structured_projection(report)
    public_hash = _canonical_json_sha256(public_result)
    report_hash = _canonical_json_sha256(report_projection)
    if public_hash != report_hash or public_result != report_projection:
        raise ValidationError("private and public structured results differ")
    if private.get("public_result_sha256") != public_hash or public_hash != report_hash:
        raise ValidationError("private public_result commitment mismatch")


def _validate_private_envelope(
    private: dict[str, Any], report: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    if set(private) != {
        "schema_version",
        "kind",
        "status",
        "classification",
        "started_at_utc",
        "completed_at_utc",
        "candidate_tree_sha256",
        "commitments",
        "runner",
        "runner_evidence",
        "cleanup",
        "mlsysforge_evaluation",
        "public_result",
        "public_result_sha256",
        "private_errors",
    }:
        raise ValidationError("private envelope schema fields are not exact")
    if (
        private.get("schema_version") != 2
        or private.get("kind") != "kvbench-mlsysforge-private-evaluation"
        or private.get("status") != "terminal"
    ):
        raise ValidationError("private envelope identity is invalid")
    classification = private.get("classification")
    if not isinstance(classification, str) or classification not in {
        "valid",
        "candidate_invalid",
        "resource_busy",
        "cancelled",
        "infrastructure_error",
    }:
        raise ValidationError("private envelope classification is invalid")
    _terminal_utc_pair(
        private.get("started_at_utc"),
        private.get("completed_at_utc"),
        "private envelope",
    )
    tree = private.get("candidate_tree_sha256")
    if tree is not None and (
        not isinstance(tree, str) or not re.fullmatch(r"[0-9a-f]{64}", tree)
    ):
        raise ValidationError("private candidate tree commitment is malformed")
    if private.get("commitments") != {
        "hidden_baseline_identity": hidden_baseline_identity(),
    }:
        raise ValidationError("private frozen commitments mismatch")
    runner = private.get("runner")
    if not isinstance(runner, dict) or set(runner) != {
        "returncode",
        "timed_out",
        "cancel_signal",
        "termination_sent",
        "forced_kill",
        "runtime_seconds",
        "force_reconcile",
    }:
        raise ValidationError("private runner metadata schema mismatch")
    if (
        runner.get("returncode") is not None
        and (isinstance(runner["returncode"], bool) or not isinstance(runner["returncode"], int))
    ):
        raise ValidationError("private runner returncode is invalid")
    if not isinstance(runner.get("timed_out"), bool):
        raise ValidationError("private runner timed_out is invalid")
    cancel_signal = runner.get("cancel_signal")
    if cancel_signal is not None and (
        not isinstance(cancel_signal, str)
        or cancel_signal not in {"SIGINT", "SIGTERM"}
    ):
        raise ValidationError("private runner cancel_signal is invalid")
    if not isinstance(runner.get("termination_sent"), bool) or not isinstance(
        runner.get("forced_kill"), bool
    ):
        raise ValidationError("private runner termination metadata is invalid")
    runtime = _finite(runner.get("runtime_seconds"), "private runner runtime")
    if runtime < 0.0 or runtime > 3060.0:
        raise ValidationError("private runner runtime is outside the formal bound")
    if runner.get("force_reconcile") is not None and not isinstance(
        runner.get("force_reconcile"), dict
    ):
        raise ValidationError("private runner force_reconcile is malformed")
    errors = private.get("private_errors")
    if not isinstance(errors, list) or any(not isinstance(item, str) for item in errors):
        raise ValidationError("private_errors must be a string list")
    evidence = private.get("runner_evidence")
    if evidence is not None and not isinstance(evidence, dict):
        raise ValidationError("runner_evidence must be an object or null")
    expected_cleanup = evidence.get("cleanup") if isinstance(evidence, dict) else None
    expected_edge = (
        evidence.get("mlsysforge_evaluation") if isinstance(evidence, dict) else None
    )
    if private.get("cleanup") != expected_cleanup:
        raise ValidationError("private envelope cleanup differs from runner evidence")
    if private.get("mlsysforge_evaluation") != expected_edge:
        raise ValidationError("private envelope edge evaluation differs from runner evidence")
    if isinstance(evidence, dict) and private.get("candidate_tree_sha256") != evidence.get(
        "candidate_tree_sha256"
    ):
        raise ValidationError("private envelope candidate tree differs from runner evidence")
    _require_private_projection(private, report)
    return classification, runner, evidence


def _validate_packaging_invalid(
    report: dict[str, Any], private: dict[str, Any], details: list[dict[str, Any]], metrics: dict[str, Any]
) -> tuple[bool, float]:
    if private.get("classification") != "candidate_invalid":
        raise ValidationError("packaging failure private classification mismatch")
    if set(metrics) != {
        "source",
        "primary_score_name",
        "primary_score_value",
        "primary_score_direction",
        "correctness_gate_passed",
        "evaluation_status",
        "submission_failure_code",
        "auto_feedback_safe",
    }:
        raise ValidationError("packaging failure metrics schema is not exact")
    code = metrics.get("submission_failure_code")
    if not isinstance(code, str) or code not in {
        "submission_not_directory",
        "submission_symlink",
        "submission_special_file",
        "submission_too_large",
        "launch_not_executable",
        "submission_contract_invalid",
    }:
        raise ValidationError("submission failure code is not allowlisted")
    if (
        report["valid"] is not False
        or report["errors"] != 0
        or report["failed"] != 1
        or report["passed"] != 0
        or len(details) != 1
        or details[0]["name"] != "hard_gate:submission_contract"
        or details[0]["status"] != "FAILED"
        or details[0]["message"] != code
        or _finite(details[0]["score"], "packaging detail score") != 0.0
        or _finite(details[0]["weight"], "packaging detail weight") != 1.0
        or metrics.get("evaluation_status") != "candidate_invalid"
        or metrics.get("source") != "kvbench_mlsysforge_adapter"
        or metrics.get("correctness_gate_passed") is not False
        or metrics.get("auto_feedback_safe") is not True
    ):
        raise ValidationError("candidate packaging-invalid semantics are inconsistent")
    score = _finite(report["score"], "score")
    if score != 0.0:
        raise ValidationError("candidate-invalid score must be zero")
    runner = private["runner"]
    private_errors = private["private_errors"]
    if (
        private.get("candidate_tree_sha256") is not None
        or private.get("runner_evidence") is not None
        or private.get("cleanup") is not None
        or private.get("mlsysforge_evaluation") is not None
        or runner.get("returncode") is not None
        or runner.get("timed_out") is not False
        or runner.get("cancel_signal") is not None
        or runner.get("termination_sent") is not False
        or runner.get("forced_kill") is not False
        or runner.get("force_reconcile") is not None
        or _finite(runner.get("runtime_seconds"), "packaging runner runtime") != 0.0
        or private_errors != [f"SubmissionError: {code}"]
    ):
        raise ValidationError("packaging-invalid private terminal semantics mismatch")
    return False, score


def _validate_hard_gate_result(
    report: dict[str, Any], private: dict[str, Any], details: list[dict[str, Any]], metrics: dict[str, Any]
) -> tuple[bool, float]:
    if set(private) != {
        "schema_version",
        "kind",
        "status",
        "phase",
        "started_at_utc",
        "completed_at_utc",
        "direct_leg_executed",
        "candidate_dir",
        "candidate_tree_sha256",
        "preflight_backend_serving_contract_sha256",
        "baseline_evidence",
        "leg",
        "evaluation",
        "errors",
        "cleanup",
        "limitations",
        "mlsysforge_evaluation",
        "mlsysforge_adapter",
    }:
        raise ValidationError("private runner root schema fields are not exact")
    if (
        isinstance(private.get("schema_version"), bool)
        or private.get("schema_version") != 1
    ):
        raise ValidationError("private runner schema_version mismatch")
    if private.get("kind") != "gpu67-v2.2-hidden-candidate-only-vs-frozen-v2.1-direct":
        raise ValidationError("private runner kind mismatch")
    if private.get("status") != "complete":
        raise ValidationError("private runner status is not complete")
    if private.get("phase") != "quiescent":
        raise ValidationError("private runner phase is not quiescent")
    if private.get("direct_leg_executed") is not False:
        raise ValidationError("private runner unexpectedly executed Direct")
    if private.get("errors") != []:
        raise ValidationError("private runner recorded an error, including teardown failure")
    cleanup = private.get("cleanup")
    if not isinstance(cleanup, dict) or set(cleanup) != {
        "attempted",
        "stack_stop_succeeded",
        "quiescent",
        "reconciled_before_start",
        "residue",
    }:
        raise ValidationError("private runner cleanup schema mismatch")
    if (
        cleanup.get("attempted") is not True
        or cleanup.get("stack_stop_succeeded") is not True
        or cleanup.get("quiescent") is not True
        or not isinstance(cleanup.get("reconciled_before_start"), bool)
        or cleanup.get("residue")
        != {"compose_container_ids": [], "sandbox_container_ids": []}
    ):
        raise ValidationError("private runner cleanup is not proven quiescent")
    _terminal_utc_pair(
        private.get("started_at_utc"),
        private.get("completed_at_utc"),
        "private runner",
    )
    limitations = private.get("limitations")
    if not isinstance(limitations, list) or any(
        not isinstance(item, str) or not item for item in limitations
    ):
        raise ValidationError("private runner limitations are malformed")
    candidate_dir = private.get("candidate_dir")
    candidate_tree = private.get("candidate_tree_sha256")
    if not isinstance(candidate_dir, str) or not candidate_dir.startswith("/"):
        raise ValidationError("private candidate_dir is not absolute")
    if not isinstance(candidate_tree, str) or not re.fullmatch(r"[0-9a-f]{64}", candidate_tree):
        raise ValidationError("private candidate tree commitment is malformed")
    if (
        private.get("preflight_backend_serving_contract_sha256")
        != baseline_contract_sha256()
    ):
        raise ValidationError("private preflight backend serving contract mismatch")
    baseline = private.get("baseline_evidence")
    if not isinstance(baseline, dict) or set(baseline) != {
        "path",
        "file_sha256",
        "workload_nonce_sha256",
        "direct_valid_steps_per_min",
        "backend_serving_contract_sha256",
    }:
        raise ValidationError("private baseline evidence schema mismatch")
    if (
        baseline.get("file_sha256") != baseline_sha256()
        or baseline.get("workload_nonce_sha256") != EXPECTED_NONCE_SHA256
        or baseline.get("backend_serving_contract_sha256")
        != baseline_contract_sha256()
    ):
        raise ValidationError("private baseline commitment mismatch")
    _close(
        baseline.get("direct_valid_steps_per_min"),
        direct_rate(),
        "private baseline Direct rate",
    )
    if not isinstance(private.get("leg"), dict) or not isinstance(private.get("evaluation"), dict):
        raise ValidationError("private runner leg/evaluation is malformed")
    leg = private["leg"]
    evaluation = private["evaluation"]
    provenance = leg.get("model_provenance")
    if (
        isinstance(leg.get("schema_version"), bool)
        or leg.get("schema_version") != 1
        or leg.get("mode") != "candidate"
        or leg.get("profile") != "hidden-pressure-192"
        or leg.get("workload_nonce_sha256") != EXPECTED_NONCE_SHA256
        or not isinstance(provenance, dict)
        or provenance.get("backend_serving_contract_sha256")
        not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
    ):
        raise ValidationError("private candidate leg identity mismatch")
    if (
        evaluation.get("suite") != "hidden"
        or evaluation.get("profile") != "hidden-pressure-192"
        or evaluation.get("backend_serving_contract_sha256")
        not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
    ):
        raise ValidationError("private evaluation identity mismatch")
    _close(
        evaluation.get("direct_valid_steps_per_min"),
        direct_rate(),
        "private evaluation Direct rate",
    )
    edge = private.get("mlsysforge_evaluation")
    adapter = private.get("mlsysforge_adapter")
    if not isinstance(edge, dict) or not isinstance(adapter, dict):
        raise ValidationError("private MLSys Forge adapter evidence is missing")
    if set(edge) != {
        "schema_version",
        "valid",
        "score",
        "primary_score_name",
        "direct_valid_steps_per_min",
        "speedup_over_frozen_direct",
        "change_percent",
        "strictly_better_than_frozen_direct",
        "reasons",
        "checks",
    }:
        raise ValidationError("private MLSys Forge result schema is not exact")
    if isinstance(adapter.get("schema_version"), bool) or adapter != {
        "schema_version": 1,
        "candidate_only": True,
        "fresh_paired_ab": False,
        "canonical_additional_hard_gates_applied": True,
        "primary_score": "candidate_valid_steps_per_min",
    }:
        raise ValidationError("private MLSys Forge adapter contract mismatch")
    checks = edge.get("checks")
    if not isinstance(checks, list) or len(checks) != len(HARD_GATES):
        raise ValidationError("private hard-gate count mismatch")
    if any(
        not isinstance(check, dict)
        or set(check) != {"name", "passed", "message", "metrics"}
        or not isinstance(check.get("message"), str)
        or not check.get("message")
        or not isinstance(check.get("metrics"), dict)
        for check in checks
    ):
        raise ValidationError("private hard-gate schemas are not exact")
    if any(
        not isinstance(check.get("name"), str)
        or not isinstance(check.get("passed"), bool)
        for check in checks
    ):
        raise ValidationError("private hard-gate names or pass states are malformed")
    check_names = [check["name"] for check in checks]
    if len(check_names) != len(checks) or set(check_names) != HARD_GATES:
        raise ValidationError("private hard gates are missing, duplicated, or unknown")
    failed_names = [str(check["name"]) for check in checks if check["passed"] is not True]
    edge_valid = not failed_names
    if (
        isinstance(edge.get("schema_version"), bool)
        or edge.get("schema_version") != 1
        or edge.get("valid") is not edge_valid
    ):
        raise ValidationError("private hard-gate validity is not the conjunction of all gates")
    if edge.get("reasons") != failed_names:
        raise ValidationError("private hard-gate failure reasons do not match checks")
    public_names = [detail["name"] for detail in details]
    if len(public_names) != len(HARD_GATES) or set(public_names) != {
        f"hard_gate:{name}" for name in HARD_GATES
    }:
        raise ValidationError("public hard gates are missing, duplicated, or unknown")
    public_passed = {
        detail["name"].removeprefix("hard_gate:")
        for detail in details
        if detail["status"] == "PASSED"
    }
    private_passed = {str(check["name"]) for check in checks if check["passed"] is True}
    if public_passed != private_passed or any(d["status"] == "ERROR" for d in details):
        raise ValidationError("public and private hard-gate states differ")
    if any(
        _finite(detail["weight"], "hard-gate detail weight") != 1.0
        or _finite(detail["score"], "hard-gate detail score")
        != (1.0 if detail["status"] == "PASSED" else 0.0)
        for detail in details
    ):
        raise ValidationError("public hard-gate detail scores are inconsistent")
    score = _finite(report["score"], "score")
    edge_score = _finite(edge.get("score"), "private edge score")
    _close(score, edge_score, "public/private score")
    if edge.get("primary_score_name") != "valid_steps_per_min":
        raise ValidationError("private edge primary score name mismatch")
    _close(
        edge.get("direct_valid_steps_per_min"),
        direct_rate(),
        "private edge Direct rate",
    )
    expected_strictly_better = bool(edge_valid and score > direct_rate())
    if edge_valid:
        candidate_rate = _finite(
            private["leg"].get("valid_steps_per_min"), "private candidate rate"
        )
        evaluation_rate = _finite(
            private["evaluation"].get("candidate_valid_steps_per_min"),
            "private evaluation candidate rate",
        )
        _close(score, candidate_rate, "score/candidate rate")
        _close(score, evaluation_rate, "score/evaluation rate")
        if score <= 0.0:
            raise ValidationError("valid candidate score must be positive")
        expected_speedup = score / direct_rate()
        expected_change = (expected_speedup - 1.0) * 100.0
        _close(
            edge.get("speedup_over_frozen_direct"),
            expected_speedup,
            "private edge speedup",
        )
        _close(
            edge.get("change_percent"),
            expected_change,
            "private edge change percent",
        )
    else:
        if score != 0.0:
            raise ValidationError("hard-gate-invalid candidate score must be zero")
        expected_speedup = None
        expected_change = None
        if (
            edge.get("speedup_over_frozen_direct") is not None
            or edge.get("change_percent") is not None
        ):
            raise ValidationError("invalid private edge leaked derived score values")
    if edge.get("strictly_better_than_frozen_direct") is not expected_strictly_better:
        raise ValidationError("private edge Direct comparison mismatch")
    if report["valid"] is not edge_valid:
        raise ValidationError("public valid differs from private hard-gate validity")
    if metrics.get("source") != "kvbench_candidate_only_vs_serving_contract_direct":
        raise ValidationError("hard-gate metrics source mismatch")
    if set(metrics) != {
        "source",
        "evaluation_contract",
        "primary_score_name",
        "primary_score_value",
        "primary_score_unit",
        "primary_score_direction",
        "correctness_gate_mode",
        "correctness_gate_passed",
        "frozen_direct_valid_steps_per_min",
        "speedup_over_frozen_direct",
        "change_percent",
        "strictly_better_than_frozen_direct",
        "hard_gates_passed",
        "hard_gates_total",
        "hard_gate_failures",
        "evaluation_status",
        "auto_feedback_safe",
    }:
        raise ValidationError("hard-gate metrics schema is not exact")
    if metrics.get("evaluation_status") != (
        "valid" if edge_valid else "candidate_invalid"
    ):
        raise ValidationError("hard-gate evaluation_status mismatch")
    if metrics.get("evaluation_contract") != "candidate_only_same_nonce_same_serving_contract":
        raise ValidationError("evaluation contract mismatch")
    if (
        metrics.get("primary_score_unit")
        != "valid executable agent steps/min"
        or metrics.get("correctness_gate_mode") != "binary"
    ):
        raise ValidationError("hard-gate metric identity values mismatch")
    _close(
        metrics.get("frozen_direct_valid_steps_per_min"),
        direct_rate(),
        "metrics frozen Direct rate",
    )
    if edge_valid:
        _close(
            metrics.get("speedup_over_frozen_direct"),
            expected_speedup,
            "metrics speedup",
        )
        _close(
            metrics.get("change_percent"),
            expected_change,
            "metrics change percent",
        )
    elif (
        metrics.get("speedup_over_frozen_direct") is not None
        or metrics.get("change_percent") is not None
    ):
        raise ValidationError("invalid metrics leaked derived score values")
    if metrics.get("strictly_better_than_frozen_direct") is not expected_strictly_better:
        raise ValidationError("metrics Direct comparison mismatch")
    if metrics.get("correctness_gate_passed") is not edge_valid:
        raise ValidationError("metrics correctness gate mismatch")
    if metrics.get("auto_feedback_safe") is not True:
        raise ValidationError("hard-gate feedback safety marker mismatch")
    if metrics.get("hard_gates_total") != len(HARD_GATES):
        raise ValidationError("metrics hard-gate total mismatch")
    if metrics.get("hard_gates_passed") != len(private_passed):
        raise ValidationError("metrics hard-gate passed count mismatch")
    if metrics.get("hard_gate_failures") != failed_names:
        raise ValidationError("metrics hard-gate failures mismatch")
    return edge_valid, score


def _validate_infrastructure(
    report: dict[str, Any],
    private: dict[str, Any],
    details: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    classification: str,
    runner: dict[str, Any],
    evidence: dict[str, Any] | None,
) -> str:
    if not isinstance(classification, str) or classification not in {
        "resource_busy",
        "cancelled",
        "infrastructure_error",
    }:
        raise ValidationError("non-authoritative private classification mismatch")
    if set(metrics) != {
        "source",
        "primary_score_name",
        "primary_score_value",
        "primary_score_direction",
        "correctness_gate_passed",
        "evaluation_status",
        "auto_feedback_safe",
    }:
        raise ValidationError("infrastructure metrics schema is not exact")
    private_errors = private.get("private_errors")
    if (
        not isinstance(private_errors, list)
        or not private_errors
        or report["valid"] is not False
        or _finite(report["score"], "score") != 0.0
        or report["errors"] != 1
        or report["failed"] != 0
        or report["passed"] != 0
        or len(details) != 1
        or details[0]["name"] != "judge:infrastructure"
        or details[0]["status"] != "ERROR"
        or _finite(details[0]["score"], "infrastructure detail score") != 0.0
        or _finite(details[0]["weight"], "infrastructure detail weight") != 1.0
        or metrics.get("source") != "kvbench_mlsysforge_adapter"
        or metrics.get("evaluation_status") != classification
        or metrics.get("correctness_gate_passed") is not False
        or metrics.get("auto_feedback_safe") is not False
    ):
        raise ValidationError("infrastructure report semantics are inconsistent")

    if classification == "resource_busy":
        # A formal retry is reserved for the launcher's own lock-release race.
        # Ingress has already stopped and quiescence was verified under this
        # same lock, so any container residue means the handoff invariant was
        # broken by an external/stale runtime and must not be retried.
        if not isinstance(evidence, dict):
            raise ValidationError("resource-busy runner evidence is missing")
        if set(evidence) != {
            "schema_version",
            "kind",
            "status",
            "phase",
            "started_at_utc",
            "completed_at_utc",
            "direct_leg_executed",
            "candidate_dir",
            "candidate_tree_sha256",
            "preflight_backend_serving_contract_sha256",
            "baseline_evidence",
            "leg",
            "evaluation",
            "errors",
            "cleanup",
            "limitations",
        }:
            raise ValidationError("resource-busy runner evidence schema is not exact")
        cleanup = evidence.get("cleanup")
        baseline = evidence.get("baseline_evidence")
        if (
            runner.get("returncode") != 4
            or runner.get("timed_out") is not False
            or runner.get("cancel_signal") is not None
            or runner.get("termination_sent") is not False
            or runner.get("forced_kill") is not False
            or runner.get("force_reconcile") is not None
            or isinstance(evidence.get("schema_version"), bool)
            or evidence.get("schema_version") != 1
            or evidence.get("kind")
            != "gpu67-v2.2-hidden-candidate-only-vs-frozen-v2.1-direct"
            or evidence.get("status") != "resource_busy"
            or evidence.get("phase") != "terminal"
            or evidence.get("direct_leg_executed") is not False
            or evidence.get("preflight_backend_serving_contract_sha256") is not None
            or evidence.get("leg") is not None
            or evidence.get("evaluation") is not None
            or evidence.get("errors")
            != ["BlockingIOError: measurement lock is already held"]
            or not isinstance(evidence.get("candidate_dir"), str)
            or not evidence["candidate_dir"].startswith("/")
            or evidence.get("candidate_tree_sha256")
            != private.get("candidate_tree_sha256")
            or not isinstance(baseline, dict)
            or baseline.get("file_sha256") != baseline_sha256()
            or baseline.get("workload_nonce_sha256") != EXPECTED_NONCE_SHA256
            or baseline.get("backend_serving_contract_sha256")
            != baseline_contract_sha256()
            or not isinstance(cleanup, dict)
            or cleanup
            != {
                "attempted": False,
                "stack_stop_succeeded": False,
                "quiescent": True,
                "reconciled_before_start": False,
                "residue": {
                    "compose_container_ids": [],
                    "sandbox_container_ids": [],
                },
            }
            or private_errors
            != ["RuntimeError: trusted runner returned a non-terminal exit"]
        ):
            raise ValidationError(
                "resource-busy signature is not the exact quiescent self-handoff LOCK_NB contract"
            )
        return "measurement_lock_contention"
    if classification == "cancelled":
        if runner.get("cancel_signal") not in {"SIGINT", "SIGTERM"}:
            raise ValidationError("cancelled evaluation lacks a supervisor signal")
        return "cancelled"
    return "infrastructure_error"


def _non_authoritative_payload(
    *,
    classification: str,
    expected_submission_id: str,
    eval_rc: int,
    report_path: Path,
    artifact_path: Path,
    evaluated_archive_path: Path,
    expected_archive_path: Path,
) -> dict[str, Any]:
    report_exists = report_path.is_file() and not report_path.is_symlink()
    artifact_exists = artifact_path.is_file() and not artifact_path.is_symlink()
    return {
        "schema_version": 1,
        "hidden_baseline_identity": hidden_baseline_identity(),
        "classification": classification,
        "authoritative": False,
        "retryable": False,
        "task_id": TASK_ID,
        "submission_id": expected_submission_id,
        "eval_rc": eval_rc,
        "valid": None,
        "score": None,
        "report": {
            "path": str(report_path.resolve(strict=False)),
            "sha256": sha256_file(report_path) if report_exists else None,
        },
        "private_artifact": {
            "path": str(artifact_path.resolve(strict=False)),
            "sha256": sha256_file(artifact_path) if artifact_exists else None,
        },
        "evaluated_archive": _optional_file_record(evaluated_archive_path),
        "expected_archive": _optional_file_record(expected_archive_path),
        "errors": [],
    }


def validate(
    *,
    report_path: Path,
    artifact_path: Path,
    expected_submission_id: str,
    eval_rc: int,
    max_runtime: float,
    evaluated_archive_path: Path,
    expected_archive_path: Path,
) -> dict[str, Any]:
    report = _load_json(report_path, "final report")
    if eval_rc != 0:
        return _non_authoritative_payload(
            classification="evaluation_process_error",
            expected_submission_id=expected_submission_id,
            eval_rc=eval_rc,
            report_path=report_path,
            artifact_path=artifact_path,
            evaluated_archive_path=evaluated_archive_path,
            expected_archive_path=expected_archive_path,
        )
    if report.get("timed_out") is True:
        if report.get("task_id") != TASK_ID or report.get("submission_id") != expected_submission_id:
            raise ValidationError("timed-out report identity mismatch")
        return _non_authoritative_payload(
            classification="timeout",
            expected_submission_id=expected_submission_id,
            eval_rc=eval_rc,
            report_path=report_path,
            artifact_path=artifact_path,
            evaluated_archive_path=evaluated_archive_path,
            expected_archive_path=expected_archive_path,
        )
    details, metrics = _validate_public_shape(
        report,
        expected_submission_id=expected_submission_id,
        max_runtime=max_runtime,
    )
    score = _finite(report["score"], "score")
    _validate_common_metrics(metrics, score)
    evaluated_archive_record, expected_archive_record = _validate_archive_pair(
        report_path=report_path,
        evaluated_archive_path=evaluated_archive_path,
        expected_archive_path=expected_archive_path,
    )
    report_record = {
        "path": str(report_path.resolve(strict=True)),
        "sha256": sha256_file(report_path),
    }
    artifact_exists = artifact_path.is_file() and not artifact_path.is_symlink()
    artifact_record = {
        "path": str(artifact_path.resolve(strict=False)),
        "sha256": sha256_file(artifact_path) if artifact_exists else None,
    }
    private = _load_private_artifact(artifact_path)
    artifact_record = {
        "path": str(artifact_path.resolve(strict=True)),
        "sha256": sha256_file(artifact_path),
    }
    envelope_classification, runner, evidence = _validate_private_envelope(
        private, report
    )
    if envelope_classification in {
        "resource_busy",
        "cancelled",
        "infrastructure_error",
    }:
        classification = _validate_infrastructure(
            report,
            private,
            details,
            metrics,
            classification=envelope_classification,
            runner=runner,
            evidence=evidence,
        )
        authoritative = False
        valid: bool | None = None
    elif (
        envelope_classification == "candidate_invalid"
        and metrics.get("source") == "kvbench_mlsysforge_adapter"
    ):
        valid, score = _validate_packaging_invalid(report, private, details, metrics)
        classification = "candidate_invalid"
        authoritative = True
    elif (
        envelope_classification in {"valid", "candidate_invalid"}
        and metrics.get("source")
        == "kvbench_candidate_only_vs_serving_contract_direct"
    ):
        if (
            runner.get("returncode") not in (0, 3)
            or runner.get("timed_out") is not False
            or runner.get("cancel_signal") is not None
            or runner.get("termination_sent") is not False
            or runner.get("forced_kill") is not False
            or runner.get("force_reconcile") is not None
            or private.get("private_errors") != []
            or not isinstance(evidence, dict)
        ):
            raise ValidationError("authoritative runner terminal metadata is inconsistent")
        valid, score = _validate_hard_gate_result(report, evidence, details, metrics)
        classification = "candidate_valid" if valid else "candidate_invalid"
        expected_envelope_classification = "valid" if valid else "candidate_invalid"
        if envelope_classification != expected_envelope_classification:
            raise ValidationError("private envelope and hard-gate classifications differ")
        authoritative = True
    else:
        raise ValidationError("private classification and final report source disagree")
    return {
        "schema_version": 1,
        "hidden_baseline_identity": hidden_baseline_identity(),
        "classification": classification,
        "authoritative": authoritative,
        "retryable": classification == "measurement_lock_contention",
        "task_id": TASK_ID,
        "submission_id": expected_submission_id,
        "eval_rc": eval_rc,
        "valid": valid,
        "score": score if authoritative else None,
        "report": report_record,
        "private_artifact": artifact_record,
        "evaluated_archive": evaluated_archive_record,
        "expected_archive": expected_archive_record,
        "errors": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--private-artifact", required=True, type=Path)
    parser.add_argument("--expected-submission-id", required=True)
    parser.add_argument("--eval-rc", required=True, type=int)
    parser.add_argument("--max-runtime-seconds", type=float, default=3060.0)
    parser.add_argument("--evaluated-archive", required=True, type=Path)
    parser.add_argument("--expected-archive", required=True, type=Path)
    parser.add_argument("--expected-preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        configure_display_rescale_contract()
        configure_hidden_baseline_contract(args.expected_preflight)
        payload = validate(
            report_path=args.report,
            artifact_path=args.private_artifact,
            expected_submission_id=args.expected_submission_id,
            eval_rc=args.eval_rc,
            max_runtime=args.max_runtime_seconds,
            evaluated_archive_path=args.evaluated_archive,
            expected_archive_path=args.expected_archive,
        )
    except Exception as exc:
        report_exists = args.report.is_file() and not args.report.is_symlink()
        artifact_exists = (
            args.private_artifact.is_file() and not args.private_artifact.is_symlink()
        )
        evaluated_exists = (
            args.evaluated_archive.is_file()
            and not args.evaluated_archive.is_symlink()
        )
        expected_exists = (
            args.expected_archive.is_file() and not args.expected_archive.is_symlink()
        )
        classification = (
            "artifact_missing"
            if report_exists
            and not (artifact_exists and evaluated_exists and expected_exists)
            else "malformed"
        )
        if args.eval_rc != 0:
            classification = "evaluation_process_error"
        diagnostic = (
            str(exc)
            if isinstance(exc, ValidationError)
            else f"unexpected validation failure: {type(exc).__name__}: {exc}"
        )
        payload = {
            "schema_version": 1,
            "hidden_baseline_identity": hidden_baseline_identity(),
            "classification": classification,
            "authoritative": False,
            "retryable": False,
            "task_id": TASK_ID,
            "submission_id": args.expected_submission_id,
            "eval_rc": args.eval_rc,
            "valid": None,
            "score": None,
            "report": {
                "path": str(args.report.resolve(strict=False)),
                "sha256": sha256_file(args.report) if report_exists else None,
            },
            "private_artifact": {
                "path": str(args.private_artifact.resolve(strict=False)),
                "sha256": sha256_file(args.private_artifact) if artifact_exists else None,
            },
            "evaluated_archive": _optional_file_record(args.evaluated_archive),
            "expected_archive": _optional_file_record(args.expected_archive),
            "errors": [diagnostic],
        }
    _atomic_write(args.output, payload)
    return EXIT_CODES[payload["classification"]]


if __name__ == "__main__":
    raise SystemExit(main())
