#!/usr/bin/env python3
"""Trusted MLSys Forge adapter for the serving-contract Candidate-only screen."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any


# Backend identity is the *serving system*, not the model of GPU hosting it.
# Contract revision v2.5 replaced ``allowed_names: ["NVIDIA H100 NVL"]`` with a
# capability predicate (2 devices, compute capability >= 8.0, >= 95000 MiB of
# device memory each).  Every field that determines measured throughput — model
# revision, runtime image id, tensor-parallel size, KV dtype and bytes,
# scheduler limits — is byte-identical across v1 and v2.5, so evidence recorded
# under either revision describes the same serving system and both digests are
# accepted.  ``scripts/preflight.py`` re-derives that equivalence from the two
# contract files on disk before every run, and refuses to start if a revision
# ever changes a field outside its accelerator declaration.
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
EXPECTED_NONCE_COMMITMENT_SHA256 = (
    "c525ae31a2bb1d3ed0232c345f9fc978fd52574d415aaa2777bead17f2fac20a"
)
EXPECTED_RUNNER_KIND = (
    "gpu67-v2.2-hidden-candidate-only-vs-frozen-v2.1-direct"
)
EXPECTED_MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
# Since contract revision v3 the runtime image identity is the sha256 of the
# build recipe in ``backend-serving-contract.json`` (``runtime_image_recipe``),
# not the Docker image ID of one host's build.  Evidence recorded under v1 or
# v2.5 still carries that host's image ID, and is accepted only together with
# one of those two digests.
EXPECTED_RUNTIME_IMAGE_ID = (
    "sha256:55e81244ebb7d124b8f5bbb6f2daba163d4d506fc803dd6a19ae1f09139d6e1a"
)
LEGACY_RUNTIME_IMAGE_ID = (
    "sha256:1d8369b9e3c91aa8cba5c7aeeef26e3c4afd455e5d2ce9aebe6f30f6dd9344bd"
)
LEGACY_MODEL_PROVENANCE = {
    "profile_id": "qwen3.6-27b-6a9e13bd-h100-nvl-tp2-kv30g-v2",
    "profile_sha256": (
        "7289ed315324b9b5165d4d680efb93004c59e0d34f0f98ef9d85406ed5a9153a"
    ),
    "model_revision": EXPECTED_MODEL_REVISION,
    "runtime_image_id": LEGACY_RUNTIME_IMAGE_ID,
    "tensor_parallel_size": 2,
    "kv_cache_dtype": "fp8",
}
MODEL_PROVENANCE_FIELDS = tuple(LEGACY_MODEL_PROVENANCE)
HIDDEN_BASELINE_IDENTITY_FIELDS = frozenset(
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
BASELINE_PATH = Path(
    os.environ.get("KVBENCH_HIDDEN_BASELINE_PATH", "/run/kvbench-baselines/hidden.json")
)
MAX_FILES = 2048
MAX_SOURCE_BYTES = 32 * 1024 * 1024
DEFAULT_RUNNER_TIMEOUT_SEC = 2500
DEFAULT_RUNNER_CLEANUP_GRACE_SEC = 300
DEFAULT_FORCE_RECONCILE_TIMEOUT_SEC = 120
MAX_TRUSTED_JSON_BYTES = 128 * 1024 * 1024

MIN_REQUEST_SUCCESS = 0.98
REQUEST_SUCCESS_TOLERANCE = 0.02
MIN_VALID_STEP_RATIO = 0.90
VALID_STEP_RATIO_TOLERANCE = 0.06
# Bounded waiting is expressed as a fraction of the leg's own scored window,
# not as a wall-clock constant.  720 s on the 1800 s hidden window was measured
# on one backend; as an absolute number it silently tightens whenever the
# backend is slower and loosens whenever it is faster, so a candidate could be
# invalidated by the host rather than by its own queueing.  The fraction keeps
# the same meaning on any window and any backend: no more than 40 % of the
# scored window may be spent waiting at the 99th percentile.  0.4 * 1800 = 720
# reproduces the hidden gate exactly, and 0.4 * 600 = 240 is the matching
# advisory ceiling reported by the public plane.
P99_GATEWAY_WAIT_WINDOW_FRACTION = 0.4
# paired_backend_work is a floor only.  A better
# scheduler moves programs deeper into their trajectories within the fixed window, so
# later steps carry longer prompts and more backend tokens per valid step; on faster
# backends that alone pushed the frozen Oracle to 1.159 and the seed to 1.17, above
# a 1.15 cap.  More backend work per step is not a way to
# inflate the step count, and the one-to-one audit already proves every request is
# forwarded unchanged, so only a candidate doing materially less work per step than
# Direct (below 0.90) is refused.
BACKEND_WORK_RATIO_MIN = 0.90
BACKEND_WORK_RATIO_MAX = None
RUNNER_REASON_ALLOWLIST = frozenset(
    {
        "release_id_mismatch",
        "leg_identity_mismatch",
        "infrastructure_failure",
        "candidate_failure",
        "profile_shape_mismatch",
        "workflow_count_mismatch",
        "task_ids_not_192_distinct",
        "run_ids_not_192_distinct",
        "per_workflow_step_ceiling_invalid",
        "workflow_step_totals_mismatch",
        "backend_audit_invalid",
        "warmup_audit_invalid",
        "measurement_window_missing",
        "measurement_window_contract_invalid",
        "truncated_workflow_count_invalid",
        "step_partition_invalid",
        "measurement_boundaries_invalid",
        "throughput_accounting_invalid",
        "request_success_below_floor",
        "valid_step_ratio_below_floor",
        "baseline_high_kv_pressure_invalid",
        "candidate_pressure_diagnostic_invalid",
        "candidate_program_lifecycle_invalid",
        "workload_nonce_pair_mismatch",
        "controller_config_sha256_pair_mismatch",
        "corpus_fingerprint_pair_mismatch",
        "corpus_namespace_pair_mismatch",
        "release_id_pair_mismatch",
        "controller_config_sha256_invalid",
        "profile_config_pair_mismatch",
        "model_adapter_pair_mismatch",
        "task_set_pair_mismatch",
        "model_provenance_mismatch",
        "backend_serving_contract_mismatch",
        "controller_config_mismatch",
        "ordered_task_assignment_mismatch",
    }
)
HARD_GATE_NAMES = frozenset(
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


class SubmissionError(ValueError):
    """Safe, fixed-code candidate packaging failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_json_object_bytes(raw: bytes, *, description: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8")
        parsed = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"{description} is not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{description} is not an object")
    return parsed


def _strict_json_object(path: Path) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size > MAX_TRUSTED_JSON_BYTES
    ):
        raise RuntimeError("trusted runner evidence file is unavailable or unsafe")
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_TRUSTED_JSON_BYTES:
        raise RuntimeError("trusted runner evidence file is unavailable or unsafe")
    return _strict_json_object_bytes(raw, description="trusted runner evidence")


def _hidden_baseline_identity_from_runner_evidence(
    baseline: Any,
) -> dict[str, Any]:
    if not isinstance(baseline, dict) or set(baseline) != RUNNER_BASELINE_EVIDENCE_FIELDS:
        raise RuntimeError("trusted runner baseline evidence contract failed")
    rate = _finite(baseline.get("direct_valid_steps_per_min"))
    identity = {
        "file_sha256": baseline.get("file_sha256"),
        "valid_steps_per_min": rate,
        "backend_serving_contract_sha256": baseline.get(
            "backend_serving_contract_sha256"
        ),
        "workload_nonce_sha256": baseline.get("workload_nonce_sha256"),
    }
    if (
        not isinstance(baseline.get("path"), str)
        or not baseline["path"]
        or not _valid_hidden_baseline_identity(identity)
    ):
        raise RuntimeError("trusted runner baseline evidence contract failed")
    return identity


def _valid_hidden_baseline_identity(identity: Any) -> bool:
    if not isinstance(identity, dict) or set(identity) != HIDDEN_BASELINE_IDENTITY_FIELDS:
        return False
    file_sha = identity.get("file_sha256")
    rate = _finite(identity.get("valid_steps_per_min"))
    return bool(
        isinstance(file_sha, str)
        and len(file_sha) == 64
        and all(character in "0123456789abcdef" for character in file_sha)
        and rate is not None
        and rate > 0.0
        and identity.get("backend_serving_contract_sha256")
        in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
        and identity.get("workload_nonce_sha256")
        == EXPECTED_NONCE_COMMITMENT_SHA256
    )


def _validate_complete_runner_report(
    report: dict[str, Any],
    candidate_sha256: str,
    baseline_bundle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cleanup = report.get("cleanup")
    baseline = report.get("baseline_evidence")
    residue = cleanup.get("residue") if isinstance(cleanup, dict) else None
    expected_identity = baseline_bundle.get("hidden_baseline_identity")
    try:
        observed_identity = _hidden_baseline_identity_from_runner_evidence(baseline)
    except RuntimeError:
        observed_identity = None
    if (
        report.get("schema_version") != 1
        or report.get("kind") != EXPECTED_RUNNER_KIND
        or report.get("status") != "complete"
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
        or not _valid_hidden_baseline_identity(expected_identity)
        or observed_identity != expected_identity
        or baseline.get("path") != baseline_bundle.get("path")
    ):
        raise RuntimeError("trusted runner terminal evidence contract failed")
    candidate = report.get("leg")
    evaluation = report.get("evaluation")
    if not isinstance(candidate, dict) or not isinstance(evaluation, dict):
        raise RuntimeError("trusted runner evidence has no complete candidate leg")
    return candidate, evaluation


def submission_failure(code: str) -> dict[str, Any]:
    safe_codes = {
        "submission_not_directory",
        "submission_symlink",
        "submission_special_file",
        "submission_too_large",
        "launch_not_executable",
    }
    normalized = code if code in safe_codes else "submission_contract_invalid"
    return {
        "valid": False,
        "score": 0.0,
        "pass_rate": 0.0,
        "total_tests": 1,
        "passed": 0,
        "failed": 1,
        "errors": 0,
        "summary": f"INVALID SUBMISSION: packaging gate failed ({normalized})",
        "details": [
            {
                "name": "hard_gate:submission_contract",
                "status": "FAILED",
                "message": normalized,
                "score": 0.0,
                "weight": 1.0,
            }
        ],
        "metrics": {
            "source": "kvbench_mlsysforge_adapter",
            "primary_score_name": "valid_steps_per_min",
            "primary_score_value": 0.0,
            "primary_score_direction": "maximize",
            "correctness_gate_passed": False,
            "evaluation_status": "candidate_invalid",
            "submission_failure_code": normalized,
            "auto_feedback_safe": True,
        },
    }


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _bounded(value: Any, limit: int = 800) -> str:
    text = " ".join(str(value).replace("\x00", "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    """Match the frozen runner's candidate tree commitment."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def fixed_window_backend_audit_valid(audit: Any) -> bool:
    """Validate strict traffic accounting with hard-cutoff right-censoring."""
    count_fields = (
        "client_requests",
        "audited_backend_requests",
        "strict_client_requests",
        "cutoff_client_requests",
        "cutoff_backend_admitted",
        "cutoff_backend_unadmitted",
        "extra_backend_requests",
        "missing_backend_requests",
    )
    if (
        not isinstance(audit, dict)
        or audit.get("passed") is not True
        or audit.get("window_censored") is not True
        or audit.get("mismatches") != []
        or any(
            not isinstance(audit.get(key), int)
            or isinstance(audit.get(key), bool)
            or audit[key] < 0
            for key in count_fields
        )
        or audit.get("extra_backend_requests") != 0
    ):
        return False
    return bool(
        audit["missing_backend_requests"] == audit["cutoff_backend_unadmitted"]
        and audit["strict_client_requests"] + audit["cutoff_client_requests"]
        == audit["client_requests"]
        and audit["cutoff_backend_admitted"] + audit["cutoff_backend_unadmitted"]
        == audit["cutoff_client_requests"]
        and audit["strict_client_requests"] + audit["cutoff_backend_admitted"]
        == audit["audited_backend_requests"]
    )


def measurement_window_sec(leg: Any) -> float | None:
    """Return the leg's own scored-window length, or None when it is unusable.

    Window-relative gates must fail closed rather than fall back to a constant,
    so a leg that does not agree with itself about its scored window yields no
    limit at all.
    """

    if not isinstance(leg, dict):
        return None
    config = leg.get("profile_config")
    configured = (
        config.get("measurement_window_sec") if isinstance(config, dict) else None
    )
    if (
        not isinstance(configured, int)
        or isinstance(configured, bool)
        or configured <= 0
    ):
        return None
    window = leg.get("measurement_window")
    if isinstance(window, dict):
        observed = window.get("duration_sec")
        if (
            not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not math.isfinite(float(observed))
            or float(observed) != float(configured)
        ):
            return None
    return float(configured)


def _check(name: str, passed: bool, message: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "message": _bounded(message),
        "metrics": metrics,
    }


def canonical_hard_gate_result(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Add the canonical paired quality gates omitted by the screening runner."""
    checks: list[dict[str, Any]] = []

    raw_runner_reasons = evaluation.get("all_invalid_reasons")
    if not isinstance(raw_runner_reasons, list):
        runner_reasons = ["runner_evaluation_shape_invalid"]
    else:
        runner_reasons = []
        for reason in raw_runner_reasons:
            normalized = str(reason)
            runner_reasons.append(
                normalized
                if normalized in RUNNER_REASON_ALLOWLIST
                else "runner_validation_failed"
            )
        runner_reasons = sorted(set(runner_reasons))
    runner_valid = evaluation.get("valid") is True and not runner_reasons
    checks.append(
        _check(
            "protocol_identity_fixed_window_lifecycle",
            runner_valid,
            "complete frozen-runner protocol and identity validation"
            if runner_valid
            else "runner gates failed: " + ", ".join(map(str, runner_reasons[:12])),
            {"runner_invalid_reasons": [str(x) for x in runner_reasons[:24]]},
        )
    )

    baseline_audit_ok = fixed_window_backend_audit_valid(baseline.get("backend_audit"))
    candidate_audit_ok = fixed_window_backend_audit_valid(candidate.get("backend_audit"))
    checks.append(
        _check(
            "backend_one_to_one_audit",
            baseline_audit_ok and candidate_audit_ok,
            "strict one-to-one backend audit and cutoff reconciliation passed"
            if baseline_audit_ok and candidate_audit_ok
            else "backend request accounting failed",
            {
                "baseline_right_censor_reconciled": baseline_audit_ok,
                "candidate_right_censor_reconciled": candidate_audit_ok,
            },
        )
    )

    baseline_success = _finite(baseline.get("request_success_rate"))
    candidate_success = _finite(candidate.get("request_success_rate"))
    success_floor = max(
        MIN_REQUEST_SUCCESS,
        (baseline_success - REQUEST_SUCCESS_TOLERANCE)
        if baseline_success is not None
        else math.inf,
    )
    success_ok = (
        baseline_success is not None
        and baseline_success >= MIN_REQUEST_SUCCESS
        and candidate_success is not None
        and candidate_success >= success_floor
    )
    checks.append(
        _check(
            "paired_request_success",
            success_ok,
            f"candidate={candidate_success!r}, required>={success_floor:.6f}",
            {
                "baseline": baseline_success,
                "candidate": candidate_success,
                "floor": success_floor if math.isfinite(success_floor) else None,
            },
        )
    )

    baseline_ratio = _finite(baseline.get("valid_step_ratio"))
    candidate_ratio = _finite(candidate.get("valid_step_ratio"))
    ratio_floor = max(
        MIN_VALID_STEP_RATIO,
        (baseline_ratio - VALID_STEP_RATIO_TOLERANCE)
        if baseline_ratio is not None
        else math.inf,
    )
    ratio_ok = (
        baseline_ratio is not None
        and baseline_ratio >= MIN_VALID_STEP_RATIO
        and candidate_ratio is not None
        and candidate_ratio >= ratio_floor
    )
    checks.append(
        _check(
            "paired_valid_step_ratio",
            ratio_ok,
            f"candidate={candidate_ratio!r}, required>={ratio_floor:.6f}",
            {
                "baseline": baseline_ratio,
                "candidate": candidate_ratio,
                "floor": ratio_floor if math.isfinite(ratio_floor) else None,
            },
        )
    )

    candidate_audit = candidate.get("backend_audit")
    candidate_wait = _finite(
        candidate_audit.get("p99_gateway_wait_sec")
        if isinstance(candidate_audit, dict)
        else None
    )
    window_sec = measurement_window_sec(candidate)
    wait_maximum = (
        P99_GATEWAY_WAIT_WINDOW_FRACTION * window_sec
        if window_sec is not None
        else None
    )
    wait_ok = (
        wait_maximum is not None
        and candidate_wait is not None
        and 0.0 <= candidate_wait <= wait_maximum
    )
    checks.append(
        _check(
            "candidate_p99_gateway_wait",
            wait_ok,
            (
                f"candidate={candidate_wait!r}s, maximum="
                f"{wait_maximum:.1f}s "
                f"({P99_GATEWAY_WAIT_WINDOW_FRACTION:.2f} x {window_sec:g}s "
                "scored window)"
                if wait_maximum is not None
                else "the leg declares no scored measurement window"
            ),
            {
                "candidate_sec": candidate_wait,
                "maximum_sec": wait_maximum,
                "measurement_window_sec": window_sec,
                "window_fraction": P99_GATEWAY_WAIT_WINDOW_FRACTION,
            },
        )
    )

    baseline_work = _finite(baseline.get("backend_tokens_per_valid_step"))
    candidate_work = _finite(candidate.get("backend_tokens_per_valid_step"))
    work_ratio = (
        candidate_work / baseline_work
        if candidate_work is not None and baseline_work is not None and baseline_work > 0
        else None
    )
    work_ok = (
        work_ratio is not None
        and work_ratio >= BACKEND_WORK_RATIO_MIN
    )
    checks.append(
        _check(
            "paired_backend_work",
            work_ok,
            (
                f"candidate/direct tokens-per-valid-step={work_ratio:.6f}, "
                f"required >= {BACKEND_WORK_RATIO_MIN:.2f}"
                if work_ratio is not None
                else "backend work ratio is unavailable"
            ),
            {
                "baseline_tokens_per_valid_step": baseline_work,
                "candidate_tokens_per_valid_step": candidate_work,
                "ratio": work_ratio,
                "minimum": BACKEND_WORK_RATIO_MIN,
                "maximum": BACKEND_WORK_RATIO_MAX,
            },
        )
    )

    direct_rate = _finite(baseline.get("valid_steps_per_min"))
    if direct_rate is None or direct_rate <= 0.0:
        raise RuntimeError("frozen Direct rate is unavailable")
    candidate_rate = _finite(candidate.get("valid_steps_per_min"))
    evaluation_rate = _finite(evaluation.get("candidate_valid_steps_per_min"))
    rate_ok = (
        candidate_rate is not None
        and candidate_rate > 0.0
        and evaluation_rate is not None
        and math.isclose(candidate_rate, evaluation_rate, rel_tol=0.0, abs_tol=1e-12)
    )
    checks.append(
        _check(
            "throughput_accounting",
            rate_ok,
            f"candidate valid_steps_per_min={candidate_rate!r}",
            {"candidate_valid_steps_per_min": candidate_rate},
        )
    )

    reasons = [check["name"] for check in checks if not check["passed"]]
    valid = not reasons
    score = candidate_rate if valid and candidate_rate is not None else 0.0
    speedup = score / direct_rate if valid else None
    return {
        "schema_version": 1,
        "valid": valid,
        "score": score,
        "primary_score_name": "valid_steps_per_min",
        "direct_valid_steps_per_min": direct_rate,
        "speedup_over_frozen_direct": speedup,
        "change_percent": (speedup - 1.0) * 100.0 if speedup is not None else None,
        "strictly_better_than_frozen_direct": bool(valid and score > direct_rate),
        "reasons": reasons,
        "checks": checks,
    }


def structured_result(report: dict[str, Any]) -> dict[str, Any]:
    edge = report.get("mlsysforge_evaluation")
    if not isinstance(edge, dict):
        return infrastructure_failure("trusted runner did not produce an MLSys Forge evaluation")
    checks = edge.get("checks")
    if (
        not isinstance(checks, list)
        or len(checks) != len(HARD_GATE_NAMES)
        or any(
            not isinstance(check, dict)
            or not isinstance(check.get("name"), str)
            or not isinstance(check.get("passed"), bool)
            or not isinstance(check.get("metrics"), dict)
            for check in checks
        )
    ):
        return infrastructure_failure("MLSys Forge hard-gate result is malformed")
    names = [check["name"] for check in checks]
    if len(set(names)) != len(names) or set(names) != HARD_GATE_NAMES:
        return infrastructure_failure("MLSys Forge hard-gate identities are malformed")
    all_passed = all(check["passed"] is True for check in checks)
    if not isinstance(edge.get("valid"), bool) or edge["valid"] is not all_passed:
        return infrastructure_failure("MLSys Forge hard-gate validity is inconsistent")
    raw_score = _finite(edge.get("score"))
    direct_rate = _finite(edge.get("direct_valid_steps_per_min"))
    if (
        raw_score is None
        or direct_rate is None
        or direct_rate <= 0.0
        or raw_score < 0.0
        or (all_passed and raw_score <= 0.0)
        or (not all_passed and raw_score != 0.0)
    ):
        return infrastructure_failure("MLSys Forge primary score is inconsistent")
    valid = all_passed
    score = raw_score
    passed = sum(1 for check in checks if isinstance(check, dict) and check.get("passed") is True)
    total = len(checks)
    details = [
        {
            "name": f"hard_gate:{check.get('name', 'unknown')}",
            "status": "PASSED" if check.get("passed") is True else "FAILED",
            # Agent-visible feedback is an allowlist. Exact aggregate quality
            # values stay in the private evidence because they can reveal the
            # concealed schedule when combined across repeated submissions.
            "message": (
                "required hard gate passed"
                if check.get("passed") is True
                else "required hard gate failed"
            ),
            "score": 1.0 if check.get("passed") is True else 0.0,
            "weight": 1.0,
        }
        for check in checks
        if isinstance(check, dict)
    ]
    speedup = _finite(edge.get("speedup_over_frozen_direct"))
    change = _finite(edge.get("change_percent"))
    reasons = [
        str(check["name"])
        for check in checks
        if check.get("passed") is not True
    ]
    if valid:
        relation = "above" if score > direct_rate else "at or below"
        summary = (
            f"VALID: primary score valid_steps_per_min={score:.6f} (maximize), "
            f"{relation} frozen Direct={direct_rate:.1f}; all {total} hard gates passed. "
            "A below-Direct result remains valid and is ranked by raw throughput."
        )
    else:
        summary = (
            "INVALID: primary score forced to 0 because required hard gates failed "
            f"({', '.join(map(str, reasons[:8])) or 'unknown'}); progress={passed}/{total}."
        )
    metrics: dict[str, Any] = {
        "source": "kvbench_candidate_only_vs_serving_contract_direct",
        "evaluation_contract": "candidate_only_same_nonce_same_serving_contract",
        "primary_score_name": "valid_steps_per_min",
        "primary_score_value": score,
        "primary_score_unit": "valid executable agent steps/min",
        "primary_score_direction": "maximize",
        "correctness_gate_mode": "binary",
        "correctness_gate_passed": valid,
        "frozen_direct_valid_steps_per_min": direct_rate,
        "speedup_over_frozen_direct": speedup,
        "change_percent": change,
        "strictly_better_than_frozen_direct": edge.get(
            "strictly_better_than_frozen_direct"
        )
        is True,
        "hard_gates_passed": passed,
        "hard_gates_total": total,
        "hard_gate_failures": [str(x) for x in reasons[:24]],
        "evaluation_status": "valid" if valid else "candidate_invalid",
        "auto_feedback_safe": True,
    }
    return {
        "valid": valid,
        "score": score,
        "pass_rate": passed / total if total else 0.0,
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "errors": 0,
        "summary": _bounded(summary, 1600),
        "details": details,
        "metrics": metrics,
    }


def infrastructure_failure(
    message: str, *, classification: str = "infrastructure_error"
) -> dict[str, Any]:
    del message
    if classification not in {"infrastructure_error", "resource_busy", "cancelled"}:
        classification = "infrastructure_error"
    public_message = (
        "trusted evaluation infrastructure did not produce a complete result; "
        "the private artifact contains the operator diagnostic"
    )
    return {
        "valid": False,
        "score": 0.0,
        "pass_rate": 0.0,
        "total_tests": 1,
        "passed": 0,
        "failed": 0,
        "errors": 1,
        "summary": "JUDGE INFRASTRUCTURE ERROR: " + public_message,
        "details": [
            {
                "name": "judge:infrastructure",
                "status": "ERROR",
                "message": public_message,
                "score": 0.0,
                "weight": 1.0,
            }
        ],
        "metrics": {
            "source": "kvbench_mlsysforge_adapter",
            "primary_score_name": "valid_steps_per_min",
            "primary_score_value": 0.0,
            "primary_score_direction": "maximize",
            "correctness_gate_passed": False,
            "evaluation_status": classification,
            "auto_feedback_safe": False,
        },
    }


def _validate_and_copy_submission(workspace: Path, destination: Path) -> str:
    workspace = workspace.resolve(strict=True)
    if not workspace.is_dir() or workspace.is_symlink():
        raise SubmissionError("submission_not_directory")
    count = 0
    source_bytes = 0
    for path in workspace.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise SubmissionError("submission_symlink")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise SubmissionError("submission_special_file")
        count += 1
        source_bytes += path.stat().st_size
        if count > MAX_FILES or source_bytes > MAX_SOURCE_BYTES:
            raise SubmissionError("submission_too_large")
    launch = workspace / "launch.sh"
    if not launch.is_file() or launch.is_symlink() or not os.access(launch, os.X_OK):
        raise SubmissionError("launch_not_executable")
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("trusted candidate scratch is not empty")
    shutil.copytree(workspace, destination, symlinks=False, copy_function=shutil.copy2)
    observed = tree_sha256(destination)
    if observed != tree_sha256(workspace):
        raise RuntimeError("candidate tree changed during trusted copy")
    return observed


def _normalize_baseline_provenance(provenance: Any) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise RuntimeError("frozen Direct serving contract is invalid")
    projected = {key: provenance.get(key) for key in MODEL_PROVENANCE_FIELDS}
    profile_id = projected.get("profile_id")
    profile_sha = projected.get("profile_sha256")
    if (
        not isinstance(profile_id, str)
        or not profile_id
        or not isinstance(profile_sha, str)
        or len(profile_sha) != 64
        or any(character not in "0123456789abcdef" for character in profile_sha)
        or projected.get("model_revision") != EXPECTED_MODEL_REVISION
        or projected.get("runtime_image_id")
        not in (EXPECTED_RUNTIME_IMAGE_ID, LEGACY_RUNTIME_IMAGE_ID)
        or projected.get("tensor_parallel_size") != 2
        or projected.get("kv_cache_dtype") != "fp8"
    ):
        raise RuntimeError("frozen Direct serving contract is invalid")
    digest = provenance.get("backend_serving_contract_sha256")
    if digest is None:
        # Evidence predating the contract digest can only be the v1 boot.
        if projected != LEGACY_MODEL_PROVENANCE:
            raise RuntimeError("frozen Direct legacy serving provenance is invalid")
        digest = LEGACY_BACKEND_SERVING_CONTRACT_SHA256
    elif digest not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256:
        raise RuntimeError("frozen Direct backend serving contract is invalid")
    # The image identity has to be the one its revision defines: the recipe
    # digest under v3, the qualification host's image ID under v1 / v2.5.
    expected_image = (
        EXPECTED_RUNTIME_IMAGE_ID
        if digest == EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
        else LEGACY_RUNTIME_IMAGE_ID
    )
    if projected.get("runtime_image_id") != expected_image:
        raise RuntimeError("frozen Direct runtime image identity is invalid")
    # Keep the revision the anchor was actually measured under instead of
    # restamping it with the current one.
    projected["backend_serving_contract_sha256"] = digest
    return projected


def _load_frozen_baseline_bundle() -> dict[str, Any]:
    if (
        not BASELINE_PATH.is_file()
        or BASELINE_PATH.is_symlink()
        or BASELINE_PATH.stat().st_size > MAX_TRUSTED_JSON_BYTES
    ):
        raise RuntimeError("frozen Direct evidence is unavailable")
    raw = BASELINE_PATH.read_bytes()
    if not raw or len(raw) > MAX_TRUSTED_JSON_BYTES:
        raise RuntimeError("frozen Direct evidence is unavailable")
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    report = _strict_json_object_bytes(raw, description="frozen Direct evidence")
    baseline = (report.get("legs") or {}).get("baseline")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "complete"
        or report.get("suite") != "hidden"
        or report.get("profile") != "hidden-pressure-192"
        or report.get("workload_nonce_sha256")
        != EXPECTED_NONCE_COMMITMENT_SHA256
        or not isinstance(baseline, dict)
    ):
        raise RuntimeError("frozen Direct evidence has no baseline leg")
    direct_rate = _finite(baseline.get("valid_steps_per_min"))
    if direct_rate is None or direct_rate <= 0.0:
        raise RuntimeError("frozen Direct serving contract is invalid")
    normalized_provenance = _normalize_baseline_provenance(
        baseline.get("model_provenance")
    )
    normalized = copy.deepcopy(baseline)
    normalized["model_provenance"] = normalized_provenance
    hidden_baseline_identity = {
        "file_sha256": observed_sha256,
        "valid_steps_per_min": direct_rate,
        "backend_serving_contract_sha256": normalized_provenance[
            "backend_serving_contract_sha256"
        ],
        "workload_nonce_sha256": report["workload_nonce_sha256"],
    }
    if not _valid_hidden_baseline_identity(hidden_baseline_identity):
        raise RuntimeError("frozen Direct identity is invalid")
    return {
        "path": str(BASELINE_PATH),
        "hidden_baseline_identity": hidden_baseline_identity,
        "leg": normalized,
    }


def _load_frozen_baseline() -> dict[str, Any]:
    """Compatibility helper used by unit tests and the canonical hard gates."""

    return _load_frozen_baseline_bundle()["leg"]


def _clear_host_scratch(raw_scratch: str) -> None:
    """Let trusted Judge root remove files it created through the host bind."""
    if not raw_scratch:
        return
    raw_root = os.environ.get("KVBENCH_EVAL_ROOT", "")
    if not raw_root:
        raise RuntimeError("trusted Judge eval root is not configured")
    root = Path(raw_root).resolve(strict=True)
    scratch = Path(raw_scratch).resolve(strict=True)
    if (
        scratch.parent != root
        or not scratch.name.startswith("judge-")
        or scratch.is_symlink()
    ):
        raise RuntimeError("refusing unsafe trusted scratch cleanup")
    for child in scratch.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            raise RuntimeError("trusted scratch contains a special file")


def validate_public_result(result: dict[str, Any], classification: str) -> None:
    """Reject malformed structured output before SForge can grade it."""

    valid = result.get("valid")
    score = _finite(result.get("score"))
    pass_rate = _finite(result.get("pass_rate"))
    counts = [result.get(key) for key in ("total_tests", "passed", "failed", "errors")]
    details = result.get("details")
    metrics = result.get("metrics")
    if (
        not isinstance(valid, bool)
        or score is None
        or score < 0.0
        or pass_rate is None
        or not 0.0 <= pass_rate <= 1.0
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        )
        or counts[0] <= 0
        or counts[1] + counts[2] + counts[3] != counts[0]
        or not math.isclose(pass_rate, counts[1] / counts[0], rel_tol=0.0, abs_tol=1e-12)
        or not isinstance(details, list)
        or len(details) != counts[0]
        or not isinstance(metrics, dict)
        or metrics.get("evaluation_status") != classification
        or metrics.get("primary_score_name") != "valid_steps_per_min"
        or metrics.get("primary_score_direction") != "maximize"
        or metrics.get("correctness_gate_passed") is not valid
        or _finite(metrics.get("primary_score_value")) != score
        or (valid and score <= 0.0)
        or (not valid and score != 0.0)
    ):
        raise RuntimeError("public structured result contract failed")
    names: list[str] = []
    for detail in details:
        if (
            not isinstance(detail, dict)
            or not isinstance(detail.get("name"), str)
            or detail.get("status") not in {"PASSED", "FAILED", "ERROR"}
        ):
            raise RuntimeError("public structured detail contract failed")
        names.append(detail["name"])
    if len(names) != len(set(names)):
        raise RuntimeError("public structured details contain duplicates")
    if classification in {"valid", "candidate_invalid"} and counts[0] == len(HARD_GATE_NAMES):
        if set(names) != {f"hard_gate:{name}" for name in HARD_GATE_NAMES}:
            raise RuntimeError("public hard-gate detail identities are incomplete")
        if valid is not (classification == "valid"):
            raise RuntimeError("public classification and validity disagree")


def _assert_frozen_baseline_snapshot_current(
    baseline_bundle: dict[str, Any],
) -> None:
    identity = baseline_bundle.get("hidden_baseline_identity")
    raw_path = baseline_bundle.get("path")
    if not _valid_hidden_baseline_identity(identity) or not isinstance(raw_path, str):
        raise RuntimeError("frozen Direct snapshot identity is unavailable")
    path = Path(raw_path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("frozen Direct snapshot path drifted")
    observed_sha256 = file_sha256(path)
    if observed_sha256 != identity["file_sha256"]:
        raise RuntimeError("frozen Direct snapshot bytes drifted")


def _write_private_envelope(
    path: Path,
    *,
    started_at_utc: str,
    classification: str,
    candidate_tree_sha256: str | None,
    runner: dict[str, Any],
    runner_evidence: dict[str, Any] | None,
    public_result: dict[str, Any],
    private_errors: list[str],
    baseline_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hidden_baseline_identity: dict[str, Any] | None = None
    if baseline_bundle is not None:
        candidate_identity = baseline_bundle.get("hidden_baseline_identity")
        if not _valid_hidden_baseline_identity(candidate_identity):
            raise RuntimeError("frozen Direct snapshot identity is unavailable")
        hidden_baseline_identity = copy.deepcopy(candidate_identity)
    envelope = {
        "schema_version": 2,
        "kind": "kvbench-mlsysforge-private-evaluation",
        "status": "terminal",
        "classification": classification,
        "started_at_utc": started_at_utc,
        "completed_at_utc": utc_now(),
        "candidate_tree_sha256": candidate_tree_sha256,
        "commitments": {
            "hidden_baseline_identity": hidden_baseline_identity,
        },
        "runner": runner,
        "runner_evidence": runner_evidence,
        "cleanup": (
            runner_evidence.get("cleanup")
            if isinstance(runner_evidence, dict)
            else None
        ),
        "mlsysforge_evaluation": (
            runner_evidence.get("mlsysforge_evaluation")
            if isinstance(runner_evidence, dict)
            else None
        ),
        "public_result": public_result,
        "public_result_sha256": canonical_json_sha256(public_result),
        "private_errors": private_errors,
    }
    _atomic_write_json(path, envelope)
    return envelope


def _force_runtime_reconcile(migration_root: Path, candidate_dir: Path) -> dict[str, Any]:
    stack = (
        migration_root
        / "runtime/frozen/harbor_run_scripts/"
        "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/"
        "gpu67-v2.1-validation-stack.sh"
    )
    if not stack.is_file() or stack.is_symlink():
        return {"attempted": False, "succeeded": False, "reason": "stack_missing"}
    timeout = int(
        os.environ.get(
            "KVBENCH_FORCE_RECONCILE_TIMEOUT_SEC",
            str(DEFAULT_FORCE_RECONCILE_TIMEOUT_SEC),
        )
    )
    environment = {
        **os.environ,
        "KV_AUTHOR_REFERENCE_GATEWAY_DIR": str(candidate_dir),
        "KV_GPU67_V21_OWNER_TOKEN": "gpu67-v2-2-candidate-only",
    }
    try:
        completed = subprocess.run(
            [str(stack), "stop"],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"attempted": True, "succeeded": False, "reason": "timeout"}
    return {
        "attempted": True,
        "succeeded": completed.returncode == 0,
        "returncode": completed.returncode,
        "reason": "complete" if completed.returncode == 0 else "nonzero_exit",
    }


def _run_supervised(
    command: list[str],
    *,
    environment: dict[str, str],
    runner_log: Path,
) -> dict[str, Any]:
    """Run the hidden evaluator with bounded cooperative teardown."""

    timeout = int(
        os.environ.get("KVBENCH_RUNNER_TIMEOUT_SEC", str(DEFAULT_RUNNER_TIMEOUT_SEC))
    )
    cleanup_grace = int(
        os.environ.get(
            "KVBENCH_RUNNER_CLEANUP_GRACE_SEC",
            str(DEFAULT_RUNNER_CLEANUP_GRACE_SEC),
        )
    )
    if timeout <= 0 or cleanup_grace <= 0 or timeout + cleanup_grace >= 2990:
        raise RuntimeError("trusted runner timeout budget is unsafe")

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
        os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_CLOEXEC,
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
                if not termination_sent and (cancel_signal is not None or now >= deadline):
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
    started_at_utc = utc_now()
    log_dir = args.log_dir.resolve(strict=False)
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    private_path = log_dir / "kvbench-private-evidence.json"
    host_scratch_raw = os.environ.get("KVBENCH_HOST_EVAL_DIR", "")
    migration_raw = os.environ.get("KVBENCH_MIGRATION_ROOT", "")
    nonce_source_raw = os.environ.get("KVBENCH_HIDDEN_NONCE_SOURCE", "")
    candidate_sha256: str | None = None
    runner_evidence: dict[str, Any] | None = None
    baseline_bundle: dict[str, Any] | None = None
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
    classification = "infrastructure_error"
    public_result = infrastructure_failure("evaluation did not start")
    host_scratch: Path | None = None
    migration_root: Path | None = None
    candidate_dir: Path | None = None
    try:
        if not host_scratch_raw or not migration_raw or not nonce_source_raw:
            raise RuntimeError("trusted Judge host mounts are not configured")
        host_scratch = Path(host_scratch_raw).resolve(strict=True)
        migration_root = Path(migration_raw).resolve(strict=True)
        nonce_source = Path(nonce_source_raw).resolve(strict=True)
        if not host_scratch.is_dir() or host_scratch.is_symlink():
            raise RuntimeError("trusted Judge scratch is unsafe")
        if not nonce_source.is_file() or nonce_source.is_symlink():
            raise RuntimeError("hidden nonce source is unsafe")
        baseline_bundle = _load_frozen_baseline_bundle()
        candidate_dir = host_scratch / "candidate"
        evidence_path = host_scratch / "evidence.json"
        candidate_sha256 = _validate_and_copy_submission(args.workspace, candidate_dir)
        runner = (
            migration_root
            / "runtime/frozen/harbor_run_scripts/"
            "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.2-gpu45/"
            "run-gpu67-v2.2-hidden-candidate-only.py"
        )
        if not runner.is_file() or runner.is_symlink():
            raise RuntimeError("frozen Candidate runner is unavailable")
        command = [
            sys.executable,
            str(runner),
            "--candidate-dir",
            str(candidate_dir),
            "--expected-candidate-tree-sha256",
            candidate_sha256,
            "--hidden-workload-nonce-source",
            str(nonce_source),
            "--output",
            str(evidence_path),
        ]
        runner_metadata = _run_supervised(
            command,
            environment={**os.environ},
            runner_log=log_dir / "runner.log",
        )
        if runner_metadata["forced_kill"] is True:
            runner_metadata["force_reconcile"] = _force_runtime_reconcile(
                migration_root, candidate_dir
            )
        if evidence_path.is_file() and not evidence_path.is_symlink():
            runner_evidence = _strict_json_object(evidence_path)
        if runner_metadata["cancel_signal"] is not None:
            classification = "cancelled"
            raise RuntimeError("trusted runner was cancelled")
        if runner_metadata["timed_out"] is True:
            raise RuntimeError("trusted runner timed out")
        if runner_metadata["forced_kill"] is True:
            raise RuntimeError("trusted runner exceeded cleanup grace")
        if runner_evidence is None:
            raise RuntimeError("trusted runner exited without evidence")
        if runner_metadata["returncode"] not in (0, 3):
            if runner_evidence.get("status") == "resource_busy":
                classification = "resource_busy"
            raise RuntimeError("trusted runner returned a non-terminal exit")
        candidate, evaluation = _validate_complete_runner_report(
            runner_evidence, candidate_sha256, baseline_bundle
        )
        baseline = baseline_bundle["leg"]
        edge = canonical_hard_gate_result(baseline, candidate, evaluation)
        runner_evidence["mlsysforge_evaluation"] = edge
        runner_evidence["mlsysforge_adapter"] = {
            "schema_version": 1,
            "candidate_only": True,
            "fresh_paired_ab": False,
            "canonical_additional_hard_gates_applied": True,
            "primary_score": "candidate_valid_steps_per_min",
        }
        public_result = structured_result(runner_evidence)
        classification = "valid" if public_result.get("valid") is True else "candidate_invalid"
        validate_public_result(public_result, classification)
    except SubmissionError as exc:
        classification = "candidate_invalid"
        private_errors.append(f"SubmissionError: {exc}")
        public_result = submission_failure(str(exc))
    except BaseException as exc:
        private_errors.append(f"{type(exc).__name__}: {exc}")
        public_result = infrastructure_failure(
            "trusted evaluator did not complete", classification=classification
        )
    finally:
        if host_scratch is not None:
            try:
                _clear_host_scratch(host_scratch_raw)
            except BaseException as cleanup_exc:
                classification = "infrastructure_error"
                private_errors.append(
                    f"scratch cleanup: {type(cleanup_exc).__name__}: {cleanup_exc}"
                )
                public_result = infrastructure_failure("trusted scratch cleanup failed")

        if baseline_bundle is not None:
            try:
                _assert_frozen_baseline_snapshot_current(baseline_bundle)
            except BaseException as drift_exc:
                classification = "infrastructure_error"
                private_errors.append(
                    "baseline drift: "
                    f"{type(drift_exc).__name__}: {drift_exc}"
                )
                public_result = infrastructure_failure(
                    "frozen Direct snapshot changed during evaluation"
                )

        try:
            validate_public_result(public_result, classification)
        except BaseException as validation_exc:
            classification = "infrastructure_error"
            private_errors.append(
                f"public validation: {type(validation_exc).__name__}: {validation_exc}"
            )
            public_result = infrastructure_failure("structured output validation failed")
            validate_public_result(public_result, classification)

        _write_private_envelope(
            private_path,
            started_at_utc=started_at_utc,
            classification=classification,
            candidate_tree_sha256=candidate_sha256,
            runner=runner_metadata,
            runner_evidence=runner_evidence,
            public_result=public_result,
            private_errors=private_errors,
            baseline_bundle=baseline_bundle,
        )

    _emit(public_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
