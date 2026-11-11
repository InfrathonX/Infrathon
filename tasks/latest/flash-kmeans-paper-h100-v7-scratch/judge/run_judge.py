#!/usr/bin/env python3
"""Run the private verifier and emit one redacted MLSys Forge record."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


MARKER_START = ">>>>> Start Structured Result"
MARKER_END = ">>>>> End Structured Result"
RESULT_FILES = ("reward.json", "reward.txt", "flash_kmeans_results.json")
ADAPTER_SCHEMA_VERSION = 8
FEEDBACK_SCHEMA_VERSION = 7
GATES = (
    ("protected_ok", "Protected workspace"),
    ("source_ok", "Clean-room source policy"),
    ("interface_ok", "Candidate interface"),
    ("correctness", "Functional correctness"),
    ("input_integrity_ok", "Input integrity"),
    ("memory_ok", "Streaming peak memory"),
    ("timing_protocol_ok", "Synchronized timing protocol"),
    ("deployment_protocol_ok", "Fresh-process deployment protocol"),
)
SAFE_NUMERIC_KEYS = (
    "reward",
    "score",
    "weighted_latency_ms",
    "diagnostic_score",
    "diagnostic_weighted_latency_ms",
    "first_run_latency_ms",
    "diagnostic_first_run_latency_ms",
    "warmed_iteration_latency_ms",
    "diagnostic_warmed_iteration_latency_ms",
    "runtime_weighted_latency_ms",
    "diagnostic_runtime_weighted_latency_ms",
    "assign_latency_ms",
    "diagnostic_assign_latency_ms",
    "update_latency_ms",
    "diagnostic_update_latency_ms",
    "full_latency_ms",
    "diagnostic_full_latency_ms",
    "compile_overhead_ratio",
    "diagnostic_compile_overhead_ratio",
    "primary_score_value",
    "final_version",
    "requested_final_version",
    "evaluated_final_version",
    "num_candidate_versions",
    *(name for name, _label in GATES),
)
ACCEPTED_ONLY_NUMERIC_KEYS = frozenset(
    {
        "reward",
        "score",
        "weighted_latency_ms",
        "first_run_latency_ms",
        "warmed_iteration_latency_ms",
        "runtime_weighted_latency_ms",
        "assign_latency_ms",
        "update_latency_ms",
        "full_latency_ms",
        "compile_overhead_ratio",
        "primary_score_value",
    }
)
STAGE_LABELS = {
    "assign": "assignment",
    "update": "centroid update",
    "full": "combined iteration",
}
GATE_ORDER = tuple(name for name, _label in GATES)
GATE_LABELS = dict(GATES)
MEMORY_FEEDBACK_BUCKET = 0.05
DEPLOYMENT_FAMILY_BAND_RATIO = 1.25
DEPLOYMENT_FAMILY_METRICS = {
    "large-K": "coarse_deployment_large_k_latency_ms",
    "large-N/high-D": "coarse_deployment_large_n_high_d_latency_ms",
}
DEPLOYMENT_FAMILY_STATUS_KEYS = {
    "large-K": "large_k",
    "large-N/high-D": "large_n_high_d",
}
ALLOWED_DEPLOYMENT_FAMILY_STATUSES = frozenset(
    {"passed", "failed", "not_reached"}
)


def selected_version(
    reward_data: Mapping[str, Any],
    detail_data: Mapping[str, Any],
) -> int | None:
    value = metric(reward_data, detail_data, "final_version")
    if value is None or not value.is_integer():
        return None
    version = int(value)
    return version if 1 <= version <= 999 else None


def version_label(version: int | None) -> str:
    return f"v{version:03d}" if version is not None else "the selected candidate"


def add_diagnosis(
    diagnoses: dict[str, dict[str, set[str]]],
    gate: str,
    stage: str,
    reason: str,
) -> None:
    gate_diagnoses = diagnoses.setdefault(
        gate,
        {"stages": set(), "reasons": set()},
    )
    gate_diagnoses["stages"].add(stage)
    gate_diagnoses["reasons"].add(reason)


def safe_failure_diagnoses(
    detail_data: Mapping[str, Any],
) -> dict[str, dict[str, set[str]]]:
    """Extract only allowlisted failure classes; never disclose private cases."""

    diagnoses: dict[str, dict[str, set[str]]] = {}
    protected = finite_number(detail_data.get("protected_ok"))
    source = finite_number(detail_data.get("source_ok"))
    interface = finite_number(detail_data.get("interface_ok"))
    if protected is not None and protected < 1.0:
        add_diagnosis(
            diagnoses,
            "protected_ok",
            "submission",
            "the protected workspace check failed",
        )
    elif source is not None and source < 1.0:
        add_diagnosis(
            diagnoses,
            "source_ok",
            "submission",
            "source admission rejected the submitted files",
        )
    elif interface is not None and interface < 1.0:
        add_diagnosis(
            diagnoses,
            "interface_ok",
            "candidate import",
            "the selected module did not load with the required callable ABI",
        )

    raw_cases = detail_data.get("cases")
    if not isinstance(raw_cases, list):
        return diagnoses

    for raw_row in raw_cases:
        if not isinstance(raw_row, Mapping):
            continue
        raw_stage = raw_row.get("stage")
        stage = STAGE_LABELS.get(raw_stage) if isinstance(raw_stage, str) else None
        if stage is None:
            continue
        raw_measurement_mode = raw_row.get("measurement_mode")
        if raw_measurement_mode == "fresh_process":
            stage = f"fresh-process {stage}"
        raw_error = raw_row.get("error")
        error = raw_error.lower() if isinstance(raw_error, str) else ""

        if (
            "input tensor was mutated" in error
            or "input integrity" in error
        ):
            add_diagnosis(
                diagnoses,
                "input_integrity_ok",
                stage,
                "an input tensor was modified",
            )
        elif (
            "streaming-memory budget" in error
            or "out of memory" in error
        ):
            add_diagnosis(
                diagnoses,
                "memory_ok",
                stage,
                "the bounded-memory requirement was exceeded",
            )
        elif (
            "timing trials remained unstable" in error
            or "non-positive/non-finite latency" in error
        ):
            add_diagnosis(
                diagnoses,
                "timing_protocol_ok",
                stage,
                "synchronized timing did not produce a stable valid latency",
            )
        else:
            correctness_reason: str | None = None
            if "exact tie did not choose" in error:
                correctness_reason = "exact ties did not choose the smallest cluster id"
            elif "meaningfully farther centroid" in error:
                correctness_reason = "nearest-centroid FP32 distance ordering was wrong"
            elif "cluster counts are not exact" in error:
                correctness_reason = "cluster counts were not exact"
            elif "empty cluster did not retain" in error:
                correctness_reason = "an empty cluster did not retain its old centroid"
            elif "centroid error exceeds" in error:
                correctness_reason = "FP32 centroid accumulation exceeded tolerance"
            elif (
                "wrong shape" in error
                or "shape mismatch" in error
                or "must be on cuda" in error
                or "must have an integer dtype" in error
                or "must be fp16" in error
                or "out-of-range cluster id" in error
                or "must return" in error
                or "outputs must be tensors" in error
                or "invalid internal output" in error
            ):
                correctness_reason = "the output ABI or value range was invalid"
            elif (
                "correctness" in error
                or "full-stage" in error
                or (
                    raw_measurement_mode != "fresh_process"
                    and raw_row.get("correctness_ok") is False
                    and bool(error)
                )
            ):
                correctness_reason = "the numerical output was incorrect"
            if correctness_reason is not None:
                add_diagnosis(
                    diagnoses,
                    "correctness",
                    stage,
                    correctness_reason,
                )

        if (
            raw_measurement_mode == "fresh_process"
            and raw_row.get("deployment_protocol_ok") is not True
        ):
            if (
                raw_row.get("candidate_initial_cache_empty") is False
                and "cache" in error
            ):
                deployment_reason = (
                    "the isolated deployment cache was not empty"
                )
            elif "timeout" in error:
                deployment_reason = (
                    "the fresh-process run exceeded its time limit"
                )
            else:
                deployment_reason = (
                    "the candidate did not complete a valid isolated "
                    "first-run/warmed-iteration measurement"
                )
            add_diagnosis(
                diagnoses,
                "deployment_protocol_ok",
                stage,
                deployment_reason,
            )

    max_memory_ratio = finite_number(detail_data.get("max_memory_ratio"))
    raw_worst_stage = detail_data.get("worst_memory_stage")
    worst_stage = (
        STAGE_LABELS.get(raw_worst_stage)
        if isinstance(raw_worst_stage, str)
        else None
    )
    memory_gate = finite_number(detail_data.get("memory_ok"))
    if (
        memory_gate is not None
        and memory_gate < 1.0
        and max_memory_ratio is not None
        and max_memory_ratio > 1.0
    ):
        stage = worst_stage or "one hidden workload"
        bounds = memory_ratio_bounds(max_memory_ratio)
        assert bounds is not None
        lower, upper = bounds
        memory_diagnosis = diagnoses.setdefault(
            "memory_ok",
            {"stages": set(), "reasons": set()},
        )
        memory_diagnosis["stages"].add(stage)
        memory_diagnosis["reasons"] = {
            "the worst candidate-owned peak was in the redacted "
            f"[{lower:.2f}x, {upper:.2f}x) budget band "
            f"(less than {100.0 * max(0.0, upper - 1.0):.0f}% over)"
        }

    return diagnoses


def format_diagnosis(diagnosis: Mapping[str, set[str]]) -> str:
    stages = ", ".join(sorted(diagnosis.get("stages", set())))
    reasons = "; ".join(sorted(diagnosis.get("reasons", set())))
    if stages and reasons:
        return f"in {stages}: {reasons}"
    if stages:
        return f"in {stages}"
    return reasons


def failed_gate_summary(
    details: list[dict[str, Any]],
    diagnoses: Mapping[str, Mapping[str, set[str]]],
) -> str:
    fragments: list[str] = []
    for name in GATE_ORDER:
        detail = next(item for item in details if item["name"] == name)
        if detail["status"] != "FAILED":
            continue
        label = GATE_LABELS[name].lower()
        diagnosis = diagnoses.get(name)
        if diagnosis:
            fragments.append(f"{label} {format_diagnosis(diagnosis)}")
        else:
            fragments.append(label)
    return "; ".join(fragments)


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def memory_ratio_bounds(ratio: float | None) -> tuple[float, float] | None:
    """Quantize hidden memory feedback to avoid a private-shape side channel."""

    if ratio is None or not math.isfinite(ratio) or ratio < 0.0:
        return None
    lower = math.floor(ratio / MEMORY_FEEDBACK_BUCKET) * MEMORY_FEEDBACK_BUCKET
    upper = lower + MEMORY_FEEDBACK_BUCKET
    return round(lower, 10), round(upper, 10)


def deployment_latency_band(value: float | None) -> dict[str, float] | None:
    """Quantize an aggregate latency to a fixed 1.25x multiplicative band."""

    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    try:
        exponent = math.floor(
            math.log(value) / math.log(DEPLOYMENT_FAMILY_BAND_RATIO)
        )
        lower = DEPLOYMENT_FAMILY_BAND_RATIO**exponent
        upper = lower * DEPLOYMENT_FAMILY_BAND_RATIO
    except (OverflowError, ValueError):
        return None
    if not all(math.isfinite(bound) and bound > 0.0 for bound in (lower, upper)):
        return None
    return {
        "lower_ms": float(f"{lower:.6g}"),
        "upper_ms": float(f"{upper:.6g}"),
    }


def coarse_deployment_feedback(
    reward_data: Mapping[str, Any],
    detail_data: Mapping[str, Any],
    *,
    include_latency_bands: bool,
) -> dict[str, dict[str, Any]] | None:
    """Build the only allowed family view; invalid results are status-only."""

    raw_statuses = detail_data.get("coarse_deployment_family_statuses")
    if not isinstance(raw_statuses, Mapping) or set(raw_statuses) != set(
        DEPLOYMENT_FAMILY_STATUS_KEYS.values()
    ):
        return None
    feedback: dict[str, dict[str, Any]] = {}
    for family, metric_name in DEPLOYMENT_FAMILY_METRICS.items():
        private_key = DEPLOYMENT_FAMILY_STATUS_KEYS[family]
        status = raw_statuses.get(private_key)
        if status not in ALLOWED_DEPLOYMENT_FAMILY_STATUSES:
            return None
        entry: dict[str, Any] = {"status": status}
        if include_latency_bands:
            if status != "passed":
                return None
            band = deployment_latency_band(
                metric(reward_data, detail_data, metric_name)
            )
            if band is None:
                return None
            entry["warmed_latency_band_ms"] = band
            entry["trend_hint"] = (
                "compare this categorical band with earlier valid submissions; "
                "a lower band is an improvement"
            )
        feedback[family] = entry
    return feedback


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def prepare_log_dir(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir.chmod(0o700)
    for name in RESULT_FILES:
        path = log_dir / name
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
        except OSError:
            pass


def metric(
    reward_data: Mapping[str, Any],
    detail_data: Mapping[str, Any],
    name: str,
) -> float | None:
    value = finite_number(reward_data.get(name))
    return value if value is not None else finite_number(detail_data.get(name))


def gate_detail(
    name: str,
    label: str,
    value: float | None,
    diagnosis: Mapping[str, set[str]] | None,
    selected: str,
) -> dict[str, Any]:
    if value is None:
        status = "ERROR"
        message = f"{label} did not produce a trusted status."
    elif value >= 1.0:
        status = "PASSED"
        message = f"{label} passed."
    elif diagnosis:
        status = "FAILED"
        message = f"{selected} failed {label.lower()} {format_diagnosis(diagnosis)}."
    else:
        status = "ERROR"
        message = (
            f"{label} was not independently established for {selected}; "
            "an earlier hidden failure stopped at least one evaluation path."
        )
    return {
        "name": name,
        "status": status,
        "score": 1.0 if status == "PASSED" else 0.0,
        "message": message,
    }


def build_result(
    reward_data: Mapping[str, Any],
    detail_data: Mapping[str, Any],
    *,
    verifier_completed: bool,
    verifier_returncode: int | None,
) -> dict[str, Any]:
    raw_reward = finite_number(reward_data.get("reward"))
    result_available = verifier_completed and raw_reward is not None
    version = selected_version(reward_data, detail_data)
    selected = f"FINAL_VERSION={version_label(version)}"
    diagnoses = safe_failure_diagnoses(detail_data) if result_available else {}
    gate_values = {
        name: metric(reward_data, detail_data, name) if result_available else None
        for name, _label in GATES
    }
    all_gate_values_pass = result_available and all(
        value is not None and value >= 1.0 for value in gate_values.values()
    )
    gates_passed = (
        all_gate_values_pass
        and raw_reward is not None
        and raw_reward > 0.0
        and verifier_returncode == 0
    )
    gate_failed = (
        result_available
        and not all_gate_values_pass
        and verifier_returncode == 1
    )
    diagnostic_score = (
        metric(reward_data, detail_data, "diagnostic_score")
        if result_available
        else None
    )
    max_memory_ratio = (
        metric(reward_data, detail_data, "max_memory_ratio")
        if result_available
        else None
    )
    memory_gate = gate_values.get("memory_ok")
    memory_evidence_available = bool(
        memory_gate is not None
        and (
            memory_gate >= 1.0
            or (
                max_memory_ratio is not None
                and max_memory_ratio > 1.0
            )
        )
    )
    memory_bounds = (
        memory_ratio_bounds(max_memory_ratio)
        if memory_evidence_available
        else None
    )
    if gates_passed:
        outcome = "passed"
    elif gate_failed:
        outcome = "gate_failed"
    else:
        outcome = "judge_error"
    score = float(raw_reward) if gates_passed and raw_reward is not None else None
    deployment_feedback = (
        coarse_deployment_feedback(
            reward_data,
            detail_data,
            include_latency_bands=gates_passed,
        )
        if gates_passed or gate_failed
        else None
    )
    details = [
        gate_detail(
            name,
            label,
            gate_values[name],
            diagnoses.get(name),
            selected,
        )
        for name, label in GATES
    ]
    memory_detail = next(
        item for item in details if item["name"] == "memory_ok"
    )
    if (
        memory_detail["status"] == "PASSED"
        and memory_bounds is not None
    ):
        lower, upper = memory_bounds
        memory_detail["message"] = (
            "Streaming peak memory passed; the worst candidate-owned peak "
            f"was in the redacted [{lower:.2f}x, {upper:.2f}x) budget band."
        )
    details.append(
        {
            "name": "performance",
            "status": (
                "PASSED"
                if gates_passed
                else ("FAILED" if gate_failed else "ERROR")
            ),
            "score": score,
            "message": (
                "Candidate-only weighted latency accepted; lower is better."
                if gates_passed
                else (
                    "The Judge performance score was not accepted because "
                    "a hidden hard gate failed. "
                    + (
                        f"Diagnostic weighted latency {diagnostic_score:.6f} ms "
                        "is reported only to guide optimization."
                        if diagnostic_score is not None
                        and diagnostic_score > 0.0
                        else "No complete diagnostic latency score was available."
                    )
                    if gate_failed
                    else "Performance was not scored because the Judge result "
                    "protocol was incomplete or inconsistent."
                )
            ),
        }
    )
    passed_details = sum(item["status"] == "PASSED" for item in details)
    pass_rate = passed_details / len(details)
    diagnosis_summary = failed_gate_summary(details, diagnoses)
    rollback_action = (
        "Hidden invalidation overrides public confirmation: restore FINAL_VERSION "
        "to the best previously Judge-passing version. If no version has passed "
        "Judge yet, select the simplest locally correct version, keep this "
        "failing source deselected, and continue the repair in a new version."
    )
    if not result_available:
        summary = "K-means kernel verifier did not produce a valid result."
    elif gates_passed:
        assert score is not None
        family_fragment = ""
        if deployment_feedback is not None:
            bands = []
            for family in DEPLOYMENT_FAMILY_METRICS:
                band = deployment_feedback[family]["warmed_latency_band_ms"]
                bands.append(
                    f"{family} [{band['lower_ms']:g}, {band['upper_ms']:g}) ms"
                )
            family_fragment = " Coarse deployment feedback: " + "; ".join(bands) + "."
        summary = (
            f"{selected} passed every hidden hard gate; candidate-only "
            f"weighted latency {score:.6f} ms (lower is better)."
            f"{family_fragment}"
        )
    elif gate_failed:
        feedback = diagnosis_summary or "one or more required hidden gates failed"
        family_fragment = ""
        if deployment_feedback is not None:
            statuses = "; ".join(
                f"{family} {deployment_feedback[family]['status']}"
                for family in DEPLOYMENT_FAMILY_METRICS
            )
            family_fragment = f" Coarse deployment family status: {statuses}."
        diagnostic_fragment = (
            f" Diagnostic weighted latency: {diagnostic_score:.6f} ms "
            "(informational, not accepted)."
            if diagnostic_score is not None and diagnostic_score > 0.0
            else ""
        )
        summary = (
            f"{selected} is invalid. Hidden feedback: {feedback}."
            f"{family_fragment}{diagnostic_fragment} {rollback_action}"
        )
    else:
        summary = (
            f"{selected} produced an inconsistent Judge protocol result; keep "
            "the last known Judge-valid FINAL_VERSION selected."
        )

    metrics: dict[str, Any] = {
        "source": "flash_kmeans_h100_verifier_adapter",
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "feedback_schema_version": FEEDBACK_SCHEMA_VERSION,
        "evaluation_outcome": outcome,
        "candidate_output_suppressed": True,
        "result_protocol_records": 1,
        "verifier_completed": bool(verifier_completed),
        "verifier_returncode": verifier_returncode,
        "reward": score,
        "primary_score_name": "weighted_latency_ms",
        "primary_score_unit": "ms",
        "primary_score_direction": "minimize",
        "primary_score_formula": (
            "exp(sum_g(group_weight_g * log(group_GM_g)))"
        ),
        "primary_score_value": score,
    }
    if version is not None:
        metrics["selected_final_version"] = version_label(version)
        metrics["requested_final_version"] = version
    if gate_failed:
        metrics["rollback_required"] = True
        metrics["feedback_action"] = rollback_action
        metrics["failed_gates"] = [
            item["name"]
            for item in details
            if item["status"] == "FAILED" and item["name"] != "performance"
        ]
    else:
        metrics["rollback_required"] = False
    if result_available:
        for key in SAFE_NUMERIC_KEYS:
            if not gates_passed and key in ACCEPTED_ONLY_NUMERIC_KEYS:
                continue
            if (
                key.startswith("diagnostic_")
                and (diagnostic_score is None or diagnostic_score <= 0.0)
            ):
                continue
            value = metric(reward_data, detail_data, key)
            if value is not None:
                metrics[key] = value
        if memory_evidence_available:
            raw_worst_stage = detail_data.get("worst_memory_stage")
            if isinstance(raw_worst_stage, str) and raw_worst_stage in STAGE_LABELS:
                metrics["worst_memory_stage"] = STAGE_LABELS[raw_worst_stage]
            raw_pressure = detail_data.get("memory_pressure")
            if raw_pressure in {"comfortable", "tight", "over_budget"}:
                metrics["memory_pressure"] = raw_pressure
        if memory_bounds is not None:
            lower, upper = memory_bounds
            metrics["memory_ratio_bucket_lower"] = lower
            metrics["memory_ratio_bucket_upper"] = upper
            metrics["memory_overage_bucket_upper"] = round(
                max(0.0, upper - 1.0),
                10,
            )
        metrics["diagnostic_only"] = bool(
            not gates_passed
            and diagnostic_score is not None
            and diagnostic_score > 0.0
        )
    if deployment_feedback is not None:
        metrics["deployment_family_feedback"] = deployment_feedback
    return {
        "valid": bool(gates_passed),
        "score": score,
        "pass_rate": pass_rate,
        "summary": summary,
        "details": details,
        "metrics": metrics,
    }


def evaluate(
    verifier: Path,
    app_dir: Path,
    log_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    prepare_log_dir(log_dir)
    env = dict(os.environ)
    env["APP_DIR"] = str(app_dir)
    env["LOG_DIR"] = str(log_dir)
    completed = False
    returncode: int | None = None
    try:
        # The child verifier, candidate import, Python prints, and native fd
        # writes all terminate at DEVNULL. This trusted parent is the only
        # process connected to SForge output and emits exactly one marker pair.
        proc = subprocess.run(
            [
                sys.executable,
                str(verifier),
                "--app-dir",
                str(app_dir),
                "--log-dir",
                str(log_dir),
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        completed = True
        returncode = proc.returncode
    except (OSError, subprocess.SubprocessError):
        completed = False
    reward_data = read_json_object(log_dir / "reward.json")
    detail_data = read_json_object(log_dir / "flash_kmeans_results.json")
    return build_result(
        reward_data,
        detail_data,
        verifier_completed=completed,
        verifier_returncode=returncode,
    )


def emit(result: Mapping[str, Any]) -> None:
    print(MARKER_START)
    print(json.dumps(dict(result), sort_keys=True, ensure_ascii=False, allow_nan=False))
    print(MARKER_END)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verifier",
        type=Path,
        default=Path(__file__).resolve().with_name("verify_flash_kmeans.py"),
    )
    parser.add_argument(
        "--app-dir",
        type=Path,
        default=Path(os.environ.get("APP_DIR", "/app")),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("LOG_DIR", "/tmp/flash-kmeans-verifier-private")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("FLASH_KMEANS_VERIFY_TIMEOUT", "3300")),
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        result = evaluate(
            args.verifier.resolve(),
            args.app_dir.resolve(),
            args.log_dir.resolve(),
            args.timeout,
        )
    except Exception:
        result = build_result(
            {},
            {},
            verifier_completed=False,
            verifier_returncode=None,
        )
    emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
