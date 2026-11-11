#!/usr/bin/env python3
"""Run the v4 private verifier and emit one sanitized structured result."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


MARKER_START = ">>>>> Start Structured Result"
MARKER_END = ">>>>> End Structured Result"
DETAIL_NAME = "standalone_grouped_fp8_v3_results.json"
RESULT_FILES = ("reward.json", "reward.txt", DETAIL_NAME)
MAX_RESULT_BYTES = 16 * 1024 * 1024
GATES = (
    ("profile_ok", "SForge evaluation-profile contract"),
    ("suite_ok", "Scored-case and edge-case suite contract"),
    ("source_ok", "Submission source policy"),
    ("hardware_ok", "Single-GPU capability envelope (compute capability >= 8.9, >= 56 GiB)"),
    ("edge_correctness_ok", "Edge-case correctness over the declared support domain"),
    ("correctness_ok", "Scored-case correctness"),
    ("memory_safety_ok", "Memory safety (guard bands, unchanged inputs, compute-sanitizer memcheck)"),
    ("harness_integrity_ok", "Harness integrity (unpatched timing and checking, plausible throughput)"),
)
FAMILY_KEYS = ("prefill_tflops", "decode_tflops", "latency_tflops", "sequence_tflops")
IDENTITY_KEYS = ("final_version", "num_candidate_versions")
PROFILE_KEY = "profile_code"
EDGE_KEYS = ("edge_cases_total", "edge_cases_failed")
FAILED_GATE_KEY = "failed_gate_code"
# 1.0 when the compute-sanitizer pass ran and passed, 0.0 when the deployment
# skips it (a platform where it cannot attach); the verifier, not the
# candidate, sets it from the deployment's sanitizer mode.
SANITIZER_KEY = "sanitizer_ran"
MEMORY_GATE_WITHOUT_SANITIZER = (
    "Memory safety (guard bands, unchanged inputs; the compute-sanitizer pass is not run in this deployment)"
)
EVALUATION_POLICY = "full-final-shared-v4"
SELECTION_PROTOCOL_ID = "dynamic-grouped-fp8-shared-final-v4"
SCHEMA_VERSION = 4.0
SCORED_CASE_COUNT = 1000.0
EDGE_CASE_COUNT = 188.0
SAFE_KEYS = (
    "schema_version",
    "case_count",
    "score_tflops",
    *FAMILY_KEYS,
    *(name for name, _label in GATES),
    *IDENTITY_KEYS,
    *EDGE_KEYS,
    FAILED_GATE_KEY,
    PROFILE_KEY,
    SANITIZER_KEY,
)
PROFILE_IDENTITIES = {
    2.0: ("final", "final-holdout-v4"),
}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(float(value))
        except (OverflowError, ValueError):
            return False
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    return False


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > MAX_RESULT_BYTES
        ):
            return {}
        parsed = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict) or not _all_numbers_finite(parsed):
        return {}
    return parsed


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


def _signal_session(session_id: int, requested_signal: signal.Signals) -> None:
    process_groups: set[int] = set()
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        entries = ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getsid(pid) == session_id:
                process_groups.add(os.getpgid(pid))
        except (OSError, ProcessLookupError):
            continue
    for process_group in sorted(process_groups, reverse=True):
        try:
            os.killpg(process_group, requested_signal)
        except (OSError, ProcessLookupError):
            pass


def _terminate_session(process: subprocess.Popen[bytes]) -> None:
    _signal_session(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    _signal_session(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _matching_metric(
    reward_data: Mapping[str, Any],
    detail_data: Mapping[str, Any],
    key: str,
) -> float | None:
    reward_value = finite_number(reward_data.get(key))
    detail_value = finite_number(detail_data.get(key))
    if reward_value is None or detail_value is None:
        return None
    if not math.isclose(reward_value, detail_value, rel_tol=1.0e-12, abs_tol=1.0e-12):
        return None
    return reward_value


def gate_detail(
    name: str, label: str, value: float | None, *, skipped: bool = False
) -> dict[str, Any]:
    if value is None:
        status = "ERROR"
        message = f"{label} did not produce a trusted status."
    elif skipped and value < 1.0:
        status = "SKIPPED"
        message = f"{label} was not evaluated because an earlier gate failed."
    elif value >= 1.0:
        status = "PASSED"
        message = f"{label} passed."
    else:
        status = "FAILED"
        message = f"{label} failed."
    return {
        "name": name,
        "status": status,
        "score": 1.0 if status == "PASSED" else 0.0,
        "message": message,
    }


def build_structured_result(
    reward_data: Mapping[str, Any],
    detail_data: Mapping[str, Any],
    *,
    verifier_completed: bool,
    verifier_returncode: int | None,
) -> dict[str, Any]:
    reward = _matching_metric(reward_data, detail_data, "reward")
    values = {key: _matching_metric(reward_data, detail_data, key) for key in SAFE_KEYS}
    score = values["score_tflops"]
    families = [values[key] for key in FAMILY_KEYS]
    gate_values = {name: values[name] for name, _label in GATES}
    records_ok = bool(reward_data) and bool(detail_data)
    schema_ok = values["schema_version"] == SCHEMA_VERSION
    gates_well_formed = all(value in (0.0, 1.0) for value in gate_values.values())
    identities_ok = all(
        values[key] is not None and values[key].is_integer() and values[key] >= 1.0
        for key in IDENTITY_KEYS
    )
    profile_code = values[PROFILE_KEY]
    expected_profile = PROFILE_IDENTITIES.get(profile_code) if profile_code is not None else None
    profile_identity_ok = bool(
        expected_profile is not None
        and detail_data.get("evaluation_profile") == expected_profile[0]
        and detail_data.get("suite_id") == expected_profile[1]
        and detail_data.get("evaluation_policy") == EVALUATION_POLICY
        and detail_data.get("selection_protocol_id") == SELECTION_PROTOCOL_ID
    )
    digests = (detail_data.get("manifest_sha256"), detail_data.get("edge_manifest_sha256"))
    manifest_identity_ok = bool(
        values["case_count"] == SCORED_CASE_COUNT
        and values["edge_cases_total"] == EDGE_CASE_COUNT
        and values["edge_cases_failed"] == 0.0
        and all(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in digests
        )
    )
    aggregate_ok = bool(
        score is not None
        and score > 0.0
        and all(value is not None and value > 0.0 for value in families)
        and math.isclose(
            score,
            math.exp(math.fsum(math.log(value) for value in families) / len(families)),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        )
    )
    reward_ok = reward is not None and score is not None and math.isclose(
        reward, score, rel_tol=1.0e-12, abs_tol=1.0e-12
    )
    all_gates_pass = gates_well_formed and all(value == 1.0 for value in gate_values.values())
    sanitizer_ran = values[SANITIZER_KEY]
    sanitizer_record_ok = sanitizer_ran in (0.0, 1.0)
    valid = bool(
        verifier_completed
        and verifier_returncode == 0
        and records_ok
        and schema_ok
        and reward_ok
        and identities_ok
        and profile_identity_ok
        and manifest_identity_ok
        and aggregate_ok
        and all_gates_pass
        and sanitizer_record_ok
    )
    trusted_gate_failure = bool(
        verifier_completed
        and verifier_returncode == 1
        and records_ok
        and schema_ok
        and gates_well_formed
        and any(value == 0.0 for value in gate_values.values())
        and reward == 0.0
        and score == 0.0
        and all(value == 0.0 for value in families)
    )

    visible_gate_values = (
        gate_values
        if valid or trusted_gate_failure
        else {name: None for name, _label in GATES}
    )
    # Exactly one gate is reported FAILED when the verifier names the first
    # failing gate; the gates it never reached are SKIPPED, not FAILED.
    failed_code = values[FAILED_GATE_KEY]
    failed_gate = (
        GATES[int(failed_code) - 1][0]
        if failed_code is not None and failed_code.is_integer() and 1 <= failed_code <= len(GATES)
        else None
    )
    without_sanitizer = valid and sanitizer_ran == 0.0
    details = [
        gate_detail(
            name,
            MEMORY_GATE_WITHOUT_SANITIZER if without_sanitizer and name == "memory_safety_ok" else label,
            visible_gate_values[name],
            skipped=trusted_gate_failure and failed_gate is not None and name != failed_gate,
        )
        for name, label in GATES
    ]
    performance_status = "PASSED" if valid else ("FAILED" if trusted_gate_failure else "ERROR")
    accepted_score = float(reward) if valid and reward is not None else 0.0
    details.append(
        {
            "name": "performance",
            "status": performance_status,
            "score": accepted_score,
            "message": (
                "Effective TFLOPS accepted after every required gate passed."
                if valid
                else "Performance was not accepted because the Judge result was invalid."
            ),
        }
    )

    if valid:
        summary = (
            f"Standalone grouped FP8 v4 {expected_profile[0]} verification passed; "
            f"score {accepted_score:.6f} TFLOPS"
            + (" (compute-sanitizer pass not run in this deployment)." if without_sanitizer else ".")
        )
    elif trusted_gate_failure:
        summary = "Standalone grouped FP8 v4 submission failed a required gate"
        if failed_gate is not None:
            summary += f" (first failing gate: {failed_gate}; later gates were not evaluated)"
        edge_total, edge_failed = values["edge_cases_total"], values["edge_cases_failed"]
        if edge_total and edge_failed:
            summary += f"; {int(edge_failed)} of {int(edge_total)} private edge cases failed"
        summary += "."
    elif verifier_completed and verifier_returncode != 0:
        summary = "Standalone grouped FP8 v4 verifier exited unsuccessfully."
    else:
        summary = "Standalone grouped FP8 v4 verifier did not produce a trusted result."

    metrics: dict[str, Any] = {
        "source": "standalone_grouped_fp8_v4_verifier_adapter",
        "adapter_schema_version": 4,
        "candidate_output_suppressed": True,
        "result_protocol_records": 1,
        "verifier_completed": bool(verifier_completed),
        "verifier_returncode": verifier_returncode,
        "primary_score_name": "score_tflops",
        "primary_score_unit": "TFLOPS",
        "primary_score_direction": "maximize",
        "score_tflops": accepted_score,
    }
    if valid:
        metrics.update(
            {
                "evaluation_profile": expected_profile[0],
                "suite_id": expected_profile[1],
                "evaluation_policy": EVALUATION_POLICY,
                "selection_protocol_id": SELECTION_PROTOCOL_ID,
                **{key: values[key] for key in FAMILY_KEYS},
                PROFILE_KEY: profile_code,
                SANITIZER_KEY: sanitizer_ran,
            }
        )
        for key in (*IDENTITY_KEYS, *(name for name, _label in GATES)):
            metrics[key] = values[key]
    elif trusted_gate_failure:
        for name, _label in GATES:
            metrics[name] = gate_values[name]
        if failed_gate is not None:
            metrics["failed_gate"] = failed_gate
        for key in EDGE_KEYS:
            value = values[key]
            if value is not None and value.is_integer() and value >= 0.0:
                metrics[key] = value
        for key in IDENTITY_KEYS:
            value = values[key]
            if value is not None and value.is_integer() and value >= 1.0:
                metrics[key] = value

    return {
        "valid": valid,
        "score": accepted_score,
        "pass_rate": 1.0 if valid else 0.0,
        "summary": summary,
        "details": details,
        "metrics": metrics,
    }


def evaluate(
    verifier: Path,
    app_dir: Path,
    log_dir: Path,
    timeout: float,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    prepare_log_dir(log_dir)
    env = dict(os.environ if environ is None else environ)
    env["APP_DIR"] = str(app_dir)
    env["LOG_DIR"] = str(log_dir)
    completed = False
    returncode: int | None = None
    try:
        process = subprocess.Popen(
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
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
            completed = True
            _signal_session(process.pid, signal.SIGKILL)
        except subprocess.TimeoutExpired:
            _terminate_session(process)
    except (OSError, subprocess.SubprocessError):
        if "process" in locals():
            _terminate_session(process)

    reward_data = read_json_object(log_dir / "reward.json")
    detail_data = read_json_object(log_dir / DETAIL_NAME)
    return build_structured_result(
        reward_data,
        detail_data,
        verifier_completed=completed,
        verifier_returncode=returncode,
    )


def emit_structured(result: Mapping[str, Any]) -> None:
    print(MARKER_START)
    print(json.dumps(dict(result), ensure_ascii=False, sort_keys=True, allow_nan=False))
    print(MARKER_END)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verifier",
        type=Path,
        default=Path("/opt/standalone_grouped_fp8_judge/verify_grouped.py"),
    )
    parser.add_argument("--app-dir", type=Path, default=Path(os.environ.get("APP_DIR", "/app")))
    parser.add_argument(
        "--log-dir", type=Path, default=Path(os.environ.get("LOG_DIR", "/logs/verifier"))
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("STANDALONE_GROUPED_FP8_JUDGE_TIMEOUT", "6400")),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate(
        args.verifier.resolve(),
        args.app_dir.resolve(),
        args.log_dir.resolve(),
        args.timeout,
    )
    emit_structured(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
