#!/usr/bin/env python3
"""Run the v2 private verifier and emit one sanitized structured result."""

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
DETAIL_NAME = "kvzip_kv_scoring_v2_results.json"
PRIVATE_NAME = "kvzip_kv_scoring_v2_private.json.gz"
RESULT_FILES = ("reward.json", "reward.txt", DETAIL_NAME, PRIVATE_NAME)
MAX_RESULT_BYTES = 16 * 1024 * 1024
GATES = (
    ("profile_ok", "SForge evaluation-profile contract"),
    ("suite_ok", "Selected evaluation-suite contract"),
    ("source_ok", "Submission source policy"),
    ("assets_ok", "Submitted assets are not published KV-importance weights (released-weights fingerprint)"),
    ("hardware_ok", "Judge GPUs identical and within the capability envelope (compute capability >= 8.0, >= 79 GiB)"),
    ("scores_ok", "Score validity"),
    ("immutable_ok", "Candidate-input immutability"),
    ("rng_ok", "CPU/CUDA RNG immutability"),
    ("overhead_ok", "Suite and per-item overhead limits"),
    ("memory_ok", "Scorer memory limit"),
    ("metric_ok", "Finite downstream metrics"),
    ("reference_ok", "Positive full-cache task references"),
)
RATIO_KEYS = ("ratio_010", "ratio_020", "ratio_030")
GROUP_KEYS = ("group_retrieval", "group_contextual_qa", "group_redundancy")
FEEDBACK_KEYS = ("overhead_ratio_suite", "overhead_ratio_item_max", "judge_gpu_count", "question_count")
FINAL_ITEM_COUNT = 1812.0
IDENTITY_KEYS = ("final_version", "num_candidate_versions")
PROFILE_KEY = "profile_code"
SAFE_KEYS = (
    "schema_version",
    "item_count",
    "overhead_item_violations",
    "deterministic_items",
    "score_rel_auc",
    *RATIO_KEYS,
    *GROUP_KEYS,
    *FEEDBACK_KEYS,
    *(name for name, _label in GATES),
    *IDENTITY_KEYS,
    PROFILE_KEY,
)
# full-final-shared-v2: every scored round is the complete frozen final suite.
PROFILE_IDENTITIES = {
    2.0: ("final", "final-holdout-v2"),
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


def gate_detail(name: str, label: str, value: float | None) -> dict[str, Any]:
    if value is None:
        status = "ERROR"
        message = f"{label} did not produce a trusted status."
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
    score = values["score_rel_auc"]
    ratio_values = {key: values[key] for key in RATIO_KEYS}
    gate_values = {name: values[name] for name, _label in GATES}
    records_ok = bool(reward_data) and bool(detail_data)
    schema_ok = values["schema_version"] == 2.0
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
    )
    manifest_digest = detail_data.get("manifest_sha256")
    expected_item_count = FINAL_ITEM_COUNT if profile_code == 2.0 else None
    manifest_identity_ok = bool(
        values["item_count"] == expected_item_count
        and isinstance(manifest_digest, str)
        and len(manifest_digest) == 64
        and all(character in "0123456789abcdef" for character in manifest_digest)
    )
    deterministic_count = values["deterministic_items"]
    deterministic_count_ok = bool(
        deterministic_count is not None
        and deterministic_count.is_integer()
        and values["item_count"] is not None
        and 0.0 <= deterministic_count <= values["item_count"]
    )
    ratios_present = all(value is not None and value > 0.0 for value in ratio_values.values())
    aggregate_ok = bool(
        score is not None
        and score > 0.0
        and ratios_present
        and math.isclose(
            score,
            100.0 * math.fsum(float(value) for value in ratio_values.values()) / len(RATIO_KEYS),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        )
    )
    reward_ok = reward is not None and score is not None and math.isclose(
        reward, score, rel_tol=1.0e-12, abs_tol=1.0e-12
    )
    all_gates_pass = gates_well_formed and all(value == 1.0 for value in gate_values.values())
    valid = bool(
        verifier_completed
        and verifier_returncode == 0
        and records_ok
        and schema_ok
        and reward_ok
        and identities_ok
        and profile_identity_ok
        and manifest_identity_ok
        and deterministic_count_ok
        and aggregate_ok
        and all_gates_pass
    )
    trusted_gate_failure = bool(
        verifier_completed
        and verifier_returncode == 1
        and records_ok
        and schema_ok
        and identities_ok
        and profile_identity_ok
        and manifest_identity_ok
        and gates_well_formed
        and any(value == 0.0 for value in gate_values.values())
        and reward == 0.0
        and score == 0.0
        and all(value == 0.0 for value in ratio_values.values())
    )

    visible_gate_values = (
        gate_values
        if valid or trusted_gate_failure
        else {name: None for name, _label in GATES}
    )
    details = [
        gate_detail(name, label, visible_gate_values[name]) for name, label in GATES
    ]
    performance_status = "PASSED" if valid else ("FAILED" if trusted_gate_failure else "ERROR")
    accepted_score = float(reward) if valid and reward is not None else 0.0
    details.append(
        {
            "name": "performance",
            "status": performance_status,
            "score": accepted_score,
            "message": (
                "Fidelity AUC accepted after every required gate passed."
                if valid
                else "Performance was not accepted because the Judge result was invalid."
            ),
        }
    )

    if valid:
        summary = (
            f"KV scoring v2 {expected_profile[0]} verification passed; "
            f"score {accepted_score:.6f} relative-accuracy points."
        )
    elif trusted_gate_failure:
        summary = "KV scoring v2 submission failed one or more required gates."
    elif verifier_completed and verifier_returncode != 0:
        summary = "KV scoring v2 verifier exited unsuccessfully."
    else:
        summary = "KV scoring v2 verifier did not produce a trusted result."

    metrics: dict[str, Any] = {
        "source": "kvzip_kv_scoring_v2_verifier_adapter",
        "adapter_schema_version": 1,
        "candidate_output_suppressed": True,
        "result_protocol_records": 1,
        "verifier_completed": bool(verifier_completed),
        "verifier_returncode": verifier_returncode,
            "primary_score_name": "score_rel_auc",
            "primary_score_unit": "relative-accuracy points",
        "primary_score_direction": "maximize",
        "score_rel_auc": accepted_score,
    }
    if valid:
        metrics.update(
            {
                "evaluation_profile": expected_profile[0],
                "suite_id": expected_profile[1],
                PROFILE_KEY: profile_code,
            }
        )
        for key in (*RATIO_KEYS, *GROUP_KEYS, *FEEDBACK_KEYS, "item_count", *IDENTITY_KEYS, "overhead_item_violations", "deterministic_items", *(name for name, _label in GATES)):
            metrics[key] = values[key]
    elif trusted_gate_failure:
        for name, _label in GATES:
            metrics[name] = gate_values[name]
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
        default=Path("/opt/kvzip_kv_scoring_judge/verify_kv_scoring.py"),
    )
    parser.add_argument("--app-dir", type=Path, default=Path(os.environ.get("APP_DIR", "/app")))
    parser.add_argument(
        "--log-dir", type=Path, default=Path(os.environ.get("LOG_DIR", "/logs/verifier"))
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("KV_SCORING_JUDGE_TIMEOUT", "10900")),
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
