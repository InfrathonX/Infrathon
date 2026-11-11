#!/usr/bin/env python3
"""Run the private verifier and emit one redacted structured result."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


MARKER_START = ">>>>> Start Structured Result"
MARKER_END = ">>>>> End Structured Result"


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def safe_formal_feedback(private_detail: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(private_detail, dict):
        return {}
    formal = private_detail.get("e2e_formal")
    if not isinstance(formal, dict):
        return {}
    feedback: dict[str, Any] = {}
    stage = formal.get("failure_stage")
    failure_class = formal.get("failure_class")
    failure_role = formal.get("failure_role")
    if stage in {
        "reference_worker",
        "candidate_worker",
        "telemetry_validation",
        "pair_comparison",
        "lifecycle",
        "controller",
    } and failure_class in {
        "timeout",
        "nonzero_exit",
        "missing_telemetry",
        "invalid_telemetry",
        "gpu_busy",
        "infrastructure",
    }:
        feedback.update(failure_stage=stage, failure_class=failure_class)
        if stage == "telemetry_validation" and failure_role in {
            "reference",
            "candidate",
        }:
            feedback["failure_role"] = failure_role
    groups = formal.get("failed_gate_groups")
    if (
        isinstance(groups, list)
        and groups
        and len(groups) <= 3
        and len(groups) == len(set(groups))
        and set(groups) <= {"loss", "memory", "correctness_update"}
    ):
        feedback["failed_gate_groups"] = groups
    return feedback


def build_result(
    reward: dict[str, Any],
    completed: bool,
    returncode: int | None,
    private_detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gates = {
        name: finite(reward.get(name)) == 1.0
        for name in (
            "source_ok",
            "correctness_ok",
            "protocol_ok",
            "e2e_schema_ok",
            "e2e_execution_ok",
            "loss_gate_ok",
            "memory_gate_ok",
            "correctness_evidence_ok",
            "formal_ready",
            "formal_score_enabled",
        )
    }
    score = finite(reward.get("reward"))
    valid = bool(completed and returncode == 0 and score is not None and score > 0 and all(gates.values()))
    accepted = float(score) if valid and score is not None else 0.0
    safe_feedback = safe_formal_feedback(private_detail)
    details = [
        {
            "name": name,
            "status": "PASSED" if passed else "FAILED",
            "score": 1.0 if passed else 0.0,
        }
        for name, passed in gates.items()
    ]
    if valid:
        summary = f"Qwen3 ORPO verification passed; score {accepted:.6f}."
    elif "failure_stage" in safe_feedback:
        summary = (
            "Formal E2E execution did not complete: "
            f"{safe_feedback['failure_stage']}/{safe_feedback['failure_class']}."
        )
    elif "failed_gate_groups" in safe_feedback:
        summary = (
            "Formal E2E execution completed; failed gate groups: "
            + ", ".join(safe_feedback["failed_gate_groups"])
            + "."
        )
    else:
        summary = "Formal evaluation did not produce an accepted benchmark score."
    metrics: dict[str, Any] = {
        "reward": accepted,
        "auto_feedback_safe": True,
        "primary_score_name": "official-TRL-normalized E2E throughput",
        "primary_score_unit": "score points (official TRL = 100)",
        "primary_score_formula": "100 * geometric_mean(five row median ratios)",
        **safe_feedback,
    }
    return {
        "valid": valid,
        "score": accepted,
        "pass_rate": 1.0 if valid else 0.0,
        "summary": summary,
        "details": details,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, default=Path("/app"))
    parser.add_argument("--log-dir", type=Path, default=Path("/logs/verifier"))
    parser.add_argument(
        "--verifier", type=Path, default=Path("/opt/orpo_judge/verifier/verify_orpo.py")
    )
    # Forty fresh two-rank cohorts must fit below the task's 12,000-second
    # evaluator limit.  Per-cohort enforcement remains inside the controller.
    parser.add_argument("--timeout", type=float, default=11_800.0)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for name in ("reward.json", "orpo_private.json"):
        path = args.log_dir / name
        if path.exists() or path.is_symlink():
            path.unlink()
    completed = False
    returncode = None
    try:
        process = subprocess.run(
            [
                sys.executable,
                str(args.verifier),
                "--app-dir",
                str(args.app_dir),
                "--log-dir",
                str(args.log_dir),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=args.timeout,
            check=False,
        )
        completed = True
        returncode = process.returncode
    except (OSError, subprocess.SubprocessError):
        pass
    result = build_result(
        read_object(args.log_dir / "reward.json"),
        completed,
        returncode,
        read_object(args.log_dir / "orpo_private.json"),
    )
    print(MARKER_START)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    print(MARKER_END)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
