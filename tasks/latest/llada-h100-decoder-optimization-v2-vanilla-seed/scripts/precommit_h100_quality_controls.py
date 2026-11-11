#!/usr/bin/env python3
"""Precommit exact positive/negative archives before control evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from h100_policy_contract import (
    PROFILE,
    TASK_ID,
    object_at,
    sha256,
    validate_workload_policy,
    write_object,
)


def parse_control(raw: str, expected: str, root: Path) -> dict:
    control_id, separator, raw_path = raw.partition("=")
    if (
        not separator
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", control_id) is None
        or not raw_path
    ):
        raise argparse.ArgumentTypeError("control must have ID=ARCHIVE form")
    path = Path(raw_path).resolve()
    if root not in path.parents or path.is_symlink() or not path.is_file():
        raise ValueError("control archive must be a safe regular file below task root")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("control archive is outside task root") from exc
    return {
        "id": control_id,
        "archive": relative,
        "archive_sha256": sha256(path),
        "expected_classification": expected,
        "acceptance_target": {
            "metric": "producer_relative_min_retention_ppm_v1",
            "expected": expected,
        },
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, default=root)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--positive", action="append", required=True)
    parser.add_argument("--negative", action="append", required=True)
    parser.add_argument("--calibration-archive", type=Path, required=True)
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.task_root.resolve()
    policy_path = args.policy.resolve() if args.policy else root / "workload_policy.json"
    output = args.output.resolve() if args.output else root / "quality_controls.json"
    policy = object_at(policy_path, "provisional workload policy")
    validate_workload_policy(policy)
    if (
        policy.get("qualification_status") != "QUALIFIED"
        or policy.get("release_status") != "PROVISIONAL"
    ):
        raise SystemExit("control precommitment requires a provisional qualified policy")
    positives = [parse_control(value, "ACCEPT", root) for value in args.positive]
    negatives = [parse_control(value, "REJECT", root) for value in args.negative]
    calibration_archive = args.calibration_archive.resolve()
    if (
        root not in calibration_archive.parents
        or calibration_archive.is_symlink()
        or not calibration_archive.is_file()
    ):
        raise SystemExit("calibration archive must be a safe regular file below task root")
    ids = [item["id"] for item in positives + negatives]
    if len(ids) != len(set(ids)):
        raise SystemExit("quality-control IDs must be unique")
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "profile": PROFILE,
        "status": "PRECOMMITTED",
        "methodology": {
            "formula": "producer_relative_min_retention_ppm_v1",
            "margin_selection": "producer_derived_after_precommitted_controls",
            "metric": "producer_relative_min_retention_ppm_v1",
            "threshold_derivation": "midpoint_min_positive_max_negative_v1",
            "minimum_separation_ppm": 10_000,
            "control_measurement_floor_ppm": 1,
            "interactive_policy": "same_as_final",
            "final_policy": "hard_gate_after_control_qualification",
        },
        "workload_policy_sha256": sha256(policy_path),
        "hidden_cases_sha256": policy["manifest"]["hidden_cases_sha256"],
        "producer_calibration": {
            "archive": calibration_archive.relative_to(root).as_posix(),
            "archive_sha256": sha256(calibration_archive),
            "expected_mode": "calibration",
        },
        "positive_controls": positives,
        "negative_controls": negatives,
        "approval_id": args.approval_id,
        "historical_a6000_controls_applicability": "NONE",
    }
    digest = write_object(output, payload)
    if output == root / "quality_controls.json" and (
        root / "images/judge/private_data"
    ).is_dir():
        private_copy = root / "images/judge/private_data/quality_controls.json"
        private_copy.write_bytes(output.read_bytes())
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
