#!/usr/bin/env python3
"""Freeze reviewed provisional H100 cohort and producer deadline policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from h100_policy_contract import (
    CATEGORY_DEFINITION,
    INTERACTIVE_BONUS_CATEGORIES,
    INTERACTIVE_CATEGORY_BASE_QUOTA,
    INTERACTIVE_MINIMUM_PER_ROW,
    INTERACTIVE_SELECTION,
    INTERACTIVE_TOTAL_CASES,
    TASK_ID,
    object_at,
    sha256,
    validate_workload_policy,
    write_object,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=root / "workload_policy.json")
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--interactive-case-timeout-seconds", type=float, required=True)
    parser.add_argument("--interactive-judge-timeout-seconds", type=float, required=True)
    parser.add_argument("--cooldown-seconds", type=float, required=True)
    parser.add_argument("--final-case-timeout-seconds", type=float, required=True)
    parser.add_argument("--final-row-timeout-seconds", type=float, required=True)
    parser.add_argument("--final-inner-timeout-seconds", type=float, required=True)
    parser.add_argument("--final-outer-timeout-seconds", type=float, required=True)
    parser.add_argument("--wrapper-margin-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, default=root / "workload_policy.json")
    args = parser.parse_args()
    policy = object_at(args.policy, "bootstrap workload policy")
    validate_workload_policy(policy)
    if policy["qualification_status"] != "UNQUALIFIED":
        raise SystemExit("only an unqualified workload policy may be provisionally frozen")
    qualification = object_at(args.qualification, "workload qualification")
    if (
        qualification.get("status") != "PASS"
        or qualification.get("schema_version") != 1
        or qualification.get("task_id") != TASK_ID
        or qualification.get("profile") != policy["profile"]
        or qualification.get("workload_policy_sha256") != sha256(args.policy)
        or qualification.get("hidden_cases_sha256")
        != policy["manifest"]["hidden_cases_sha256"]
        or qualification.get("dataset_manifest_sha256")
        != policy["manifest"]["dataset_manifest_sha256"]
        or int(qualification.get("observation_count", 0)) < 3
    ):
        raise SystemExit("workload qualification is incomplete")
    limits = (
        args.interactive_case_timeout_seconds,
        args.interactive_judge_timeout_seconds,
        args.cooldown_seconds,
        args.final_case_timeout_seconds,
        args.final_row_timeout_seconds,
        args.final_inner_timeout_seconds,
        args.final_outer_timeout_seconds,
        args.wrapper_margin_seconds,
    )
    if any(value <= 0 for value in limits):
        raise SystemExit("all reviewed workload limits must be positive")
    if (
        args.interactive_case_timeout_seconds != args.final_case_timeout_seconds
        or args.interactive_judge_timeout_seconds != args.final_inner_timeout_seconds
    ):
        raise SystemExit("interactive and final complete-workload deadlines must match")
    max_elapsed = float(qualification.get("max_observed_elapsed_seconds", 0))
    max_row = float(qualification.get("max_observed_row_seconds", 0))
    max_case = float(qualification.get("max_observed_case_seconds", 0))
    if (
        args.final_case_timeout_seconds <= max_case
        or args.final_row_timeout_seconds <= max_row
        or args.final_inner_timeout_seconds
        < max_elapsed + args.wrapper_margin_seconds
        or args.final_outer_timeout_seconds - args.final_inner_timeout_seconds
        < args.wrapper_margin_seconds
    ):
        raise SystemExit(
            "reviewed final deadlines do not cover observations and wrapper margin"
        )
    policy["qualification_status"] = "QUALIFIED"
    policy["release_status"] = "PROVISIONAL"
    policy["interactive"].update(
        {
            "status": "QUALIFIED",
            "category_definition": CATEGORY_DEFINITION,
            "selection": INTERACTIVE_SELECTION,
            "total_cases": INTERACTIVE_TOTAL_CASES,
            "category_base_quota": INTERACTIVE_CATEGORY_BASE_QUOTA,
            "bonus_categories": INTERACTIVE_BONUS_CATEGORIES,
            "minimum_per_row": INTERACTIVE_MINIMUM_PER_ROW,
            "case_timeout_seconds": args.interactive_case_timeout_seconds,
            "judge_timeout_seconds": args.interactive_judge_timeout_seconds,
            "cooldown_seconds": args.cooldown_seconds,
        }
    )
    policy["final"].update(
        {
            "case_timeout_seconds": args.final_case_timeout_seconds,
            "row_timeout_seconds": args.final_row_timeout_seconds,
            "inner_timeout_seconds": args.final_inner_timeout_seconds,
            "outer_timeout_seconds": args.final_outer_timeout_seconds,
            "wrapper_margin_seconds": args.wrapper_margin_seconds,
        }
    )
    policy["qualification"].update(
        {
            "observations_sha256": qualification["observations_sha256"],
            "qualification_artifact_sha256": __import__("hashlib").sha256(
                args.qualification.read_bytes()
            ).hexdigest(),
            "approval_id": args.approval_id,
            "max_observed_case_seconds": max_case,
            "max_observed_row_seconds": max_row,
            "max_observed_elapsed_seconds": max_elapsed,
        }
    )
    validate_workload_policy(policy)
    digest = write_object(args.output, policy)
    embedded = root / "images/judge/private_data/workload_policy.json"
    if args.output.resolve() == (root / "workload_policy.json").resolve():
        embedded.write_bytes(args.output.read_bytes())
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
