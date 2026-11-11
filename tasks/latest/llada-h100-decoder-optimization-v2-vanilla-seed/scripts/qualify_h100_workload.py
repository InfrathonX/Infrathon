#!/usr/bin/env python3
"""Validate at least three non-release stratified-520 H100 timing observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from h100_policy_contract import (
    validate_device_envelope,
    GENERATION_COUNT,
    PROFILE,
    REQUESTED_SLOTS,
    RUNTIME_ATTESTATION_HASH_FIELDS,
    ROW_COUNTS,
    object_at,
    sha256,
    validate_attested_runtime,
    validate_fullscale_attestation,
    write_object,
)


def positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} is not positive and finite")
    return number


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.report) < 3:
        raise SystemExit(
            "at least three complete non-release stratified-520 reports are required"
        )
    observations = []
    run_ids = set()
    common_identity: dict[str, object] | None = None
    workload_policy_path = root / "workload_policy.json"
    memory_policy_path = root / "memory_policy.json"
    workload_policy = object_at(workload_policy_path, "workload policy")
    workload_sha256 = sha256(workload_policy_path)
    memory_sha256 = sha256(memory_policy_path)
    quality_controls_sha256 = sha256(root / "quality_controls.json")
    for path in args.report:
        report = object_at(path, "workload timing report")
        try:
            validate_fullscale_attestation(
                report,
                mode="qualification",
                expected_workload_policy_sha256=workload_sha256,
                expected_memory_policy_sha256=memory_sha256,
                expected_quality_controls_sha256=quality_controls_sha256,
                expected_hidden_sha256=workload_policy["manifest"][
                    "hidden_cases_sha256"
                ],
                expected_dataset_manifest_sha256=workload_policy["manifest"][
                    "dataset_manifest_sha256"
                ],
            )
        except ValueError as exc:
            raise SystemExit(f"{path} is not bound qualification evidence: {exc}") from exc
        try:
            validate_attested_runtime(report)
        except ValueError as exc:
            raise SystemExit(
                f"{path} is not bound qualification evidence: {exc}"
            ) from exc
        run_id = report.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in run_ids:
            raise SystemExit("workload timing report run IDs must be unique")
        run_ids.add(run_id)
        identity_fields = (
            "archive_sha256",
            "archive_size_bytes",
            "work_image_id",
            "judge_image_id",
            "model_validation_sha256",
            "runtime_entry_sha256",
            "source_digest",
            "validator_sha256",
            "judge_gpu_uuid",
            "workload_policy_sha256",
            "memory_policy_sha256",
            "quality_controls_sha256",
            "hidden_cases_sha256",
            "dataset_manifest_sha256",
            *RUNTIME_ATTESTATION_HASH_FIELDS,
            "sforge_runtime",
            "judge_invariant_core",
        )
        current_identity = {field: report[field] for field in identity_fields}
        if common_identity is None:
            common_identity = current_identity
        elif current_identity != common_identity:
            raise SystemExit(
                "workload timing reports do not share one corrected archive/build/GPU cycle"
            )
        if (
            report.get("status") != "PASS"
            or report.get("mode") != "qualification"
            or report.get("release_evidence") is not False
            or report.get("reference_replay") is not False
            or validate_device_envelope(report, str(path)) is None
            or report.get("visible_device_count") != 1
            or report.get("profile") != PROFILE
            or report.get("generation_count") != GENERATION_COUNT
            or report.get("requested_slots") != REQUESTED_SLOTS
            or report.get("row_counts") != ROW_COUNTS
        ):
            raise SystemExit(
                f"{path} is not a complete non-release stratified-520 run"
            )
        rows = report.get("row_seconds")
        if not isinstance(rows, dict) or set(rows) != set(ROW_COUNTS):
            raise SystemExit(f"{path} has incomplete row timing")
        observations.append(
            {
                "run_id": run_id,
                "report_sha256": sha256(path),
                "archive_sha256": report["archive_sha256"],
                "work_image_id": report["work_image_id"],
                "judge_image_id": report["judge_image_id"],
                "judge_gpu_uuid": report["judge_gpu_uuid"],
                "elapsed_seconds": positive(report.get("elapsed_seconds"), "elapsed_seconds"),
                "max_case_seconds": positive(
                    report.get("max_case_seconds"), "max_case_seconds"
                ),
                "row_seconds": {
                    row_id: positive(rows[row_id], f"{row_id} seconds")
                    for row_id in ROW_COUNTS
                },
            }
        )
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "task_id": "llada_h100_decoder_optimization_v2",
        "profile": PROFILE,
        "workload_policy_sha256": workload_sha256,
        "memory_policy_sha256": memory_sha256,
        "quality_controls_sha256": quality_controls_sha256,
        "hidden_cases_sha256": workload_policy["manifest"]["hidden_cases_sha256"],
        "dataset_manifest_sha256": workload_policy["manifest"][
            "dataset_manifest_sha256"
        ],
        "required_observations": 3,
        "observation_count": len(observations),
        "max_observed_elapsed_seconds": max(
            observation["elapsed_seconds"] for observation in observations
        ),
        "max_observed_case_seconds": max(
            observation["max_case_seconds"] for observation in observations
        ),
        "max_observed_row_seconds": max(
            seconds
            for observation in observations
            for seconds in observation["row_seconds"].values()
        ),
        "observations": observations,
        "common_producer_identity": common_identity,
        "observations_sha256": hashlib.sha256(
            json.dumps(observations, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    print(write_object(args.output, payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
