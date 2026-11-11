#!/usr/bin/env python3
"""Freeze release limits after producer and independent control qualification."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

from h100_policy_contract import (
    validate_device_envelope,
    GENERATION_COUNT,
    PROFILE,
    object_at,
    is_sha256,
    ROW_COUNTS,
    REQUESTED_SLOTS,
    RUNTIME_ATTESTATION_HASH_FIELDS,
    sha256,
    validate_attested_runtime,
    validate_workload_policy,
    write_object,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=root / "workload_policy.json")
    parser.add_argument("--producer-attestation", type=Path, required=True)
    parser.add_argument("--quality-control-attestation", type=Path, required=True)
    parser.add_argument(
        "--quality-controls", type=Path, default=root / "quality_controls.json"
    )
    parser.add_argument(
        "--reference-quality",
        type=Path,
        default=root / "images/judge/private_data/reference_quality.json",
    )
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--output", type=Path, default=root / "workload_policy.json")
    args = parser.parse_args()
    policy = object_at(args.policy, "provisional workload policy")
    validate_workload_policy(policy)
    if policy.get("release_status") != "PROVISIONAL":
        raise SystemExit("release finalization requires a provisional qualified policy")
    producer = object_at(args.producer_attestation, "H100 producer attestation")
    controls = object_at(args.quality_control_attestation, "quality control attestation")
    control_manifest = object_at(args.quality_controls, "quality-control manifest")
    reference = object_at(args.reference_quality, "calibrated quality reference")
    policy_sha256 = sha256(args.policy)
    manifest = policy["manifest"]
    required_hashes = (
        "archive_sha256",
        "hidden_cases_sha256",
        "dataset_manifest_sha256",
        "workload_policy_sha256",
        "memory_policy_sha256",
        "quality_controls_sha256",
        "model_validation_sha256",
        "runtime_entry_sha256",
        "source_digest",
        "validator_sha256",
        *RUNTIME_ATTESTATION_HASH_FIELDS,
    )
    if (
        producer.get("status") != "PASS"
        or producer.get("mode") != "calibration"
        or producer.get("release_evidence") is not False
        or producer.get("reference_replay") is not False
        or producer.get("task_id") != policy["task_id"]
        or validate_device_envelope(producer, "producer attestation") is None
        or producer.get("visible_device_count") != 1
        or producer.get("profile") != PROFILE
        or producer.get("generation_count") != GENERATION_COUNT
        or producer.get("requested_slots") != REQUESTED_SLOTS
        or producer.get("row_counts") != ROW_COUNTS
        or producer.get("hidden_cases_sha256") != manifest["hidden_cases_sha256"]
        or producer.get("dataset_manifest_sha256")
        != manifest["dataset_manifest_sha256"]
        or producer.get("quality_controls_sha256") != sha256(args.quality_controls)
        or producer.get("workload_policy_sha256") != policy_sha256
        or any(not is_sha256(producer.get(field)) for field in required_hashes)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(producer.get("work_image_id", "")))
        is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(producer.get("judge_image_id", "")))
        is None
        or re.fullmatch(r"GPU-[0-9A-Fa-f-]+", str(producer.get("judge_gpu_uuid", "")))
        is None
    ):
        raise SystemExit("producer is not accepted stratified-520 evidence")
    try:
        producer_runtime = validate_attested_runtime(producer)
    except ValueError as exc:
        raise SystemExit("producer complete SForge/Judge provenance is invalid") from exc
    producer_rows = producer.get("row_seconds")
    qualification = policy.get("qualification")
    final = policy["final"]
    if (
        not isinstance(producer_rows, dict)
        or set(producer_rows) != set(ROW_COUNTS)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in producer_rows.values()
        )
        or any(
            isinstance(producer.get(field), bool)
            or not isinstance(producer.get(field), (int, float))
            or not math.isfinite(float(producer[field]))
            or float(producer[field]) <= 0
            for field in ("max_case_seconds", "elapsed_seconds")
        )
        or not isinstance(qualification, dict)
    ):
        raise SystemExit("producer timing evidence is incomplete")
    producer_max_row = max(float(value) for value in producer_rows.values())
    qualification_max_case = float(qualification["max_observed_case_seconds"])
    qualification_max_row = float(qualification["max_observed_row_seconds"])
    qualification_max_elapsed = float(
        qualification["max_observed_elapsed_seconds"]
    )
    wrapper_margin = float(final["wrapper_margin_seconds"])
    if (
        float(final["case_timeout_seconds"])
        <= max(float(producer["max_case_seconds"]), qualification_max_case)
        or float(final["row_timeout_seconds"])
        <= max(producer_max_row, qualification_max_row)
        or float(final["inner_timeout_seconds"])
        < max(float(producer["elapsed_seconds"]), qualification_max_elapsed)
        + wrapper_margin
        or float(final["outer_timeout_seconds"])
        - float(final["inner_timeout_seconds"])
        < wrapper_margin
    ):
        raise SystemExit("release deadlines do not cover producer/qualification observations")
    expected_controls = {
        item["id"]: item
        for group in (
            control_manifest.get("positive_controls"),
            control_manifest.get("negative_controls"),
        )
        if isinstance(group, list)
        for item in group
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    evidence = controls.get("controls")
    if (
        controls.get("status") != "PASS"
        or controls.get("task_id") != policy["task_id"]
        or controls.get("profile") != PROFILE
        or control_manifest.get("status") != "PRECOMMITTED"
        or control_manifest.get("task_id") != policy["task_id"]
        or control_manifest.get("profile") != PROFILE
        or control_manifest.get("workload_policy_sha256") != policy_sha256
        or control_manifest.get("hidden_cases_sha256")
        != manifest["hidden_cases_sha256"]
        or not isinstance(control_manifest.get("producer_calibration"), dict)
        or control_manifest["producer_calibration"].get("archive_sha256")
        != producer.get("archive_sha256")
        or controls.get("quality_controls_sha256") != sha256(args.quality_controls)
        or controls.get("workload_policy_sha256") != policy_sha256
        or controls.get("hidden_cases_sha256") != manifest["hidden_cases_sha256"]
        or controls.get("positive_controls_attainable") is not True
        or controls.get("negative_controls_rejected") is not True
        or not isinstance(evidence, list)
        or len(evidence) != len(expected_controls)
    ):
        raise SystemExit("independent quality controls are not qualified")
    thresholds = controls.get("thresholds")
    common_control_identity = controls.get("common_control_identity")
    reference_provenance = reference.get("calibration_provenance")
    try:
        control_runtime = validate_attested_runtime(
            {
                **(common_control_identity or {}),
                "sforge_runtime": controls.get("sforge_runtime"),
                "judge_invariant_core": controls.get("judge_invariant_core"),
            }
        )
    except ValueError as exc:
        raise SystemExit("control complete SForge/Judge provenance is invalid") from exc
    invariant_transition_fields = (
        "judge_gpu_uuid",
        "model_validation_sha256",
        "runtime_entry_sha256",
        "validator_sha256",
        "hidden_cases_sha256",
        "dataset_manifest_sha256",
        "workload_policy_sha256",
        "memory_policy_sha256",
        "quality_controls_sha256",
        *RUNTIME_ATTESTATION_HASH_FIELDS,
    )
    if (
        not isinstance(thresholds, dict)
        or thresholds.get("metric")
        != "producer_relative_min_retention_ppm_v1"
        or thresholds.get("derivation")
        != "midpoint_min_positive_max_negative_v1"
        or not isinstance(common_control_identity, dict)
        or common_control_identity.get("reference_manifest_sha256")
        != sha256(args.reference_quality)
        or common_control_identity.get("quality_controls_sha256")
        != sha256(args.quality_controls)
        or common_control_identity.get("workload_policy_sha256") != policy_sha256
        or common_control_identity.get("source_digest") == producer.get("source_digest")
        or common_control_identity.get("sforge_runtime")
        != controls.get("sforge_runtime")
        or common_control_identity.get("judge_invariant_core")
        != controls.get("judge_invariant_core")
        or any(
            common_control_identity.get(field) != producer.get(field)
            for field in invariant_transition_fields
        )
        or control_runtime != producer_runtime
        or not isinstance(reference_provenance, dict)
        or reference_provenance.get("evaluation_attestation_sha256")
        != sha256(args.producer_attestation)
        or reference_provenance.get("quality_controls_sha256")
        != sha256(args.quality_controls)
        or reference_provenance.get("workload_policy_sha256") != policy_sha256
        or any(
            reference_provenance.get(field) != producer.get(field)
            for field in RUNTIME_ATTESTATION_HASH_FIELDS
        )
        or reference_provenance.get("sforge_runtime") != producer_runtime["sforge_runtime"]
        or reference_provenance.get("judge_invariant_core")
        != producer_runtime["judge_invariant_core"]
    ):
        raise SystemExit("producer-to-control image/reference transition is not proven")
    seen = set()
    for item in evidence:
        control_id = item.get("control_id") if isinstance(item, dict) else None
        expected = expected_controls.get(control_id)
        if (
            expected is None
            or control_id in seen
            or item.get("expected_classification")
            != expected.get("expected_classification")
            or item.get("observed_classification")
            != expected.get("expected_classification")
            or item.get("archive_sha256") != expected.get("archive_sha256")
            or item.get("hidden_cases_sha256") != manifest["hidden_cases_sha256"]
            or item.get("workload_policy_sha256") != policy_sha256
            or any(not is_sha256(item.get(field)) for field in required_hashes)
            or item.get("sforge_runtime") != control_runtime["sforge_runtime"]
            or item.get("judge_invariant_core")
            != control_runtime["judge_invariant_core"]
        ):
            raise SystemExit("quality-control evidence does not match its precommitment")
        seen.add(control_id)
    policy["release_status"] = "FROZEN"
    policy["provisional_parent_sha256"] = policy_sha256
    policy["quality_policy"] = {
        "status": "QUALIFIED",
        "interactive": "same_as_final",
        "final": "paired_non_inferiority_hard_gate",
        "provisional_parent_sha256": policy_sha256,
        "producer_attestation_sha256": sha256(args.producer_attestation),
        "control_attestation_sha256": sha256(args.quality_control_attestation),
        "quality_controls_sha256": sha256(args.quality_controls),
        "producer_archive_sha256": producer["archive_sha256"],
        "producer_image_ids": {
            "work": producer["work_image_id"],
            "judge": producer["judge_image_id"],
        },
        "producer_judge_gpu_uuid": producer["judge_gpu_uuid"],
        "control_build_identity": common_control_identity,
        "thresholds": thresholds,
        "approval_id": args.approval_id,
    }
    policy["timeout_release"] = {
        "status": "QUALIFIED",
        "producer_attestation_sha256": sha256(args.producer_attestation),
        "producer_max_case_seconds": float(producer["max_case_seconds"]),
        "producer_max_row_seconds": producer_max_row,
        "producer_elapsed_seconds": float(producer["elapsed_seconds"]),
        "qualification_max_case_seconds": qualification_max_case,
        "qualification_max_row_seconds": qualification_max_row,
        "qualification_max_elapsed_seconds": qualification_max_elapsed,
        "approved_wrapper_margin_seconds": wrapper_margin,
    }
    validate_workload_policy(policy, require_frozen=True)
    digest = write_object(args.output, policy)
    if args.output.resolve() == (root / "workload_policy.json").resolve():
        (root / "images/judge/private_data/workload_policy.json").write_bytes(
            args.output.read_bytes()
        )
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
