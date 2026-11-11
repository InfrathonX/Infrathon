#!/usr/bin/env python3
"""Qualify every precommitted positive and negative control on full H100 rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from h100_policy_contract import (
    validate_device_envelope,
    GENERATION_COUNT,
    PROFILE,
    REQUESTED_SLOTS,
    ROW_COUNTS,
    RUNTIME_ATTESTATION_HASH_FIELDS,
    TASK_ID,
    is_sha256,
    object_at,
    sha256,
    validate_attested_runtime,
    write_object,
)


IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
GPU_UUID = re.compile(r"GPU-[0-9A-Fa-f-]+\Z")
CONTROL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "images/judge/verifier")
)
from quality_threshold import METRIC, validate_measurement


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "quality_controls.json")
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = object_at(args.manifest, "quality-control manifest")
    if (
        manifest.get("status") != "PRECOMMITTED"
        or manifest.get("task_id") != TASK_ID
        or manifest.get("profile") != PROFILE
        or not is_sha256(manifest.get("workload_policy_sha256"))
        or not is_sha256(manifest.get("hidden_cases_sha256"))
    ):
        raise SystemExit("quality-control manifest is not precommitted")
    methodology = manifest.get("methodology")
    if (
        not isinstance(methodology, dict)
        or methodology.get("metric") != METRIC
        or methodology.get("threshold_derivation")
        != "midpoint_min_positive_max_negative_v1"
        or isinstance(methodology.get("minimum_separation_ppm"), bool)
        or not isinstance(methodology.get("minimum_separation_ppm"), int)
        or methodology["minimum_separation_ppm"] <= 0
    ):
        raise SystemExit("quality-control threshold derivation is not precommitted")
    controls = {
        item["id"]: item
        for group in (manifest.get("positive_controls"), manifest.get("negative_controls"))
        if isinstance(group, list)
        for item in group
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    expected_count = sum(
        len(group)
        for group in (manifest.get("positive_controls"), manifest.get("negative_controls"))
        if isinstance(group, list)
    )
    if not controls or len(controls) != expected_count or len(args.report) != len(controls):
        raise SystemExit("one unique report is required for every precommitted control")
    evidence = []
    seen = set()
    for path in args.report:
        report = object_at(path, "quality-control report")
        control_id = report.get("control_id")
        control = controls.get(control_id)
        if (
            control is None
            or control_id in seen
            or CONTROL_ID.fullmatch(str(control_id)) is None
        ):
            raise SystemExit("quality-control report identity is unknown or duplicated")
        seen.add(control_id)
        expected = control.get("expected_classification")
        required_hash_fields = (
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
            "reference_manifest_sha256",
            *RUNTIME_ATTESTATION_HASH_FIELDS,
        )
        if (
            report.get("status") != "PASS"
            or report.get("mode") != "quality_control"
            or report.get("release_evidence") is not False
            or report.get("reference_replay") is not False
            or report.get("task_id") != TASK_ID
            or report.get("profile") != PROFILE
            or validate_device_envelope(report, str(path)) is None
            or report.get("visible_device_count") != 1
            or report.get("generation_count") != GENERATION_COUNT
            or report.get("requested_slots") != REQUESTED_SLOTS
            or report.get("row_counts") != ROW_COUNTS
            or report.get("expected_classification") != expected
            or report.get("observed_classification") != expected
            or report.get("archive_sha256") != control.get("archive_sha256")
            or report.get("hidden_cases_sha256") != manifest["hidden_cases_sha256"]
            or report.get("workload_policy_sha256")
            != manifest["workload_policy_sha256"]
            or report.get("quality_controls_sha256") != sha256(args.manifest)
            or isinstance(report.get("archive_size_bytes"), bool)
            or not isinstance(report.get("archive_size_bytes"), int)
            or report["archive_size_bytes"] <= 0
            or any(not is_sha256(report.get(field)) for field in required_hash_fields)
            or IMAGE_ID.fullmatch(str(report.get("work_image_id", ""))) is None
            or IMAGE_ID.fullmatch(str(report.get("judge_image_id", ""))) is None
            or GPU_UUID.fullmatch(str(report.get("judge_gpu_uuid", ""))) is None
        ):
            raise SystemExit(f"control report {path} lacks full identity-bound evidence")
        try:
            runtime_provenance = validate_attested_runtime(report)
        except ValueError as exc:
            raise SystemExit(
                f"control report {path} has invalid complete runtime provenance"
            ) from exc
        try:
            measurement = validate_measurement(report.get("quality_measurement"))
        except ValueError as exc:
            raise SystemExit(f"control report {path} has invalid quality measurement") from exc
        if report.get("quality_score_ppm") != measurement["score_ppm"]:
            raise SystemExit(f"control report {path} quality score is inconsistent")
        evidence.append(
            {
                "control_id": control_id,
                "expected_classification": expected,
                "observed_classification": report["observed_classification"],
                "report_sha256": sha256(path),
                **{field: report[field] for field in required_hash_fields},
                "work_image_id": report["work_image_id"],
                "judge_image_id": report["judge_image_id"],
                "judge_gpu_uuid": report["judge_gpu_uuid"],
                "archive_size_bytes": report["archive_size_bytes"],
                "quality_score_ppm": measurement["score_ppm"],
                "quality_measurement_sha256": measurement["measurement_sha256"],
                "sforge_runtime": runtime_provenance["sforge_runtime"],
                "judge_invariant_core": runtime_provenance[
                    "judge_invariant_core"
                ],
            }
        )
    common_fields = (
        "work_image_id",
        "judge_image_id",
        "judge_gpu_uuid",
        "hidden_cases_sha256",
        "dataset_manifest_sha256",
        "workload_policy_sha256",
        "memory_policy_sha256",
        "quality_controls_sha256",
        "model_validation_sha256",
        "runtime_entry_sha256",
        "source_digest",
        "validator_sha256",
        "reference_manifest_sha256",
        *RUNTIME_ATTESTATION_HASH_FIELDS,
    )
    common_identity = {field: evidence[0][field] for field in common_fields}
    if any(
        any(item[field] != common_identity[field] for field in common_fields)
        for item in evidence[1:]
    ):
        raise SystemExit("quality-control reports do not share one frozen control-build identity")
    if any(
        item["sforge_runtime"] != evidence[0]["sforge_runtime"]
        or item["judge_invariant_core"] != evidence[0]["judge_invariant_core"]
        for item in evidence[1:]
    ):
        raise SystemExit(
            "quality-control reports do not share complete SForge/Judge provenance"
        )
    common_identity["sforge_runtime"] = evidence[0]["sforge_runtime"]
    common_identity["judge_invariant_core"] = evidence[0][
        "judge_invariant_core"
    ]
    evidence.sort(key=lambda item: str(item["control_id"]))
    positive_scores = [
        int(item["quality_score_ppm"])
        for item in evidence
        if item["expected_classification"] == "ACCEPT"
    ]
    negative_scores = [
        int(item["quality_score_ppm"])
        for item in evidence
        if item["expected_classification"] == "REJECT"
    ]
    minimum_positive = min(positive_scores)
    maximum_negative = max(negative_scores)
    separation = minimum_positive - maximum_negative
    if separation < int(methodology["minimum_separation_ppm"]):
        raise SystemExit("quality controls do not provide the precommitted score separation")
    minimum_score = (minimum_positive + maximum_negative + 1) // 2
    if not maximum_negative < minimum_score <= minimum_positive:
        raise SystemExit("quality-control midpoint threshold is not separating")
    for item in evidence:
        derived = (
            "ACCEPT"
            if int(item["quality_score_ppm"]) >= minimum_score
            else "REJECT"
        )
        if derived != item["expected_classification"]:
            raise SystemExit("derived threshold does not reproduce precommitted classes")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "task_id": TASK_ID,
        "profile": PROFILE,
        "quality_controls_sha256": sha256(args.manifest),
        "workload_policy_sha256": manifest["workload_policy_sha256"],
        "hidden_cases_sha256": manifest["hidden_cases_sha256"],
        "positive_controls_attainable": all(
            item["observed_classification"] == "ACCEPT"
            for item in evidence
            if item["expected_classification"] == "ACCEPT"
        ),
        "negative_controls_rejected": all(
            item["observed_classification"] == "REJECT"
            for item in evidence
            if item["expected_classification"] == "REJECT"
        ),
        "common_control_identity": common_identity,
        "sforge_runtime": evidence[0]["sforge_runtime"],
        "judge_invariant_core": evidence[0]["judge_invariant_core"],
        "thresholds": {
            "schema_version": 1,
            "metric": METRIC,
            "derivation": "midpoint_min_positive_max_negative_v1",
            "minimum_score_ppm": minimum_score,
            "minimum_positive_score_ppm": minimum_positive,
            "maximum_negative_score_ppm": maximum_negative,
            "observed_separation_ppm": separation,
            "required_separation_ppm": methodology["minimum_separation_ppm"],
            "positive_margin_ppm": minimum_positive - minimum_score,
            "negative_margin_ppm": minimum_score - maximum_negative,
        },
        "controls": evidence,
    }
    if not payload["positive_controls_attainable"] or not payload["negative_controls_rejected"]:
        raise SystemExit("quality controls did not produce the precommitted classifications")
    print(write_object(args.output, payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
