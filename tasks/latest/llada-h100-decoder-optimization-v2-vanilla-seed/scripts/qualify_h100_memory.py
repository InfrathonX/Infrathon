#!/usr/bin/env python3
"""Validate non-release H100 allocated/reserved/NVML memory evidence."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

from h100_policy_contract import (
    validate_device_envelope,
    PROFILE,
    ROW_SENTINEL_REPEAT_GUARD_BYTES,
    RUNTIME_ATTESTATION_HASH_FIELDS,
    TASK_ID,
    object_at,
    sha256,
    validate_attested_runtime,
    validate_fullscale_attestation,
    write_object,
)


METRICS = (
    "torch_peak_allocated_bytes",
    "torch_peak_reserved_bytes",
    "nvml_peak_device_used_bytes",
)


def metric(report: dict, name: str) -> int:
    value = report.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"memory report has invalid {name}")
    return value


def validate_report(
    path: Path,
    expected_kind: str,
    *,
    workload_sha256: str,
    memory_sha256: str,
    quality_controls_sha256: str,
    hidden_sha256: str,
    dataset_sha256: str,
) -> dict:
    report = object_at(path, f"{expected_kind} memory report")
    validate_fullscale_attestation(
        report,
        mode="qualification",
        expected_workload_policy_sha256=workload_sha256,
        expected_memory_policy_sha256=memory_sha256,
        expected_quality_controls_sha256=quality_controls_sha256,
        expected_hidden_sha256=hidden_sha256,
        expected_dataset_manifest_sha256=dataset_sha256,
    )
    validate_attested_runtime(report)
    if (
        report.get("status") != "PASS"
        or report.get("mode") != "qualification"
        or report.get("release_evidence") is not False
        or report.get("reference_replay") is not False
        or validate_device_envelope(report, str(path)) is None
        or report.get("visible_device_count") != 1
        or report.get("fixture_kind") != expected_kind
    ):
        raise ValueError(f"{path} is not valid non-release {expected_kind} evidence")
    values = {name: metric(report, name) for name in METRICS}
    for name in (
        "nvml_total_bytes",
        "nvml_clean_idle_free_bytes",
        "torch_device_total_bytes",
    ):
        values[name] = metric(report, name)
    observed_growth = report.get("row_sentinel_observed_max_growth_bytes")
    if (
        isinstance(observed_growth, bool)
        or not isinstance(observed_growth, int)
        or observed_growth < 0
        or report.get("row_sentinel_all_stable") is not True
    ):
        raise ValueError("memory report lacks stable long-run row-sentinel evidence")
    values["row_sentinel_observed_max_growth_bytes"] = observed_growth
    observed_repeat_growth = report.get(
        "row_sentinel_observed_max_repeat_growth_bytes"
    )
    if (
        isinstance(observed_repeat_growth, bool)
        or not isinstance(observed_repeat_growth, int)
        or observed_repeat_growth < 0
        or observed_repeat_growth > ROW_SENTINEL_REPEAT_GUARD_BYTES
        or report.get("row_sentinel_repeat_all_stable") is not True
    ):
        raise ValueError(
            "memory report lacks stable post-row repeat-sentinel evidence"
        )
    values["row_sentinel_observed_max_repeat_growth_bytes"] = (
        observed_repeat_growth
    )
    if not (
        values["torch_peak_allocated_bytes"]
        <= values["torch_peak_reserved_bytes"]
        <= values["nvml_peak_device_used_bytes"]
        < values["nvml_total_bytes"]
    ):
        raise ValueError("memory measurements do not fit the NVML device envelope")
    if not (
        values["torch_peak_allocated_bytes"] < values["torch_device_total_bytes"]
        <= values["nvml_total_bytes"]
        and values["nvml_clean_idle_free_bytes"] <= values["nvml_total_bytes"]
    ):
        raise ValueError("reported torch/NVML capacity is inconsistent")
    return values


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--stress-report", type=Path, required=True)
    parser.add_argument(
        "--fixture-manifest",
        type=Path,
        default=root / "memory_qualification_fixtures.json",
    )
    parser.add_argument("--proposed-max-peak-gib", type=float, required=True)
    parser.add_argument("--proposed-headroom-mib", type=int, required=True)
    parser.add_argument("--safety-margin-mib", type=int, required=True)
    parser.add_argument(
        "--proposed-row-sentinel-max-growth-mib", type=int, required=True
    )
    parser.add_argument("--row-sentinel-safety-margin-mib", type=int, required=True)
    parser.add_argument("--minimum-stress-cap-utilization", type=float, required=True)
    parser.add_argument("--max-stress-gap-mib", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workload_policy_path = root / "workload_policy.json"
    memory_policy_path = root / "memory_policy.json"
    workload_policy = object_at(workload_policy_path, "workload policy")
    workload_sha256 = sha256(workload_policy_path)
    memory_sha256 = sha256(memory_policy_path)
    fixture_manifest = object_at(
        args.fixture_manifest, "memory qualification fixture manifest"
    )
    fixture_manifest_sha256 = sha256(args.fixture_manifest)
    fixtures = fixture_manifest.get("fixtures")
    if (
        fixture_manifest.get("schema_version") != 1
        or fixture_manifest.get("status") != "PRECOMMITTED"
        or fixture_manifest.get("task_id") != TASK_ID
        or fixture_manifest.get("profile") != PROFILE
        or fixture_manifest.get("workload_policy_sha256") != workload_sha256
        or fixture_manifest.get("memory_policy_sha256") != memory_sha256
        or not isinstance(fixtures, dict)
        or set(fixtures) != {"corrected_reference", "near_cap_stress"}
        or not isinstance(fixture_manifest.get("approval_id"), str)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
            fixture_manifest["approval_id"],
        )
        is None
    ):
        raise SystemExit("memory qualification fixtures are not precommitted")
    validation_args = {
        "workload_sha256": workload_sha256,
        "memory_sha256": memory_sha256,
        "quality_controls_sha256": sha256(root / "quality_controls.json"),
        "hidden_sha256": workload_policy["manifest"]["hidden_cases_sha256"],
        "dataset_sha256": workload_policy["manifest"]["dataset_manifest_sha256"],
    }
    reference = validate_report(
        args.reference_report, "corrected_reference", **validation_args
    )
    stress = validate_report(args.stress_report, "near_cap_stress", **validation_args)
    reference_report = object_at(args.reference_report)
    stress_report = object_at(args.stress_report)
    for kind, report in (
        ("corrected_reference", reference_report),
        ("near_cap_stress", stress_report),
    ):
        fixture = fixtures.get(kind)
        expected_fixture_evidence = {
            "schema_version": 1,
            "manifest_sha256": fixture_manifest_sha256,
            "approval_id": fixture_manifest["approval_id"],
            "kind": kind,
            "archive_sha256": report.get("archive_sha256"),
            "archive_size_bytes": report.get("archive_size_bytes"),
        }
        if (
            not isinstance(fixture, dict)
            or fixture.get("kind") != kind
            or fixture.get("archive_sha256") != report.get("archive_sha256")
            or fixture.get("archive_size_bytes") != report.get("archive_size_bytes")
            or (kind == "corrected_reference" and fixture.get("stress_extra_mib") is not None)
            or (
                kind == "near_cap_stress"
                and (
                    isinstance(fixture.get("stress_extra_mib"), bool)
                    or not isinstance(fixture.get("stress_extra_mib"), int)
                    or fixture["stress_extra_mib"] <= 0
                )
            )
            or report.get("memory_fixture_manifest_sha256")
            != fixture_manifest_sha256
            or report.get("memory_fixture_approval_id")
            != fixture_manifest["approval_id"]
            or report.get("memory_fixture") != expected_fixture_evidence
        ):
            raise SystemExit(f"{kind} report does not match its precommitted archive")
    common_fields = (
        "work_image_id",
        "judge_image_id",
        "judge_gpu_uuid",
        "model_validation_sha256",
        "runtime_entry_sha256",
        "source_digest",
        "validator_sha256",
        "workload_policy_sha256",
        "memory_policy_sha256",
        "quality_controls_sha256",
        "hidden_cases_sha256",
        "dataset_manifest_sha256",
        *RUNTIME_ATTESTATION_HASH_FIELDS,
    )
    common_identity = {field: reference_report.get(field) for field in common_fields}
    if any(stress_report.get(field) != value for field, value in common_identity.items()):
        raise SystemExit("memory reports do not share one build/GPU/policy cycle")
    if stress_report.get("completed_without_oom") is not True:
        raise SystemExit("near-cap stress did not complete without allocator/driver OOM")
    if (
        not math.isfinite(args.proposed_max_peak_gib)
        or args.proposed_max_peak_gib <= 0
        or args.proposed_max_peak_gib >= 95_830 / 1024
        or args.proposed_headroom_mib <= 0
        or args.safety_margin_mib <= 0
        or args.proposed_row_sentinel_max_growth_mib <= 0
        or args.row_sentinel_safety_margin_mib <= 0
        or not math.isfinite(args.minimum_stress_cap_utilization)
        or not 0.8 <= args.minimum_stress_cap_utilization < 1.0
        or args.max_stress_gap_mib <= 0
    ):
        raise SystemExit("proposed cap/headroom is outside the H100 envelope")
    cap_bytes = args.proposed_max_peak_gib * 2**30
    max_allocated = max(
        reference["torch_peak_allocated_bytes"],
        stress["torch_peak_allocated_bytes"],
    )
    if max_allocated >= cap_bytes:
        raise SystemExit("proposed allocated cap does not exceed measured allocation")
    if any(stress[name] <= reference[name] for name in METRICS):
        raise SystemExit(
            "near-cap stress allocated/reserved/NVML peaks must all exceed reference"
        )
    stress_utilization = stress["torch_peak_allocated_bytes"] / cap_bytes
    stress_gap_bytes = cap_bytes - stress["torch_peak_allocated_bytes"]
    if (
        stress_utilization < args.minimum_stress_cap_utilization
        or stress_gap_bytes > args.max_stress_gap_mib * 2**20
    ):
        raise SystemExit("near-cap stress is not sufficiently close to the proposed cap")
    admission_mib = math.ceil(args.proposed_max_peak_gib * 1024) + args.proposed_headroom_mib
    admission_bytes = admission_mib * 2**20
    safety_bytes = args.safety_margin_mib * 2**20
    max_nonallocated_peak = max(
        reference["torch_peak_reserved_bytes"],
        reference["nvml_peak_device_used_bytes"],
        stress["torch_peak_reserved_bytes"],
        stress["nvml_peak_device_used_bytes"],
    )
    required_headroom_bytes = max(0, max_nonallocated_peak - cap_bytes) + safety_bytes
    if args.proposed_headroom_mib * 2**20 < required_headroom_bytes:
        raise SystemExit(
            "proposed headroom does not cover reserved/NVML overhead plus safety"
        )
    if admission_bytes < max_nonallocated_peak + safety_bytes:
        raise SystemExit("admission floor is below measured non-allocation peak plus safety")
    clean_idle_free_bytes = min(
        reference["nvml_clean_idle_free_bytes"],
        stress["nvml_clean_idle_free_bytes"],
    )
    if admission_bytes + safety_bytes > clean_idle_free_bytes:
        raise SystemExit(
            "admission floor plus safety exceeds qualified clean idle free memory"
        )
    torch_total_bytes = min(
        reference["torch_device_total_bytes"], stress["torch_device_total_bytes"]
    )
    if cap_bytes >= torch_total_bytes:
        raise SystemExit("allocated cap exceeds the qualified PyTorch device total")
    observed_row_growth = max(
        reference["row_sentinel_observed_max_growth_bytes"],
        stress["row_sentinel_observed_max_growth_bytes"],
    )
    observed_row_repeat_growth = max(
        reference["row_sentinel_observed_max_repeat_growth_bytes"],
        stress["row_sentinel_observed_max_repeat_growth_bytes"],
    )
    row_growth_bound_bytes = args.proposed_row_sentinel_max_growth_mib * 2**20
    row_growth_safety_bytes = args.row_sentinel_safety_margin_mib * 2**20
    if row_growth_bound_bytes < observed_row_growth + row_growth_safety_bytes:
        raise SystemExit(
            "proposed row-sentinel bound does not cover observed growth plus safety"
        )
    # The qualification cycle records the device it actually ran on.  The
    # string is provenance for the frozen policy, never an admission gate.
    qualification_hardware = validate_device_envelope(
        reference_report, "memory reference report"
    )
    if qualification_hardware != validate_device_envelope(
        stress_report, "memory stress report"
    ):
        raise SystemExit("reference and stress evidence come from different devices")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "task_id": TASK_ID,
        "profile": PROFILE,
        "hardware": qualification_hardware,
        "workload_policy_sha256": workload_sha256,
        "memory_policy_sha256": memory_sha256,
        "quality_controls_sha256": validation_args["quality_controls_sha256"],
        "hidden_cases_sha256": validation_args["hidden_sha256"],
        "dataset_manifest_sha256": validation_args["dataset_sha256"],
        "reference_report_sha256": sha256(args.reference_report),
        "stress_report_sha256": sha256(args.stress_report),
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "fixture_approval_id": fixture_manifest["approval_id"],
        "common_producer_identity": common_identity,
        "precommitted_fixtures": fixtures,
        "reference_measurements": reference,
        "stress_measurements": stress,
        "proposed_max_peak_gib": args.proposed_max_peak_gib,
        "proposed_headroom_mib": args.proposed_headroom_mib,
        "safety_margin_mib": args.safety_margin_mib,
        "qualified_clean_idle_free_mib": clean_idle_free_bytes // 2**20,
        "qualified_torch_device_total_bytes": torch_total_bytes,
        "max_measured_allocated_bytes": max_allocated,
        "max_measured_reserved_or_nvml_bytes": max_nonallocated_peak,
        "required_headroom_bytes": int(required_headroom_bytes),
        "admission_mib": admission_mib,
        "row_sentinel_observed_max_growth_bytes": observed_row_growth,
        "row_sentinel_observed_max_repeat_growth_bytes": (
            observed_row_repeat_growth
        ),
        "row_sentinel_repeat_growth_guard_bytes": (
            ROW_SENTINEL_REPEAT_GUARD_BYTES
        ),
        "proposed_row_sentinel_max_growth_bytes": row_growth_bound_bytes,
        "row_sentinel_safety_margin_bytes": row_growth_safety_bytes,
        "minimum_stress_cap_utilization": args.minimum_stress_cap_utilization,
        "max_stress_gap_mib": args.max_stress_gap_mib,
        "observed_stress_cap_utilization": stress_utilization,
        "observed_stress_cap_gap_bytes": int(stress_gap_bytes),
    }
    print(write_object(args.output, payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
