"""Shared validation helpers for the workload and memory policy tools.

Hardware strings recorded by these artifacts are provenance: they say which
device a qualification cycle ran on.  Admission is decided by the capability
envelope (MIN_COMPUTE_CAPABILITY / MIN_DEVICE_MEMORY_MIB) and, at evaluation
time, by the visible device's own capacity.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import re
from typing import Any


TASK_ID = "llada_h100_decoder_optimization_v2"
PROFILE = "fast_dllm_v1_llada_12row_stratified520_batch1"
GENERATION_COUNT = 520
REQUESTED_SLOTS = 216_320
TASK_EVAL_TIMEOUT_SECONDS = 7_200
ROW_COUNTS = {
    "table1_gsm8k_5shot_g256": 39,
    "table1_gsm8k_5shot_g512": 39,
    "table1_math_4shot_g256": 65,
    "table1_math_4shot_g512": 65,
    "table1_humaneval_0shot_g256": 65,
    "table1_humaneval_0shot_g512": 65,
    "table1_mbpp_3shot_g256": 65,
    "table1_mbpp_3shot_g512": 65,
    "table4_gsm8k_5shot_g1024": 13,
    "table5_gsm8k_8shot_g256": 13,
    "table5_gsm8k_8shot_g512": 13,
    "table5_gsm8k_8shot_g1024": 13,
}
ROW_BENCHMARKS = {
    row_id: (
        "gsm8k"
        if "gsm8k" in row_id
        else "math"
        if "math" in row_id
        else "humaneval"
        if "humaneval" in row_id
        else "mbpp"
    )
    for row_id in ROW_COUNTS
}
ROW_ORDER = list(ROW_COUNTS)
CATEGORY_DEFINITION = {
    "kind": "dataset",
    "field": "benchmark",
    "categories": ["gsm8k", "math", "humaneval", "mbpp"],
}
CATEGORY_COUNTS = {category: 130 for category in CATEGORY_DEFINITION["categories"]}
INTERACTIVE_SELECTION = "all_frozen_final_v1"
INTERACTIVE_TOTAL_CASES = 520
INTERACTIVE_CATEGORY_BASE_QUOTA = 130
INTERACTIVE_BONUS_CATEGORIES = 0
INTERACTIVE_MINIMUM_PER_ROW = 1


def interactive_private_cohort() -> dict[str, Any]:
    """The contract fields of an experiment manifest's ``private_cohort``."""

    return {
        "selection": INTERACTIVE_SELECTION,
        "category_definition": CATEGORY_DEFINITION,
        "total_cases": INTERACTIVE_TOTAL_CASES,
        "category_base_quota": INTERACTIVE_CATEGORY_BASE_QUOTA,
        "bonus_categories": INTERACTIVE_BONUS_CATEGORIES,
        "minimum_per_row": INTERACTIVE_MINIMUM_PER_ROW,
        "final_generations": GENERATION_COUNT,
        "final_requested_slots": REQUESTED_SLOTS,
    }
HEX = set("0123456789abcdef")
# Capability envelope shared by the launcher, the Judge and these tools.
MIN_COMPUTE_CAPABILITY = "8.0"
MIN_COMPUTE_CAPABILITY_TUPLE = (8, 0)
MIN_DEVICE_MEMORY_MIB = 34816
MIN_DEVICE_MEMORY_BYTES = MIN_DEVICE_MEMORY_MIB * 1024 * 1024
ROW_SENTINEL_METRIC = "max_positive_allocated_reserved_nvml_delta_bytes_v1"
ROW_SENTINEL_REPEAT_METRIC = (
    "max_positive_allocated_reserved_nvml_end_to_repeat_delta_bytes_v1"
)
# Non-release qualification must be able to observe the one-time CUDA allocator
# cache growth induced by a long row.  The largest planned H100 row diagnostic
# retained 10,190,061,568 bytes while an immediate identical repeat retained
# zero additional bytes.  Twelve GiB is only a pre-release observation ceiling:
# the start-to-end release bound is still derived from completed H100 evidence.
QUALIFICATION_ROW_SENTINEL_GUARD_BYTES = 12 * 1024 * 1024 * 1024
# A second, immediate post-row probe is the actual repeated-growth check.  Keep
# it strict and independent of the larger allocator-cache observation ceiling.
ROW_SENTINEL_REPEAT_GUARD_BYTES = 16 * 1024 * 1024
RUNTIME_ATTESTATION_HASH_FIELDS = (
    "runtime_provenance_sha256",
    "sforge_runtime_sha256",
    "sforge_runtime_file_set_sha256",
    "verify_decoder_sha256",
    "candidate_worker_sha256",
    "cohort_sha256",
    "quality_sha256",
    "source_admission_sha256",
)
JUDGE_INVARIANT_CORE_HASH_FIELDS = (
    "verify_decoder_sha256",
    "candidate_worker_sha256",
    "cohort_sha256",
    "quality_sha256",
    "source_admission_sha256",
)
ATTESTATION_HASH_FIELDS = (
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private_data_dir(task_root: Path) -> Path:
    """The Judge-only private_data/ directory (hidden cases, references, policies).

    The in-repository images/judge/private_data/ when it is present, else the
    task's dataset asset that assets/fetch.sh places at
    <INFRATHON_ASSETS_ROOT>/<task directory>/judge/private_data (default asset root:
    ~/.cache/infrathon/assets).
    """
    in_repository = task_root / "images/judge/private_data"
    if in_repository.is_dir() and any(in_repository.iterdir()):
        return in_repository
    assets_root = os.environ.get("INFRATHON_ASSETS_ROOT") or str(
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "infrathon/assets"
    )
    return Path(assets_root).expanduser() / task_root.name / "judge/private_data"


def canonical_hidden_bytes(task_root: Path) -> bytes:
    """Read source plain/gzip forms and return the canonical JSON bytes."""
    private = private_data_dir(task_root)
    plain = private / "hidden_cases.json"
    compressed = private / "hidden_cases.json.gz"
    values: list[bytes] = []
    if plain.exists() or plain.is_symlink():
        if plain.is_symlink() or not plain.is_file():
            raise ValueError("plain hidden workload is unsafe")
        values.append(plain.read_bytes())
    if compressed.exists() or compressed.is_symlink():
        if compressed.is_symlink() or not compressed.is_file():
            raise ValueError("compressed hidden workload is unsafe")
        try:
            value = gzip.decompress(compressed.read_bytes())
        except (OSError, EOFError) as exc:
            raise ValueError("compressed hidden workload is invalid") from exc
        if len(value) > 128 * 1024 * 1024:
            raise ValueError("decompressed hidden workload exceeds its bound")
        values.append(value)
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError("plain/compressed hidden workload forms are missing or differ")
    return values[0]


def canonical_hidden_sha256(task_root: Path) -> str:
    return hashlib.sha256(canonical_hidden_bytes(task_root)).hexdigest()


def object_at(path: Path, label: str = "JSON file") -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def write_object(path: Path, value: dict[str, Any]) -> str:
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def positive_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def validate_runtime_provenance(value: Any) -> dict[str, Any]:
    """Validate the complete, canonical SForge/Judge invariant snapshot."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("runtime provenance snapshot schema is invalid")
    sforge = value.get("sforge_runtime")
    core = value.get("judge_invariant_core")
    if not isinstance(sforge, dict) or not isinstance(core, dict):
        raise ValueError("runtime provenance snapshot sections are missing")
    files = sforge.get("files")
    if (
        sforge.get("scope") != "all_sforge_python"
        or not is_sha256(sforge.get("sha256"))
        or not is_sha256(sforge.get("file_set_sha256"))
        or not isinstance(files, list)
        or not files
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"sforge/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py", item)
            is None
            for item in files
        )
        or files != sorted(set(files))
        or isinstance(sforge.get("file_count"), bool)
        or sforge.get("file_count") != len(files)
        or sforge.get("file_set_sha256")
        != hashlib.sha256(("\n".join(files) + "\n").encode()).hexdigest()
    ):
        raise ValueError("SForge runtime provenance is invalid")
    if set(core) != set(JUDGE_INVARIANT_CORE_HASH_FIELDS) or any(
        not is_sha256(core.get(field)) for field in JUDGE_INVARIANT_CORE_HASH_FIELDS
    ):
        raise ValueError("Judge invariant-core provenance is invalid")
    return value


def validate_attested_runtime(report: dict[str, Any]) -> dict[str, Any]:
    """Cross-check flattened report fields against its complete runtime snapshot."""
    snapshot = validate_runtime_provenance(
        {
            "schema_version": 1,
            "sforge_runtime": report.get("sforge_runtime"),
            "judge_invariant_core": report.get("judge_invariant_core"),
        }
    )
    sforge = snapshot["sforge_runtime"]
    core = snapshot["judge_invariant_core"]
    expected = {
        "sforge_runtime_sha256": sforge["sha256"],
        "sforge_runtime_file_set_sha256": sforge["file_set_sha256"],
        **core,
    }
    if any(report.get(field) != value for field, value in expected.items()):
        raise ValueError("flattened runtime provenance differs from its full snapshot")
    return snapshot


def validate_fullscale_attestation(
    report: dict[str, Any],
    *,
    mode: str,
    expected_workload_policy_sha256: str,
    expected_memory_policy_sha256: str,
    expected_quality_controls_sha256: str,
    expected_hidden_sha256: str,
    expected_dataset_manifest_sha256: str,
) -> None:
    """Bind a non-release producer report to this exact task and policy state."""
    if mode not in {"qualification", "calibration", "quality_control"}:
        raise ValueError("non-release attestation mode is invalid")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "PASS"
        or report.get("mode") != mode
        or report.get("release_evidence") is not False
        or report.get("reference_replay") is not False
        or report.get("task_id") != TASK_ID
        or not isinstance(report.get("hardware"), str)
        or not report.get("hardware")
        or report.get("visible_device_count") != 1
        or report.get("profile") != PROFILE
        or report.get("generation_count") != GENERATION_COUNT
        or report.get("requested_slots") != REQUESTED_SLOTS
        or report.get("row_counts") != ROW_COUNTS
        or report.get("workload_policy_sha256")
        != expected_workload_policy_sha256
        or report.get("memory_policy_sha256") != expected_memory_policy_sha256
        or report.get("quality_controls_sha256")
        != expected_quality_controls_sha256
        or report.get("hidden_cases_sha256") != expected_hidden_sha256
        or report.get("dataset_manifest_sha256")
        != expected_dataset_manifest_sha256
        or any(not is_sha256(report.get(field)) for field in ATTESTATION_HASH_FIELDS)
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(report.get("work_image_id", ""))
        )
        is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(report.get("judge_image_id", ""))
        )
        is None
        or re.fullmatch(
            r"GPU-[0-9A-Fa-f-]+", str(report.get("judge_gpu_uuid", ""))
        )
        is None
        or not isinstance(report.get("run_id"), str)
        or not report["run_id"]
        or isinstance(report.get("archive_size_bytes"), bool)
        or not isinstance(report.get("archive_size_bytes"), int)
        or report["archive_size_bytes"] <= 0
    ):
        raise ValueError("stratified-520 attestation identity/provenance is invalid")


def validate_workload_policy(policy: dict[str, Any], *, require_frozen: bool = False) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("task_id") != TASK_ID
        or policy.get("profile") != PROFILE
    ):
        raise ValueError("workload policy identity mismatch")
    manifest = policy.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("workload policy manifest section is missing")
    if manifest.get("row_counts") != ROW_COUNTS:
        raise ValueError("workload policy row counts are not the exact stratified-520 profile")
    if manifest.get("row_order") != ROW_ORDER:
        raise ValueError("workload policy row order is not explicit and canonical")
    if manifest.get("row_benchmarks") != ROW_BENCHMARKS:
        raise ValueError("workload policy row benchmark mapping is invalid")
    if (
        manifest.get("generation_count") != GENERATION_COUNT
        or manifest.get("requested_slots") != REQUESTED_SLOTS
        or manifest.get("context_exclusions") != ["mbpp:test:482/task:493"]
        or not is_sha256(manifest.get("hidden_cases_sha256"))
        or not is_sha256(manifest.get("dataset_manifest_sha256"))
    ):
        raise ValueError("workload policy manifest commitment is invalid")
    interactive = policy.get("interactive")
    final = policy.get("final")
    if not isinstance(interactive, dict) or not isinstance(final, dict):
        raise ValueError("workload policy execution sections are missing")
    if (
        interactive.get("selection") != INTERACTIVE_SELECTION
        or interactive.get("category_definition") != CATEGORY_DEFINITION
        or interactive.get("total_cases") != INTERACTIVE_TOTAL_CASES
        or interactive.get("row_atomic") is not True
        or interactive.get("timeout_policy") != "same_as_final"
        or interactive.get("category_base_quota")
        != INTERACTIVE_CATEGORY_BASE_QUOTA
        or interactive.get("bonus_categories") != INTERACTIVE_BONUS_CATEGORIES
        or interactive.get("minimum_per_row") != INTERACTIVE_MINIMUM_PER_ROW
        or "per_row_quotas" in interactive
        or "per_category_quotas" in interactive
    ):
        raise ValueError("interactive complete 520-case cohort policy is invalid")
    canonical_categories = {category: 0 for category in CATEGORY_COUNTS}
    for row_id, benchmark in ROW_BENCHMARKS.items():
        canonical_categories[benchmark] += ROW_COUNTS[row_id]
    if (
        canonical_categories != CATEGORY_COUNTS
        or INTERACTIVE_CATEGORY_BASE_QUOTA * len(CATEGORY_COUNTS)
        + INTERACTIVE_BONUS_CATEGORIES
        != INTERACTIVE_TOTAL_CASES
        or any(
            sum(1 for value in ROW_BENCHMARKS.values() if value == category)
            > INTERACTIVE_CATEGORY_BASE_QUOTA
            for category in CATEGORY_COUNTS
        )
    ):
        raise AssertionError("hard-coded dataset category totals are inconsistent")
    if (
        final.get("selection") != "all"
        or final.get("row_atomic") is not True
        or final.get("max_infrastructure_retries_per_row") != 1
        or final.get("candidate_failures_retriable") is not False
    ):
        raise ValueError("final row lifecycle policy is invalid")
    qualification = policy.get("qualification_status")
    release = policy.get("release_status")
    if qualification == "UNQUALIFIED" and release == "PENDING":
        if require_frozen:
            raise ValueError("workload policy is not qualified and frozen")
        return
    if qualification != "QUALIFIED" or release not in {"PROVISIONAL", "FROZEN"}:
        raise ValueError("workload policy status transition is invalid")
    qualification_evidence = policy.get("qualification")
    if (
        not isinstance(qualification_evidence, dict)
        or not is_sha256(qualification_evidence.get("observations_sha256"))
        or any(
            not positive_number(qualification_evidence.get(field))
            for field in (
                "max_observed_case_seconds",
                "max_observed_row_seconds",
                "max_observed_elapsed_seconds",
            )
        )
    ):
        raise ValueError("qualified workload policy lacks timing observations")
    if any(
        not positive_number(final.get(field))
        for field in (
            "case_timeout_seconds",
            "row_timeout_seconds",
            "inner_timeout_seconds",
            "outer_timeout_seconds",
        )
    ):
        raise ValueError("qualified workload policy has incomplete final deadlines")
    if not positive_number(final.get("wrapper_margin_seconds")):
        raise ValueError("qualified workload policy has no wrapper packaging margin")
    if any(
        not positive_number(interactive.get(field))
        for field in ("case_timeout_seconds", "judge_timeout_seconds", "cooldown_seconds")
    ):
        raise ValueError("qualified workload policy has incomplete interactive limits")
    if not (
        float(interactive["case_timeout_seconds"])
        < float(interactive["judge_timeout_seconds"])
        and float(final["case_timeout_seconds"])
        < float(final["row_timeout_seconds"])
        < float(final["inner_timeout_seconds"])
        < float(final["outer_timeout_seconds"])
        <= TASK_EVAL_TIMEOUT_SECONDS
        and float(final["outer_timeout_seconds"])
        - float(final["inner_timeout_seconds"])
        >= float(final["wrapper_margin_seconds"])
    ):
        raise ValueError("qualified workload deadlines are not coherently nested")
    if (
        interactive["case_timeout_seconds"] != final["case_timeout_seconds"]
        or interactive["judge_timeout_seconds"] != final["inner_timeout_seconds"]
    ):
        raise ValueError("interactive and final complete-workload deadlines must match")
    if require_frozen and release != "FROZEN":
        raise ValueError("workload release policy is not frozen")
    if release == "FROZEN":
        if not is_sha256(policy.get("provisional_parent_sha256")):
            raise ValueError("frozen workload policy lacks its provisional parent hash")
        quality = policy.get("quality_policy")
        thresholds = quality.get("thresholds") if isinstance(quality, dict) else None
        timeout_release = policy.get("timeout_release")
        if (
            not isinstance(quality, dict)
            or quality.get("status") != "QUALIFIED"
            or quality.get("provisional_parent_sha256")
            != policy.get("provisional_parent_sha256")
            or any(
                not is_sha256(quality.get(field))
                for field in (
                    "producer_attestation_sha256",
                    "control_attestation_sha256",
                    "quality_controls_sha256",
                    "producer_archive_sha256",
                )
            )
            or not isinstance(thresholds, dict)
            or thresholds.get("schema_version") != 1
            or thresholds.get("metric")
            != "producer_relative_min_retention_ppm_v1"
            or thresholds.get("derivation")
            != "midpoint_min_positive_max_negative_v1"
            or any(
                isinstance(thresholds.get(field), bool)
                or not isinstance(thresholds.get(field), int)
                or thresholds[field] <= 0
                for field in (
                    "minimum_score_ppm",
                    "minimum_positive_score_ppm",
                    "observed_separation_ppm",
                    "required_separation_ppm",
                    "positive_margin_ppm",
                    "negative_margin_ppm",
                )
            )
            or isinstance(thresholds.get("maximum_negative_score_ppm"), bool)
            or not isinstance(thresholds.get("maximum_negative_score_ppm"), int)
            or thresholds["maximum_negative_score_ppm"] < 0
            or not (
                thresholds["maximum_negative_score_ppm"]
                < thresholds["minimum_score_ppm"]
                <= thresholds["minimum_positive_score_ppm"]
            )
            or thresholds["observed_separation_ppm"]
            != thresholds["minimum_positive_score_ppm"]
            - thresholds["maximum_negative_score_ppm"]
            or thresholds["observed_separation_ppm"]
            < thresholds["required_separation_ppm"]
            or not isinstance(timeout_release, dict)
            or timeout_release.get("status") != "QUALIFIED"
            or not is_sha256(timeout_release.get("producer_attestation_sha256"))
            or any(
                not positive_number(timeout_release.get(field))
                for field in (
                    "producer_max_case_seconds",
                    "producer_max_row_seconds",
                    "producer_elapsed_seconds",
                    "qualification_max_case_seconds",
                    "qualification_max_row_seconds",
                    "qualification_max_elapsed_seconds",
                    "approved_wrapper_margin_seconds",
                )
            )
        ):
            raise ValueError("frozen workload policy quality/parent chain is invalid")


def validate_memory_policy(policy: dict[str, Any], *, require_frozen: bool = False) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("task_id") != TASK_ID
        or not isinstance(policy.get("qualification_hardware"), str)
        or not policy.get("qualification_hardware")
        or policy.get("min_compute_capability") != MIN_COMPUTE_CAPABILITY
        or policy.get("min_device_memory_mib") != MIN_DEVICE_MEMORY_MIB
    ):
        raise ValueError("memory policy identity mismatch")
    legacy = policy.get("historical_a6000_policy")
    if (
        not isinstance(legacy, dict)
        or legacy.get("max_peak_gib") != 47.0
        or legacy.get("applicability") != "NONE"
        or legacy.get("status") != "UNCALIBRATED_FOR_H100"
    ):
        raise ValueError("historical A6000 cap is not explicitly non-applicable")
    qualification = policy.get("qualification_status")
    release = policy.get("release_status")
    row_sentinel = policy.get("row_sentinel")
    stress = policy.get("near_cap_stress")
    if (
        not isinstance(row_sentinel, dict)
        or row_sentinel.get("metric") != ROW_SENTINEL_METRIC
        or row_sentinel.get("repeat_metric") != ROW_SENTINEL_REPEAT_METRIC
        or row_sentinel.get("qualification_guard_bytes")
        != QUALIFICATION_ROW_SENTINEL_GUARD_BYTES
        or row_sentinel.get("repeat_growth_guard_bytes")
        != ROW_SENTINEL_REPEAT_GUARD_BYTES
        or not isinstance(stress, dict)
    ):
        raise ValueError("row-sentinel/stress policy is missing")
    if qualification == "UNQUALIFIED" and release == "PENDING":
        if any(
            policy.get(key) is not None
            for key in (
                "qualified_max_peak_gib",
                "headroom_mib",
                "qualified_clean_idle_free_mib",
                "safety_margin_mib",
                "qualification_run_sha256",
                "approval_id",
            )
        ):
            raise ValueError("unqualified memory policy must not contain invented limits")
        if (
            row_sentinel.get("status") != "PENDING"
            or any(
                row_sentinel.get(key) is not None
                for key in (
                    "max_growth_bytes",
                    "observed_max_growth_bytes",
                    "observed_max_repeat_growth_bytes",
                    "safety_margin_bytes",
                )
            )
            or stress
            != {
                "status": "PENDING",
                "minimum_cap_utilization": None,
                "maximum_gap_mib": None,
                "observed_cap_utilization": None,
            }
        ):
            raise ValueError("unqualified row-sentinel/stress policy invents limits")
        if require_frozen:
            raise ValueError("memory policy is not qualified and frozen")
        return
    if qualification != "QUALIFIED" or release != "FROZEN":
        raise ValueError("memory policy status transition is invalid")
    cap = policy.get("qualified_max_peak_gib")
    headroom = policy.get("headroom_mib")
    clean_idle_free = policy.get("qualified_clean_idle_free_mib")
    safety_margin = policy.get("safety_margin_mib")
    if not positive_number(cap) or float(cap) >= 95_830 / 1024:
        raise ValueError("H100 allocated-memory cap is invalid")
    if isinstance(headroom, bool) or not isinstance(headroom, int) or headroom <= 0:
        raise ValueError("H100 runtime headroom is invalid")
    if (
        isinstance(clean_idle_free, bool)
        or not isinstance(clean_idle_free, int)
        or clean_idle_free <= 0
        or isinstance(safety_margin, bool)
        or not isinstance(safety_margin, int)
        or safety_margin <= 0
    ):
        raise ValueError("clean-idle-free/safety qualification is invalid")
    admission_mib = math.ceil(float(cap) * 1024) + headroom
    if admission_mib + safety_margin > clean_idle_free:
        raise ValueError("qualification evidence is not self-consistent with the safety margin")
    if not is_sha256(policy.get("qualification_run_sha256")):
        raise ValueError("memory qualification hash is invalid")
    if not isinstance(policy.get("approval_id"), str) or not policy["approval_id"]:
        raise ValueError("memory policy approval is missing")
    max_growth = row_sentinel.get("max_growth_bytes")
    observed_growth = row_sentinel.get("observed_max_growth_bytes")
    observed_repeat_growth = row_sentinel.get(
        "observed_max_repeat_growth_bytes"
    )
    growth_safety = row_sentinel.get("safety_margin_bytes")
    if (
        row_sentinel.get("status") != "FROZEN"
        or isinstance(max_growth, bool)
        or not isinstance(max_growth, int)
        or max_growth <= 0
        or isinstance(observed_growth, bool)
        or not isinstance(observed_growth, int)
        or observed_growth < 0
        or isinstance(observed_repeat_growth, bool)
        or not isinstance(observed_repeat_growth, int)
        or observed_repeat_growth < 0
        or observed_repeat_growth > ROW_SENTINEL_REPEAT_GUARD_BYTES
        or isinstance(growth_safety, bool)
        or not isinstance(growth_safety, int)
        or growth_safety <= 0
        or max_growth < observed_growth + growth_safety
    ):
        raise ValueError("H100 row-sentinel growth bound is not evidence-backed")
    minimum_utilization = stress.get("minimum_cap_utilization")
    maximum_gap_mib = stress.get("maximum_gap_mib")
    observed_utilization = stress.get("observed_cap_utilization")
    if (
        stress.get("status") != "QUALIFIED"
        or not positive_number(minimum_utilization)
        or not 0.8 <= float(minimum_utilization) < 1.0
        or isinstance(maximum_gap_mib, bool)
        or not isinstance(maximum_gap_mib, int)
        or maximum_gap_mib <= 0
        or not positive_number(observed_utilization)
        or float(observed_utilization) < float(minimum_utilization)
        or float(observed_utilization) >= 1.0
    ):
        raise ValueError("near-cap stress qualification is invalid")
    if require_frozen and release != "FROZEN":
        raise ValueError("memory release policy is not frozen")




def capability_rank(value: object) -> int:
    """Rank a "major.minor" compute capability for ordered comparison."""
    text = str(value)
    if re.fullmatch(r"[0-9]+\.[0-9]+", text) is None:
        return -1
    major, minor = text.split(".")
    return int(major) * 1000 + int(minor)


def validate_device_envelope(record: dict[str, Any], label: str) -> str:
    """Accept any device inside the capability envelope; return its name.

    The GPU model is measurement provenance.  Admission is decided by compute
    capability and device capacity, never by a model name.  Reports written
    before the Judge recorded a full inventory carry only ``gpu_name``; those
    are accepted as provenance-only evidence.
    """
    hardware = record.get("hardware")
    name = hardware.get("name") if isinstance(hardware, dict) else record.get("gpu_name")
    if isinstance(hardware, str):
        name = hardware
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label} does not record the measured device")
    if not isinstance(hardware, dict):
        return name
    capability = hardware.get("capability")
    total_memory = hardware.get("total_memory_bytes")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in capability)
        or tuple(capability) < MIN_COMPUTE_CAPABILITY_TUPLE
        or isinstance(total_memory, bool)
        or not isinstance(total_memory, int)
        or total_memory < MIN_DEVICE_MEMORY_BYTES
    ):
        raise ValueError(f"{label} device is outside the capability envelope: {name}")
    return name


def public_memory_contract(policy: dict[str, Any]) -> dict[str, Any]:
    """Expose only the candidate allocation contract, never evidence.

    The enforced peak cap is device-relative, so the public contract publishes
    the capability envelope and the reserve subtracted from the visible
    device's own capacity.  The qualification numbers are labelled provenance.
    """
    validate_memory_policy(policy)
    frozen = policy.get("qualification_status") == "QUALIFIED"
    reserve = (
        int(policy["headroom_mib"]) + int(policy["safety_margin_mib"])
        if frozen
        else None
    )
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "release_status": "FROZEN" if frozen else "PENDING",
        "min_compute_capability": MIN_COMPUTE_CAPABILITY,
        "min_device_memory_mib": MIN_DEVICE_MEMORY_MIB,
        "memory_reserve_mib": reserve,
        "peak_cap_formula": (
            "floor(total_device_memory_mib) - memory_reserve_mib, evaluated on "
            "the device the Judge is running on"
        ),
        "qualification_hardware": policy.get("qualification_hardware"),
        "qualified_max_peak_gib": (
            policy.get("qualified_max_peak_gib") if frozen else None
        ),
    }
