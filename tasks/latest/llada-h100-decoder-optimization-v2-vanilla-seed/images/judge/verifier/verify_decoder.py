#!/usr/bin/env python3
"""Root controller for real-model LLaDA decoder private evaluation.

Hidden answers, executable tests, scoring, and the authoritative wall clock
remain in this process.  Candidate Python runs only in a persistent gold-free
worker and receives prompt token ids plus the public generation shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import selectors
import signal
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer

from cohort import (
    CATEGORY_DEFINITION,
    INTERACTIVE_BONUS_CATEGORIES,
    INTERACTIVE_CATEGORY_BASE_QUOTA,
    INTERACTIVE_MINIMUM_PER_ROW,
    INTERACTIVE_SELECTION,
    INTERACTIVE_TOTAL_CASES,
    ROW_PLAN,
    WORKLOAD_POLICY,
    context_from_metadata,
    select_cohort,
)
from row_lifecycle import (
    CandidateFailure,
    EvaluationIdentity,
    InfrastructureFailure,
    evaluation_checkpoint_root,
    load_or_create_absolute_deadline,
    run_row_with_retry,
    validate_completed_rows,
)
from nvml_sampler import (
    NvmlSamplerError,
    PeakMemorySampler,
    bind_single_visible_nvml_used_probe,
    fixed_shape_memory_stable,
    nvml_driver_version,
)
from source_admission import AdmissionError, admit, locate_lab
from quality_threshold import (
    AGGREGATE_FLOOR_PPM,
    BENCHMARK_FLOOR_PPM,
    MAX_USEFUL_TOKENS_RETENTION_PPM,
    QUALITY_GATE,
    benchmark_retention,
    fixed_gate_failures,
    measure_quality_retention,
)


JUDGE_ROOT = Path("/opt/decoder_judge")
HIDDEN_PATH = JUDGE_ROOT / "private_data/hidden_cases.json"
REFERENCE_PATH = JUDGE_ROOT / "private_data/reference_quality.json"
REFERENCE_CACHE_DIR = JUDGE_ROOT / "private_data/reference_cache"
WORKLOAD_POLICY_PATH = JUDGE_ROOT / "private_data/workload_policy.json"
MEMORY_POLICY_PATH = JUDGE_ROOT / "private_data/memory_policy.json"
QUALITY_CONTROLS_PATH = JUDGE_ROOT / "private_data/quality_controls.json"
PROTECTED_PATH = JUDGE_ROOT / "protected_hashes.txt"
WORKER_PATH = JUDGE_ROOT / "verifier/candidate_worker.py"
PUBLIC_CASES_PATH = Path("/workspace/public_tests/public_cases.json")
MASK_TOKEN_ID = 126336
EOS_TOKEN_ID = 126081
VOCAB_SIZE = 126464
MAX_PROTOCOL_LINE = 2 * 1024 * 1024
WORKER_START_TIMEOUT = 240.0
ROW_SENTINEL_REPEAT_GROWTH_GUARD_BYTES = 16 * 1024 * 1024
EVALUATION_CONTEXT_PATH = Path("/tmp/.sforge-evaluation-context.json")
CHECKPOINT_ROOT = Path("/logs/verifier/checkpoints")
CALIBRATION_PROVENANCE_HASHES = (
    "candidate_archive_sha256",
    "canonical_candidate_archive_sha256",
    "candidate_source_sha256",
    "evaluation_validator_sha256",
    "calibration_tool_sha256",
    "image_manifest_sha256",
    "source_digest",
    "judge_runtime_tree_sha256",
    "model_validation_sha256",
    "evaluation_attestation_sha256",
    "private_artifact_sha256",
    "quality_controls_sha256",
    "sforge_report_sha256",
    "private_payload_sha256",
    "workload_policy_sha256",
    "memory_policy_sha256",
    "runtime_provenance_sha256",
    "sforge_runtime_sha256",
    "sforge_runtime_file_set_sha256",
    "verify_decoder_sha256",
    "candidate_worker_sha256",
    "cohort_sha256",
    "quality_sha256",
    "source_admission_sha256",
)
JUDGE_INVARIANT_CORE = {
    "verify_decoder_sha256": JUDGE_ROOT / "verifier/verify_decoder.py",
    "candidate_worker_sha256": JUDGE_ROOT / "verifier/candidate_worker.py",
    "cohort_sha256": JUDGE_ROOT / "verifier/cohort.py",
    "quality_sha256": JUDGE_ROOT / "verifier/quality.py",
    "source_admission_sha256": JUDGE_ROOT / "verifier/source_admission.py",
}
# Reference-binding lineage.  verify_decoder.py is part of the Judge invariant
# core, so any byte change would orphan the frozen producer reference.  A
# revision whose generation, timing, quality and memory paths are unchanged
# names its predecessor here, and a reference calibrated under that
# predecessor stays admissible; every other core file must still match
# exactly.  6a8bd4f0... is verify_decoder.py at d8a28b56 (the r1
# requalification), which keyed the reference on the host driver version and
# the image ids.  This revision changes only reference selection/refusal (GPU
# model + compute capability + software ABI) and the report fields that record
# it (owner decision 2026-10-12: no numeric change, no recalibration).
SCORING_EQUIVALENT_VERIFY_DECODER_SHA256 = frozenset(
    {"6a8bd4f02121e0b1e3aeec44a54036de834c86de03d02def2140207b2a3f3226"}
)
# A producer reference is comparable on another device only if the hardware
# that executes the kernels and the software that chooses them match.  The
# hardware half is the reference profile (GPU model + compute capability); the
# software half is this ABI, the one every cached reference was measured under
# (Work image pytorch/pytorch:2.7.1-cuda12.8-cudnn9 + transformers 4.49.0).
# Driver version and image ids are recorded provenance and never refuse a
# reference.  Changing this ABI is an invariant-core change and therefore a
# recalibration.
REFERENCE_SOFTWARE_ABI = {
    "cuda_runtime": "12.8",
    "torch": "2.7.1+cu128",
    "transformers": "4.49.0",
}
REFERENCE_PROFILE_FIELDS = ("compute_capability", "gpu_name")
RUNTIME_CONTEXT_FIELDS = {
    "runtime_provenance_sha256": "RUNTIME_PROVENANCE_SHA256",
    "sha256": "SFORGE_RUNTIME_SHA256",
    "file_set_sha256": "SFORGE_RUNTIME_FILE_SET_SHA256",
    "file_count": "SFORGE_RUNTIME_FILE_COUNT",
}
MEMORY_FIXTURE_CONTEXT_FIELDS = (
    "MEMORY_FIXTURE_MANIFEST_SHA256",
    "MEMORY_FIXTURE_APPROVAL_ID",
    "MEMORY_FIXTURE_KIND",
    "MEMORY_FIXTURE_ARCHIVE_SHA256",
    "MEMORY_FIXTURE_ARCHIVE_SIZE_BYTES",
)

TASK_ID = "llada_h100_decoder_optimization_v2"
WORKLOAD_PROFILE = "fast_dllm_v1_llada_12row_stratified520_batch1"
# Capability envelope.  The task admits any CUDA device that clears these two
# floors; GPU model, UUID and clocks are recorded as measurement provenance and
# never gate.  The memory floor is set by the largest scored row (GSM8K 8-shot
# G1024): ~20 GiB peak allocated for the BF16 LLaDA-8B weights plus the
# retained decode cache, measured on H100 NVL, doubled for allocator slack.
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 34 * 1024**3
GENERATION_COUNT = 520
REQUESTED_SLOTS = 216_320
HEX = set("0123456789abcdef")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_at(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} is not positive and finite")
    return number


def _positive_env(name: str) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise ValueError(f"{name} is required")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} is not numeric") from exc
    return _positive(value, name)


def validate_workload_policy(policy: dict[str, Any], *, bootstrap: bool) -> None:
    manifest = policy.get("manifest")
    interactive = policy.get("interactive")
    final = policy.get("final")
    expected_rows = {row_id: plan.canonical_count for row_id, plan in ROW_PLAN.items()}
    if (
        policy != WORKLOAD_POLICY
        or policy.get("task_id") != TASK_ID
        or policy.get("profile") != WORKLOAD_PROFILE
        or not isinstance(manifest, dict)
        or manifest.get("row_counts") != expected_rows
        or manifest.get("generation_count") != GENERATION_COUNT
        or manifest.get("requested_slots") != REQUESTED_SLOTS
        or manifest.get("hidden_cases_sha256") != sha256(HIDDEN_PATH)
        or not isinstance(interactive, dict)
        or interactive.get("category_definition") != CATEGORY_DEFINITION
        or interactive.get("selection") != INTERACTIVE_SELECTION
        or interactive.get("total_cases") != INTERACTIVE_TOTAL_CASES
        or interactive.get("row_atomic") is not True
        or interactive.get("timeout_policy") != "same_as_final"
        or interactive.get("category_base_quota")
        != INTERACTIVE_CATEGORY_BASE_QUOTA
        or interactive.get("bonus_categories") != INTERACTIVE_BONUS_CATEGORIES
        or interactive.get("minimum_per_row") != INTERACTIVE_MINIMUM_PER_ROW
        or "per_row_quotas" in interactive
        or "per_category_quotas" in interactive
        or not isinstance(final, dict)
        or final.get("selection") != "all"
        or final.get("row_atomic") is not True
        or final.get("max_infrastructure_retries_per_row") != 1
        or final.get("candidate_failures_retriable") is not False
    ):
        raise ValueError("trusted workload policy does not match the stratified-520 manifest")
    if bootstrap:
        if (
            policy.get("qualification_status") != "UNQUALIFIED"
            or policy.get("release_status") != "PENDING"
        ):
            raise ValueError("bootstrap mode is only for the unqualified policy")
        return
    if (
        policy.get("qualification_status") != "QUALIFIED"
        or policy.get("release_status") not in {"PROVISIONAL", "FROZEN"}
    ):
        raise ValueError("workload policy is not qualified")
    if policy.get("release_status") == "FROZEN":
        parent = policy.get("provisional_parent_sha256")
        quality_policy = policy.get("quality_policy")
        thresholds = (
            quality_policy.get("thresholds")
            if isinstance(quality_policy, dict)
            else None
        )
        timeout_release = policy.get("timeout_release")
        if (
            not isinstance(parent, str)
            or len(parent) != 64
            or any(ch not in HEX for ch in parent)
            or not isinstance(quality_policy, dict)
            or quality_policy.get("status") != "QUALIFIED"
            or quality_policy.get("provisional_parent_sha256") != parent
            or not isinstance(thresholds, dict)
            or thresholds.get("metric")
            != "producer_relative_min_retention_ppm_v1"
            or thresholds.get("derivation")
            != "midpoint_min_positive_max_negative_v1"
            or isinstance(thresholds.get("minimum_score_ppm"), bool)
            or not isinstance(thresholds.get("minimum_score_ppm"), int)
            or thresholds["minimum_score_ppm"] <= 0
            or not isinstance(timeout_release, dict)
            or timeout_release.get("status") != "QUALIFIED"
            or timeout_release.get("approved_wrapper_margin_seconds")
            != final.get("wrapper_margin_seconds")
        ):
            raise ValueError("frozen workload policy lacks a valid provisional parent chain")
    for section, fields in (
        (interactive, ("case_timeout_seconds", "judge_timeout_seconds", "cooldown_seconds")),
        (
            final,
            (
                "case_timeout_seconds",
                "row_timeout_seconds",
                "inner_timeout_seconds",
                "outer_timeout_seconds",
                "wrapper_margin_seconds",
            ),
        ),
    ):
        for field in fields:
            _positive(section.get(field), f"workload policy {field}")
    if not (
        float(interactive["case_timeout_seconds"])
        < float(interactive["judge_timeout_seconds"])
        and float(final["case_timeout_seconds"])
        < float(final["row_timeout_seconds"])
        < float(final["inner_timeout_seconds"])
        < float(final["outer_timeout_seconds"])
        <= 7_200
        and float(final["outer_timeout_seconds"])
        - float(final["inner_timeout_seconds"])
        >= float(final["wrapper_margin_seconds"])
    ):
        raise ValueError("workload policy deadlines are not coherently nested")


def validate_memory_policy(policy: dict[str, Any], *, bootstrap: bool) -> int | None:
    """Return the device-memory reserve in MiB, or None while unqualified.

    The policy records the GPU the qualification cycle ran on, but that string
    is provenance: the enforced peak cap is derived on-device from the visible
    device capacity minus this reserve, so any device inside the capability
    envelope is gated by its own capacity rather than by an H100 constant.
    """
    qualification_hardware = policy.get("qualification_hardware")
    if (
        policy.get("schema_version") != 1
        or policy.get("task_id") != TASK_ID
        or not isinstance(qualification_hardware, str)
        or not qualification_hardware
    ):
        raise ValueError("trusted memory policy identity mismatch")
    legacy = policy.get("historical_a6000_policy")
    row_sentinel = policy.get("row_sentinel")
    if (
        not isinstance(row_sentinel, dict)
        or row_sentinel.get("metric")
        != "max_positive_allocated_reserved_nvml_delta_bytes_v1"
        or row_sentinel.get("repeat_metric")
        != "max_positive_allocated_reserved_nvml_end_to_repeat_delta_bytes_v1"
        or row_sentinel.get("qualification_guard_bytes")
        != 12 * 1024 * 1024 * 1024
        or row_sentinel.get("repeat_growth_guard_bytes")
        != ROW_SENTINEL_REPEAT_GROWTH_GUARD_BYTES
    ):
        raise ValueError("trusted row-sentinel policy is invalid")
    if (
        not isinstance(legacy, dict)
        or legacy.get("max_peak_gib") != 47.0
        or legacy.get("applicability") != "NONE"
        or legacy.get("status") != "UNCALIBRATED_FOR_H100"
    ):
        raise ValueError("historical A6000 cap is not explicitly non-applicable")
    if bootstrap:
        if (
            policy.get("qualification_status") != "UNQUALIFIED"
            or policy.get("release_status") != "PENDING"
            or policy.get("qualified_max_peak_gib") is not None
            or row_sentinel.get("status") != "PENDING"
            or row_sentinel.get("max_growth_bytes") is not None
            or row_sentinel.get("observed_max_repeat_growth_bytes") is not None
        ):
            raise ValueError("bootstrap mode requires an unqualified memory policy")
        return None
    if (
        policy.get("qualification_status") != "QUALIFIED"
        or policy.get("release_status") != "FROZEN"
    ):
        raise ValueError("memory policy is not qualified and frozen")
    # Provenance only: the peak actually observed on the qualification device.
    qualified_cap = _positive(
        policy.get("qualified_max_peak_gib"), "qualified peak-memory observation"
    )
    headroom = policy.get("headroom_mib")
    clean_idle_free = policy.get("qualified_clean_idle_free_mib")
    safety_margin = policy.get("safety_margin_mib")
    if isinstance(headroom, bool) or not isinstance(headroom, int) or headroom <= 0:
        raise ValueError("runtime headroom is invalid")
    if (
        isinstance(clean_idle_free, bool)
        or not isinstance(clean_idle_free, int)
        or isinstance(safety_margin, bool)
        or not isinstance(safety_margin, int)
        or clean_idle_free <= 0
        or safety_margin <= 0
        or math.ceil(qualified_cap * 1024) + headroom + safety_margin > clean_idle_free
    ):
        raise ValueError(
            "qualification evidence is not self-consistent with its clean idle free memory"
        )
    growth_bound = row_sentinel.get("max_growth_bytes")
    observed_growth = row_sentinel.get("observed_max_growth_bytes")
    observed_repeat_growth = row_sentinel.get(
        "observed_max_repeat_growth_bytes"
    )
    growth_safety = row_sentinel.get("safety_margin_bytes")
    if (
        row_sentinel.get("status") != "FROZEN"
        or isinstance(growth_bound, bool)
        or not isinstance(growth_bound, int)
        or growth_bound <= 0
        or isinstance(observed_growth, bool)
        or not isinstance(observed_growth, int)
        or observed_growth < 0
        or isinstance(observed_repeat_growth, bool)
        or not isinstance(observed_repeat_growth, int)
        or observed_repeat_growth < 0
        or observed_repeat_growth > ROW_SENTINEL_REPEAT_GROWTH_GUARD_BYTES
        or isinstance(growth_safety, bool)
        or not isinstance(growth_safety, int)
        or growth_safety <= 0
        or growth_bound < observed_growth + growth_safety
    ):
        raise ValueError("row-sentinel growth bound is not frozen")
    return headroom + safety_margin


def hardware_inventory() -> dict[str, Any]:
    """Enforce the capability envelope and record the device as provenance."""
    if not torch.cuda.is_available():
        raise ValueError("Judge requires a visible CUDA device")
    count = int(torch.cuda.device_count())
    if count != 1:
        raise ValueError(f"Judge must see exactly one CUDA device, found {count}")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    properties = torch.cuda.get_device_properties(0)
    name = str(torch.cuda.get_device_name(0))
    total_memory_bytes = int(properties.total_memory)
    if capability < MIN_COMPUTE_CAPABILITY:
        raise ValueError(
            f"GPU compute capability {capability} is below the capability envelope "
            f"minimum {MIN_COMPUTE_CAPABILITY}: {name}"
        )
    if total_memory_bytes < MIN_DEVICE_MEMORY_BYTES:
        raise ValueError(
            f"GPU memory {total_memory_bytes} bytes is below the capability envelope "
            f"minimum {MIN_DEVICE_MEMORY_BYTES} bytes: {name}"
        )
    return {
        "name": name,
        "count": count,
        "capability": list(capability),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "driver_version": nvml_driver_version(),
        "cuda_runtime_version": str(torch.version.cuda or ""),
        "total_memory_bytes": total_memory_bytes,
        "min_compute_capability": list(MIN_COMPUTE_CAPABILITY),
        "min_device_memory_bytes": MIN_DEVICE_MEMORY_BYTES,
    }


def device_peak_cap_gib(inventory: dict[str, Any], reserve_mib: int) -> float:
    """Derive the enforced peak cap from this device's own capacity."""
    capacity_mib = int(inventory["total_memory_bytes"]) // (1024 * 1024)
    usable_mib = capacity_mib - int(reserve_mib)
    if usable_mib <= 0:
        raise ValueError("device capacity does not cover the policy memory reserve")
    return usable_mib / 1024


def measurement_profile(inventory: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    """Return the full recorded measurement profile (provenance, never a gate)."""
    return {
        **device_profile(inventory),
        "work_image_id": str(provenance.get("work_image", {}).get("id", "")),
        "judge_image_id": str(provenance.get("judge_image", {}).get("id", "")),
    }


def device_profile(inventory: dict[str, Any]) -> dict[str, str]:
    """Return the recorded device half of a measurement profile."""
    return {
        "gpu_name": str(inventory["name"]),
        "compute_capability": str(inventory["compute_capability"]),
        "driver_version": str(inventory["driver_version"]),
    }


def reference_profile(profile: dict[str, Any]) -> dict[str, str]:
    """Return the hardware fields a quality reference is selected and gated on."""
    return {field: str(profile[field]) for field in REFERENCE_PROFILE_FIELDS}


def software_abi() -> dict[str, str]:
    """Return the numerics-relevant software ABI of this Judge image."""
    return {
        "cuda_runtime": str(torch.version.cuda or ""),
        "torch": str(torch.__version__),
        "transformers": str(getattr(transformers, "__version__", "")),
    }


def admissible_core_hash(field: str, recorded: object, current: object) -> bool:
    """Accept a recorded core hash if current or a declared scoring-equivalent predecessor."""
    if recorded == current:
        return True
    return (
        field == "verify_decoder_sha256"
        and recorded in SCORING_EQUIVALENT_VERIFY_DECODER_SHA256
    )


def measurement_profile_key(profile: dict[str, Any]) -> str:
    """Hash one measurement profile into a stable cache/binding key."""
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def valid_calibration_provenance(
    provenance: object,
    workload_policy: dict[str, Any] | None = None,
    runtime_provenance: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(provenance, dict):
        return False
    if (
        provenance.get("schema_version") != 1
        or provenance.get("task_id") != TASK_ID
        or provenance.get("workload_profile") != WORKLOAD_PROFILE
        or not isinstance(provenance.get("hardware"), str)
        or not provenance.get("hardware")
        or provenance.get("model_revision")
        != "6059b30c9531767ce47e0ca6a8161a6bb0c67693"
        or provenance.get("canonical_candidate_archive_sha256")
        != provenance.get("candidate_archive_sha256")
    ):
        return False
    for key in CALIBRATION_PROVENANCE_HASHES:
        value = provenance.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)
        ):
            return False
    for role in ("work_image", "judge_image"):
        image = provenance.get(role)
        if not isinstance(image, dict):
            return False
    sforge = provenance.get("sforge_runtime")
    files = sforge.get("files") if isinstance(sforge, dict) else None
    core = provenance.get("judge_invariant_core")
    if (
        not isinstance(sforge, dict)
        or sforge.get("scope") != "all_sforge_python"
        or sforge.get("sha256") != provenance.get("sforge_runtime_sha256")
        or sforge.get("file_set_sha256")
        != provenance.get("sforge_runtime_file_set_sha256")
        or not isinstance(files, list)
        or not files
        or files != sorted(set(files))
        or any(
            not isinstance(item, str)
            or re.fullmatch(
                r"sforge/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py", item
            )
            is None
            for item in files
        )
        or sforge.get("file_count") != len(files)
        or sforge.get("file_set_sha256")
        != hashlib.sha256(("\n".join(files) + "\n").encode()).hexdigest()
        or not isinstance(core, dict)
        or set(core) != set(JUDGE_INVARIANT_CORE)
        or any(
            core.get(field) != provenance.get(field)
            or not admissible_core_hash(field, core.get(field), sha256(path))
            for field, path in JUDGE_INVARIANT_CORE.items()
        )
    ):
        return False
    if runtime_provenance is not None:
        # Bind the calibration to the Judge invariant core that scores this
        # run.  The host-side SForge harness tree (``sforge/``) is recorded in
        # both snapshots as provenance but never executes inside the Judge, so
        # its drift is a recalibration signal (scripts/validate_layout.py
        # warns), not an integrity failure: gating on it silently zeroed every
        # submission after an unrelated launcher change.
        current_sforge = runtime_provenance.get("sforge_runtime")
        current_core = runtime_provenance.get("judge_invariant_core")
        if (
            not isinstance(current_sforge, dict)
            or not isinstance(current_core, dict)
            or set(current_core) != set(core)
            or any(
                not admissible_core_hash(field, core[field], current_core[field])
                for field in core
            )
        ):
            return False
        image_id = image.get("id")
        if (
            not isinstance(image_id, str)
            or not image_id.startswith("sha256:")
            or len(image_id) != 71
            or any(ch not in "0123456789abcdef" for ch in image_id[7:])
        ):
            return False
    policy = workload_policy or object_at(WORKLOAD_POLICY_PATH, "workload policy")
    if policy.get("release_status") == "FROZEN":
        expected_workload_parent = policy.get("provisional_parent_sha256")
        quality_policy = policy.get("quality_policy")
        if (
            not isinstance(expected_workload_parent, str)
            or len(expected_workload_parent) != 64
            or any(ch not in HEX for ch in expected_workload_parent)
            or not isinstance(quality_policy, dict)
            or quality_policy.get("provisional_parent_sha256")
            != expected_workload_parent
        ):
            return False
        expected_quality_controls = quality_policy.get("quality_controls_sha256")
    elif policy.get("release_status") == "PROVISIONAL":
        expected_workload_parent = sha256(WORKLOAD_POLICY_PATH)
        expected_quality_controls = sha256(QUALITY_CONTROLS_PATH)
    else:
        return False
    run_id = provenance.get("run_id")
    return (
        isinstance(run_id, str)
        and bool(run_id)
        and provenance.get("submission_id") == run_id
        and provenance.get("workload_policy_sha256") == expected_workload_parent
        and provenance.get("memory_policy_sha256") == sha256(MEMORY_POLICY_PATH)
        and provenance.get("quality_controls_sha256") == expected_quality_controls
    )


def read_evaluation_context() -> tuple[dict[str, Any], str]:
    if EVALUATION_CONTEXT_PATH.is_symlink() or not EVALUATION_CONTEXT_PATH.is_file():
        raise ValueError("private evaluation context is missing or invalid")
    stat = EVALUATION_CONTEXT_PATH.stat()
    if stat.st_uid != 0 or stat.st_mode & 0o077:
        raise ValueError("private evaluation context ownership is invalid")
    raw_payload = EVALUATION_CONTEXT_PATH.read_text(encoding="utf-8")
    EVALUATION_CONTEXT_PATH.unlink()
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("private evaluation context schema is invalid")
    private = payload.get("private")
    if not isinstance(private, dict):
        raise ValueError("private evaluation context payload is invalid")
    secret = private.get("COHORT_SECRET", "")
    if not isinstance(secret, str):
        raise ValueError("private cohort secret has invalid type")
    if secret and (
        len(secret) != 64 or any(ch not in "0123456789abcdef" for ch in secret)
    ):
        raise ValueError("private cohort secret has invalid contents")
    return payload, secret


def flag_from_private(private: dict[str, Any], name: str) -> bool:
    value = private.get(name, "")
    if value not in ("", "1"):
        raise ValueError(f"private {name} flag is invalid")
    return value == "1"


def runtime_provenance_from_private(private: dict[str, Any]) -> dict[str, Any]:
    supplied: dict[str, Any] = {}
    for field, context_key in RUNTIME_CONTEXT_FIELDS.items():
        raw = private.get(context_key)
        if field == "file_count":
            if not isinstance(raw, str) or re.fullmatch(r"[1-9][0-9]*", raw) is None:
                raise ValueError("private SForge runtime file count is invalid")
            supplied[field] = int(raw)
        else:
            if not isinstance(raw, str) or re.fullmatch(r"[0-9a-f]{64}", raw) is None:
                raise ValueError(f"private runtime provenance {field} is invalid")
            supplied[field] = raw
    core: dict[str, str] = {}
    for field, path in JUDGE_INVARIANT_CORE.items():
        context_key = "JUDGE_" + field.removesuffix("_sha256").upper() + "_SHA256"
        raw = private.get(context_key)
        if (
            not isinstance(raw, str)
            or re.fullmatch(r"[0-9a-f]{64}", raw) is None
            or raw != sha256(path)
        ):
            raise ValueError(f"private Judge invariant differs: {field}")
        core[field] = raw
    return {
        "schema_version": 1,
        "runtime_provenance_sha256": supplied["runtime_provenance_sha256"],
        "sforge_runtime": {
            "scope": "all_sforge_python",
            "sha256": supplied["sha256"],
            "file_set_sha256": supplied["file_set_sha256"],
            "file_count": supplied["file_count"],
        },
        "judge_invariant_core": core,
    }


def memory_fixture_from_private(
    private: dict[str, Any], *, qualification: bool
) -> dict[str, Any] | None:
    present = {
        key: private.get(key)
        for key in MEMORY_FIXTURE_CONTEXT_FIELDS
        if private.get(key) not in (None, "")
    }
    if not present:
        return None
    if not qualification or set(present) != set(MEMORY_FIXTURE_CONTEXT_FIELDS):
        raise ValueError("memory fixture context is present outside its qualification mode")
    kind = present["MEMORY_FIXTURE_KIND"]
    approval = present["MEMORY_FIXTURE_APPROVAL_ID"]
    raw_size = present["MEMORY_FIXTURE_ARCHIVE_SIZE_BYTES"]
    if (
        kind not in {"corrected_reference", "near_cap_stress"}
        or not isinstance(approval, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", approval) is None
        or not isinstance(raw_size, str)
        or re.fullmatch(r"[1-9][0-9]*", raw_size) is None
        or any(
            not isinstance(present[key], str)
            or re.fullmatch(r"[0-9a-f]{64}", present[key]) is None
            for key in (
                "MEMORY_FIXTURE_MANIFEST_SHA256",
                "MEMORY_FIXTURE_ARCHIVE_SHA256",
            )
        )
        or present["MEMORY_FIXTURE_ARCHIVE_SHA256"]
        != private.get("ARCHIVE_SHA256")
    ):
        raise ValueError("memory fixture context identity is invalid")
    return {
        "schema_version": 1,
        "manifest_sha256": present["MEMORY_FIXTURE_MANIFEST_SHA256"],
        "approval_id": approval,
        "kind": kind,
        "archive_sha256": present["MEMORY_FIXTURE_ARCHIVE_SHA256"],
        "archive_size_bytes": int(raw_size),
    }


def final_evaluation_identity(
    metadata: dict[str, Any], private: dict[str, Any]
) -> EvaluationIdentity:
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("final evaluation lacks a trusted run ID")
    required = {
        "archive_sha256": "ARCHIVE_SHA256",
        "work_image_id": "WORK_IMAGE_ID",
        "judge_image_id": "JUDGE_IMAGE_ID",
        "model_validation_sha256": "MODEL_VALIDATION_SHA256",
        "runtime_entry_sha256": "RUNTIME_ENTRY_SHA256",
        "source_digest": "SOURCE_DIGEST",
        "quality_controls_sha256": "QUALITY_CONTROLS_SHA256",
        "runtime_provenance_sha256": "RUNTIME_PROVENANCE_SHA256",
        "sforge_runtime_sha256": "SFORGE_RUNTIME_SHA256",
        "sforge_runtime_file_set_sha256": "SFORGE_RUNTIME_FILE_SET_SHA256",
        "verify_decoder_sha256": "JUDGE_VERIFY_DECODER_SHA256",
        "candidate_worker_sha256": "JUDGE_CANDIDATE_WORKER_SHA256",
        "cohort_sha256": "JUDGE_COHORT_SHA256",
        "quality_sha256": "JUDGE_QUALITY_SHA256",
        "source_admission_sha256": "JUDGE_SOURCE_ADMISSION_SHA256",
    }
    supplied = {}
    for field, key in required.items():
        value = private.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"final evaluation lacks private identity field {key}")
        supplied[field] = value
    row_order_sha256 = hashlib.sha256(
        ("\n".join(ROW_PLAN) + "\n").encode("utf-8")
    ).hexdigest()
    return EvaluationIdentity.from_mapping(
        {
            "run_id": run_id,
            **supplied,
            "workload_policy_sha256": sha256(WORKLOAD_POLICY_PATH),
            "memory_policy_sha256": sha256(MEMORY_POLICY_PATH),
            "hidden_manifest_sha256": sha256(HIDDEN_PATH),
            "row_order_sha256": row_order_sha256,
        }
    )


def write_results(log_dir: Path, detail: dict[str, Any], reward: float) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(detail)
    payload["reward"] = float(reward)
    (log_dir / "decoder_private.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    numeric = {
        "reward": float(reward),
        "integrity_ok": float(bool(payload.get("integrity_ok"))),
        "source_ok": float(bool(payload.get("source_ok"))),
        "quality_ok": float(bool(payload.get("quality_ok"))),
        "performance_ok": float(bool(payload.get("performance_ok"))),
        "final_version": float(payload.get("final_version", 0)),
    }
    if isinstance(payload.get("geomean_throughput_tok_s"), (int, float)):
        numeric["geomean_throughput_tok_s"] = float(payload["geomean_throughput_tok_s"])
    (log_dir / "reward.json").write_text(
        json.dumps(numeric, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def fail(log_dir: Path, message: str, **extra: Any) -> int:
    detail = {
        "schema_version": 3,
        "source_ok": bool(extra.pop("source_ok", False)),
        "integrity_ok": False,
        "quality_ok": False,
        "performance_ok": False,
        "error": message,
        **extra,
    }
    write_results(log_dir, detail, 0.0)
    return 1


def verify_protected(lab_root: Path) -> None:
    mapping = {
        "/app/workdir/decoder_lab/runtime/api.py": lab_root / "runtime/api.py",
        "/app/workdir/decoder_lab/runtime/configuration_llada.py": lab_root / "runtime/configuration_llada.py",
        "/app/workdir/decoder_lab/runtime/loader.py": lab_root / "runtime/loader.py",
        "/app/workdir/decoder_lab/runtime/modeling_llada.py": lab_root / "runtime/modeling_llada.py",
        "/app/workdir/decoder_lab/runtime/source_policy.py": lab_root / "runtime/source_policy.py",
        "/workspace/public_tests/public_quality.py": Path("/workspace/public_tests/public_quality.py"),
        "/workspace/public_tests/public_cases.json": PUBLIC_CASES_PATH,
        "/workspace/public_tests/memory_contract.json": Path(
            "/workspace/public_tests/memory_contract.json"
        ),
    }
    seen = set()
    for raw in PROTECTED_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        expected, runtime_path = line.split(None, 1)
        if runtime_path not in mapping:
            raise ValueError(f"unknown protected path: {runtime_path}")
        if sha256(mapping[runtime_path]) != expected:
            raise ValueError(f"protected file mismatch: {runtime_path}")
        seen.add(runtime_path)
    if seen != set(mapping):
        raise ValueError("protected manifest is incomplete")


class WorkerError(RuntimeError):
    pass


class CandidateWorker:
    def __init__(self, lab_root: Path, model_path: Path, expected_version: int) -> None:
        # This trusted controller binds NVML before the fork/exec that imports
        # any candidate-controlled Python.
        self._nvml_used_probe = bind_single_visible_nvml_used_probe()
        to_worker_read, to_worker_write = os.pipe()
        from_worker_read, from_worker_write = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(to_worker_write)
                os.close(from_worker_read)
                os.setsid()
                os.dup2(to_worker_read, 0)
                os.dup2(from_worker_write, 1)
                sink = os.open(os.devnull, os.O_WRONLY)
                os.dup2(sink, 2)
                os.close(sink)
                if to_worker_read > 2:
                    os.close(to_worker_read)
                if from_worker_write > 2:
                    os.close(from_worker_write)
                clean_environment = {
                    key: value
                    for key, value in os.environ.items()
                    if key
                    in {
                        "CUDA_VISIBLE_DEVICES",
                        "HF_DATASETS_OFFLINE",
                        "HF_HUB_DISABLE_TELEMETRY",
                        "HF_HUB_OFFLINE",
                        "LD_LIBRARY_PATH",
                        "PATH",
                        "TOKENIZERS_PARALLELISM",
                        "TRANSFORMERS_OFFLINE",
                        "TRITON_CACHE_DIR",
                        "TRITON_LIBCUDA_PATH",
                    }
                }
                clean_environment.update(
                    {"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}
                )
                os.execve(
                    sys.executable,
                    [
                        sys.executable,
                        str(WORKER_PATH),
                        "--lab-root",
                        str(lab_root),
                        "--model-path",
                        str(model_path),
                    ],
                    clean_environment,
                )
            except BaseException:
                os._exit(1)
        os.close(to_worker_read)
        os.close(from_worker_write)
        self.pid = pid
        self.stdin = os.fdopen(to_worker_write, "wb", buffering=0)
        self.stdout = os.fdopen(from_worker_read, "rb", buffering=0)
        self._wait_status: int | None = None
        os.set_blocking(self.stdout.fileno(), False)
        self._receive_buffer = bytearray()
        ready = self._receive(WORKER_START_TIMEOUT)
        if ready.get("event") != "ready" or int(ready.get("version", -1)) != expected_version:
            raise WorkerError("candidate worker ready/version mismatch")
        layers = ready.get("layers")
        fingerprint = ready.get("weight_fingerprint")
        if not isinstance(layers, int) or layers <= 0 or not isinstance(fingerprint, str):
            raise WorkerError("candidate worker returned invalid ready attestation")
        self.layers = layers
        self.weight_fingerprint = fingerprint

    def _receive(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while b"\n" not in self._receive_buffer:
            if len(self._receive_buffer) > MAX_PROTOCOL_LINE:
                raise WorkerError("candidate worker protocol line is oversized")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerError("candidate worker protocol timed out")
            selector = selectors.DefaultSelector()
            try:
                selector.register(self.stdout, selectors.EVENT_READ)
                if not selector.select(remaining):
                    raise WorkerError("candidate worker protocol timed out")
            finally:
                selector.close()
            chunk = os.read(self.stdout.fileno(), 65536)
            if not chunk:
                raise WorkerError(
                    f"candidate worker exited before a response (status={self._poll()})"
                )
            self._receive_buffer.extend(chunk)
        raw_line, remainder = self._receive_buffer.split(b"\n", 1)
        self._receive_buffer = bytearray(remainder)
        if len(raw_line) > MAX_PROTOCOL_LINE:
            raise WorkerError("candidate worker protocol line is oversized")
        try:
            response = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkerError("candidate worker returned invalid JSON") from exc
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise WorkerError(str(response.get("error", "candidate worker failed"))[:500])
        return response

    def request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        if self._poll() is not None:
            raise WorkerError("candidate worker is unavailable")
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_PROTOCOL_LINE:
            raise WorkerError("controller request exceeds protocol cap")
        try:
            self.stdin.write((encoded + "\n").encode("utf-8"))
            self.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerError("candidate worker protocol pipe closed") from exc
        return self._receive(timeout)

    def prepare(self, prompt_ids: list[int], gen_length: int, block_length: int) -> None:
        response = self.request(
            {
                "op": "prepare",
                "prompt_ids": prompt_ids,
                "gen_length": gen_length,
                "block_length": block_length,
                "mask_id": MASK_TOKEN_ID,
            },
            60.0,
        )
        if response.get("event") != "prepared" or response.get("prompt_length") != len(prompt_ids):
            raise WorkerError("candidate worker prepare attestation mismatch")

    def run(self, timeout: float) -> dict[str, Any]:
        sampler = PeakMemorySampler(self._nvml_used_probe)
        sampler.start()
        try:
            response = self.request({"op": "run"}, timeout)
        except Exception:
            sampler.stop()
            raise
        response["nvml_peak_device_used_bytes"] = sampler.stop()
        return response

    def integrity(self) -> str:
        response = self.request({"op": "integrity"}, 60.0)
        value = response.get("weight_fingerprint")
        if response.get("event") != "integrity" or not isinstance(value, str):
            raise WorkerError("candidate worker integrity response is invalid")
        return value

    def close(self) -> None:
        if self._poll() is None:
            try:
                self.request({"op": "close"}, 10.0)
            except Exception:
                pass
        try:
            self.stdin.close()
            self.stdout.close()
        except OSError:
            pass
        deadline = time.monotonic() + 10.0
        while self._poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._poll() is None:
            try:
                os.killpg(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5.0
            while self._poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
        if self._poll() is None:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._poll(block=True)

    def _poll(self, *, block: bool = False) -> int | None:
        if self._wait_status is not None:
            return self._wait_status
        options = 0 if block else os.WNOHANG
        try:
            pid, status = os.waitpid(self.pid, options)
        except ChildProcessError:
            self._wait_status = -1
            return self._wait_status
        if pid == 0:
            return None
        self._wait_status = status
        return status


def encode_prompt(tokenizer: Any, request_text: str) -> list[int]:
    chat = [{"role": "user", "content": request_text}]
    rendered = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if not isinstance(encoded, list) or not encoded or any(
        isinstance(token, bool) or not isinstance(token, int) for token in encoded
    ):
        raise ValueError("trusted tokenizer returned invalid prompt ids")
    return encoded


def decode_completion(
    tokenizer: Any,
    generated_ids: list[int],
    stop: list[str],
    *,
    humaneval: bool,
) -> tuple[str, int]:
    if humaneval:
        return (
            tokenizer.decode(generated_ids, skip_special_tokens=True),
            sum(token != EOS_TOKEN_ID for token in generated_ids),
        )
    generated = list(generated_ids)
    if EOS_TOKEN_ID in generated:
        generated = generated[: generated.index(EOS_TOKEN_ID)]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    for marker in stop:
        if marker in text:
            text = text.split(marker, 1)[0]
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    clean = tokenizer.decode(ids, skip_special_tokens=True)
    return clean, len(tokenizer(clean, add_special_tokens=False)["input_ids"])


def validate_worker_result(
    response: dict[str, Any],
    gen_length: int,
    layers: int,
    max_peak_gib: float | None,
) -> tuple[list[int], int, dict[str, int | float]]:
    generated = response.get("generated_ids")
    nfe = response.get("nfe")
    layer_calls = response.get("layer_calls")
    if not isinstance(generated, list) or len(generated) != gen_length:
        raise ValueError("worker result has the wrong generated length")
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        or token >= VOCAB_SIZE
        or token == MASK_TOKEN_ID
        for token in generated
    ):
        raise ValueError("worker result contains invalid generated ids")
    if not isinstance(nfe, int) or nfe <= 0:
        raise ValueError("worker result has an invalid forward count")
    if (
        not isinstance(layer_calls, list)
        or len(layer_calls) != layers
        or any(not isinstance(value, int) or value != nfe for value in layer_calls)
    ):
        raise ValueError("worker result failed layer coverage")
    memory_fields = (
        "torch_peak_allocated_bytes",
        "torch_peak_reserved_bytes",
        "nvml_peak_device_used_bytes",
    )
    values = [response.get(field) for field in memory_fields]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("worker result has invalid allocated/reserved/NVML memory")
    allocated, reserved, device_used = (int(value) for value in values)
    if not allocated <= reserved <= device_used:
        raise ValueError("worker memory measurements have invalid ordering")
    allocated_gib = allocated / 2**30
    if max_peak_gib is not None and allocated_gib >= max_peak_gib:
        raise ValueError("worker result exceeded the device-relative memory cap")
    return generated, nfe, {
        "torch_peak_allocated_bytes": allocated,
        "torch_peak_reserved_bytes": reserved,
        "nvml_peak_device_used_bytes": device_used,
        "peak_allocated_gib": allocated_gib,
        "peak_reserved_gib": reserved / 2**30,
        "nvml_device_used_gib": device_used / 2**30,
    }


def run_one(
    worker: CandidateWorker,
    tokenizer: Any,
    case: dict[str, Any],
    *,
    timeout: float,
    max_peak_gib: float | None,
) -> dict[str, Any]:
    from quality import score_completion

    prompt_ids = encode_prompt(tokenizer, case["request_text"])
    frozen_prompt_count = case.get("prompt_token_count")
    if (
        isinstance(frozen_prompt_count, bool)
        or not isinstance(frozen_prompt_count, int)
        or frozen_prompt_count != len(prompt_ids)
    ):
        raise ValueError("frozen prompt-token attestation differs from pinned tokenizer")
    if frozen_prompt_count + int(case["gen_length"]) > 4096:
        raise ValueError("frozen request exceeds the pinned model context")
    worker.prepare(prompt_ids, int(case["gen_length"]), int(case["block_length"]))
    started = time.perf_counter()
    response = worker.run(timeout)
    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("invalid controller-measured generation time")
    generated, nfe, memory = validate_worker_result(
        response,
        int(case["gen_length"]),
        worker.layers,
        max_peak_gib,
    )
    completion, useful = decode_completion(
        tokenizer,
        generated,
        case["stop"],
        humaneval=case["benchmark"] == "humaneval",
    )
    quality = score_completion(case, completion)
    return {
        "case_sha256": hashlib.sha256(case["case_id"].encode("utf-8")).hexdigest(),
        "row_id": case["row_id"],
        "benchmark": case["benchmark"],
        "gen_length": case["gen_length"],
        "seconds": elapsed,
        "useful_tokens": useful,
        "nfe": nfe,
        "slots_per_nfe": case["gen_length"] / nfe,
        "per_nfe_ms": elapsed * 1000 / nfe,
        **memory,
        "correct": bool(quality["correct"]),
        "parsed": bool(quality["parsed"]),
        "quality_detail": quality,
        "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
    }


def admission_state_test(
    worker: CandidateWorker,
    tokenizer: Any,
    public_cases: list[dict[str, Any]],
    max_peak_gib: float | None,
    *,
    enforce_memory_stability: bool = True,
) -> dict[str, Any]:
    selected = [
        next(item for item in public_cases if item["benchmark"] == benchmark)
        for benchmark in ("gsm8k", "math")
    ]
    outputs = []
    memories = []
    # The first two calls warm cross-request state. The final two execute the
    # exact same shape/prompt back-to-back, which is the leak sentinel.
    for case in (selected[0], selected[1], selected[0], selected[0]):
        prompt_ids = encode_prompt(tokenizer, case["request_text"])
        worker.prepare(prompt_ids, 32, 32)
        response = worker.run(60.0)
        generated, _nfe, memory = validate_worker_result(
            response, 32, worker.layers, max_peak_gib
        )
        outputs.append(generated)
        memories.append(memory)
    if outputs[0] != outputs[2] or outputs[2] != outputs[3]:
        raise ValueError("repeat/cross-request cache-isolation test failed")
    stable, deltas = fixed_shape_memory_stable(memories[2], memories[3])
    if not stable and enforce_memory_stability:
        raise ValueError("fixed-shape repeat memory sentinel grew materially")
    return {
        "case": "gsm8k-public-fixed-shape-32",
        "first": memories[2],
        "repeat": memories[3],
        "deltas_bytes": deltas,
        "stable": stable,
    }


def fixed_shape_row_probe(
    worker: CandidateWorker,
    tokenizer: Any,
    public_case: dict[str, Any],
    max_peak_gib: float | None,
    *,
    timeout: float,
) -> dict[str, Any]:
    prompt_ids = encode_prompt(tokenizer, public_case["request_text"])
    worker.prepare(prompt_ids, 32, 32)
    response = worker.run(timeout)
    generated, _nfe, memory = validate_worker_result(
        response, 32, worker.layers, max_peak_gib
    )
    return {
        "memory": memory,
        "output_sha256": hashlib.sha256(
            json.dumps(generated, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def row_memory_sentinel_evidence(
    start: dict[str, Any],
    end: dict[str, Any],
    repeat: dict[str, Any],
    *,
    max_material_growth_bytes: int,
    max_repeat_growth_bytes: int,
    enforce_stability: bool = True,
) -> dict[str, Any]:
    start_end_stable, deltas = fixed_shape_memory_stable(
        start["memory"],
        end["memory"],
        material_growth_bytes=max_material_growth_bytes,
    )
    repeat_stable, repeat_deltas = fixed_shape_memory_stable(
        end["memory"],
        repeat["memory"],
        material_growth_bytes=max_repeat_growth_bytes,
    )
    output_stable = (
        start["output_sha256"]
        == end["output_sha256"]
        == repeat["output_sha256"]
    )
    observed = max(0, *(int(value) for value in deltas.values()))
    observed_repeat = max(
        0, *(int(value) for value in repeat_deltas.values())
    )
    evidence = {
        "schema_version": 2,
        "case": "gsm8k-public-fixed-shape-32",
        "start": {
            field: int(start["memory"][field])
            for field in (
                "torch_peak_allocated_bytes",
                "torch_peak_reserved_bytes",
                "nvml_peak_device_used_bytes",
            )
        },
        "end": {
            field: int(end["memory"][field])
            for field in (
                "torch_peak_allocated_bytes",
                "torch_peak_reserved_bytes",
                "nvml_peak_device_used_bytes",
            )
        },
        "repeat": {
            field: int(repeat["memory"][field])
            for field in (
                "torch_peak_allocated_bytes",
                "torch_peak_reserved_bytes",
                "nvml_peak_device_used_bytes",
            )
        },
        "deltas_bytes": deltas,
        "repeat_deltas_bytes": repeat_deltas,
        "observed_max_positive_growth_bytes": observed,
        "observed_max_positive_repeat_growth_bytes": observed_repeat,
        "max_material_growth_bytes": max_material_growth_bytes,
        "max_repeat_growth_bytes": max_repeat_growth_bytes,
        "repeat_stable": bool(repeat_stable),
        "stable": bool(start_end_stable and repeat_stable and output_stable),
        "start_output_sha256": start["output_sha256"],
        "end_output_sha256": end["output_sha256"],
        "repeat_output_sha256": repeat["output_sha256"],
    }
    if not evidence["stable"] and enforce_stability:
        raise CandidateFailure(
            "row-long fixed-shape memory/output sentinel changed materially"
        )
    return evidence


def aggregate_rows(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["row_id"]].append(record)
    rows = {}
    for row_id, members in sorted(groups.items()):
        total_time = sum(item["seconds"] for item in members)
        total_tokens = sum(item["useful_tokens"] for item in members)
        rows[row_id] = {
            "cases": len(members),
            "correct": sum(item["correct"] for item in members),
            "parsed": sum(item["parsed"] for item in members),
            "useful_tokens": total_tokens,
            "seconds": total_time,
            "throughput_tok_s": total_tokens / total_time if total_time > 0 else 0.0,
            "mean_nfe": statistics.fmean(item["nfe"] for item in members),
            "mean_slots_per_nfe": statistics.fmean(item["slots_per_nfe"] for item in members),
            "peak_allocated_gib": max(item["peak_allocated_gib"] for item in members),
            "peak_reserved_gib": max(item["peak_reserved_gib"] for item in members),
            "peak_nvml_device_used_gib": max(
                item["nvml_device_used_gib"] for item in members
            ),
        }
    return rows


def calibration_quality_diagnostic(rows: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    failures = []
    for row_id, row in rows.items():
        if row["parsed"] <= 0:
            failures.append(f"{row_id}:parsed")
        if row["useful_tokens"] <= 0:
            failures.append(f"{row_id}:useful_tokens")
    return not failures, failures


def paired_reference_rows(
    records: list[dict[str, Any]], reference: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    case_records = reference.get("case_records")
    if not isinstance(case_records, dict):
        raise ValueError("quality reference has no per-case calibration records")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for record in records:
        case_sha256 = record.get("case_sha256")
        row_id = record.get("row_id")
        if not isinstance(case_sha256, str) or case_sha256 in seen:
            raise ValueError("candidate records have invalid case provenance")
        seen.add(case_sha256)
        baseline = case_records.get(case_sha256)
        if not isinstance(baseline, dict) or baseline.get("row_id") != row_id:
            raise ValueError("quality reference does not match the selected cohort")
        if (
            not isinstance(baseline.get("correct"), bool)
            or not isinstance(baseline.get("parsed"), bool)
            or isinstance(baseline.get("useful_tokens"), bool)
            or not isinstance(baseline.get("useful_tokens"), int)
            or baseline["useful_tokens"] < 0
        ):
            raise ValueError("quality reference contains invalid case metrics")
        grouped[str(row_id)].append(baseline)
    paired = {}
    for row_id, members in sorted(grouped.items()):
        cases = len(members)
        correct = sum(int(item.get("correct") is True) for item in members)
        parsed = sum(int(item.get("parsed") is True) for item in members)
        useful_tokens = sum(int(item.get("useful_tokens", -1)) for item in members)
        if cases <= 0 or useful_tokens < 0:
            raise ValueError("quality reference contains invalid cohort metrics")
        paired[row_id] = {
            "cases": cases,
            "correct": correct,
            "parsed": parsed,
            "useful_tokens": useful_tokens,
        }
    return paired


def quality_diagnostic(
    records: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    reference: dict[str, Any],
    *,
    threshold_ppm: int,
) -> tuple[bool, list[str], dict[str, Any], dict[str, Any]]:
    ref_rows = paired_reference_rows(records, reference)
    if isinstance(threshold_ppm, bool) or not isinstance(threshold_ppm, int) or threshold_ppm <= 0:
        raise ValueError("quality retention threshold is invalid")
    measurement = measure_quality_retention(rows, ref_rows)
    failures = (
        []
        if int(measurement["score_ppm"]) >= threshold_ppm
        else ["producer_relative_retention_below_threshold"]
    )
    benchmark_ppm = benchmark_retention(rows, ref_rows)
    failures.extend(fixed_gate_failures(measurement, benchmark_ppm))
    gate = {
        "gate": QUALITY_GATE,
        "minimum_score_ppm": threshold_ppm,
        "aggregate_floor_ppm": AGGREGATE_FLOOR_PPM,
        "benchmark_floor_ppm": BENCHMARK_FLOOR_PPM,
        "max_useful_tokens_retention_ppm": MAX_USEFUL_TOKENS_RETENTION_PPM,
        "benchmark_components_ppm": benchmark_ppm,
    }
    return not failures, failures, measurement, gate


def memory_checkpoints(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Record cross-case early/middle/late peaks as diagnostics only."""
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_row[str(record["row_id"])].append(record)
    checkpoints: dict[str, Any] = {}
    for row_id, members in by_row.items():
        indices = (0, len(members) // 2, len(members) - 1)
        points = []
        for label, index in zip(("early", "middle", "late"), indices):
            record = members[index]
            points.append(
                {
                    "label": label,
                    "case_sha256": record["case_sha256"],
                    "torch_peak_allocated_bytes": record[
                        "torch_peak_allocated_bytes"
                    ],
                    "torch_peak_reserved_bytes": record["torch_peak_reserved_bytes"],
                    "nvml_peak_device_used_bytes": record[
                        "nvml_peak_device_used_bytes"
                    ],
                }
            )
        checkpoints[row_id] = points
    return checkpoints


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    calibration = False
    quality_control = False
    qualification = False
    bootstrap = False
    pilot = False
    worker: CandidateWorker | None = None
    try:
        lab_root = locate_lab(args.app_dir.resolve())
        admission = admit(lab_root)
        verify_protected(lab_root)
        quality_runtime_sha256 = sha256(JUDGE_ROOT / "verifier/quality.py")
        model_path = Path(os.environ["LLADA_MODEL_PATH"]).resolve()
        sys.path.insert(0, str(lab_root))
        # The child immediately execs a clean worker process; together this is
        # the confidentiality boundary. Do not read public cases, hidden cases,
        # gold tests, or the reference until that clean child is confirmed
        # ready: no controller Python modules or later parent memory can appear
        # in candidate-controlled code.
        worker = CandidateWorker(lab_root, model_path, int(admission["final_version"]))
        # Read the root-only run secret only after the gold-free child exists.
        # The child has no inherited file descriptor and its agent UID cannot
        # open the root-owned mode-0600 controller file.
        trusted_metadata, cohort_secret = read_evaluation_context()
        private_context = trusted_metadata["private"]
        calibration = flag_from_private(private_context, "CALIBRATION")
        quality_control = flag_from_private(private_context, "QUALITY_CONTROL")
        qualification = flag_from_private(private_context, "QUALIFICATION")
        bootstrap = flag_from_private(private_context, "BOOTSTRAP")
        pilot = flag_from_private(private_context, "PILOT")
        if sum((calibration, quality_control, qualification, bootstrap, pilot)) > 1:
            raise ValueError("evaluation producer/bootstrap modes are mutually exclusive")
        memory_fixture = memory_fixture_from_private(
            private_context, qualification=qualification
        )
        workload_policy = object_at(WORKLOAD_POLICY_PATH, "workload policy")
        memory_policy = object_at(MEMORY_POLICY_PATH, "memory policy")
        validate_workload_policy(
            workload_policy, bootstrap=bootstrap or qualification
        )
        # The policy contributes only the device-memory reserve; the enforced
        # peak cap is derived below from this device's own capacity.
        memory_reserve_mib = validate_memory_policy(
            memory_policy, bootstrap=bootstrap or qualification
        )
        row_sentinel_policy = memory_policy["row_sentinel"]
        row_sentinel_growth_bound = int(
            row_sentinel_policy["qualification_guard_bytes"]
            if bootstrap or qualification
            else row_sentinel_policy["max_growth_bytes"]
        )
        row_sentinel_repeat_growth_bound = int(
            row_sentinel_policy["repeat_growth_guard_bytes"]
        )
        if (calibration or quality_control) and workload_policy.get("release_status") != "PROVISIONAL":
            raise ValueError("non-release producer evaluation requires a provisional policy")
        if not bootstrap and not qualification and not calibration and not quality_control and not pilot and workload_policy.get("release_status") != "FROZEN":
            raise ValueError("ordinary evaluation requires a frozen release policy")
        if pilot and workload_policy.get("release_status") != "PROVISIONAL":
            raise ValueError("pilot evaluation requires a provisional policy")
        control_id = None
        expected_classification = None
        control_measurement_floor_ppm = None
        if quality_control:
            control_id = private_context.get("CONTROL_ID")
            expected_classification = private_context.get("EXPECTED_CLASSIFICATION")
            if (
                not isinstance(control_id, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", control_id) is None
                or expected_classification not in {"ACCEPT", "REJECT"}
            ):
                raise ValueError("quality-control identity/classification is invalid")
        quality_controls = object_at(QUALITY_CONTROLS_PATH, "quality-control manifest")
        supplied_quality_controls_sha256 = private_context.get(
            "QUALITY_CONTROLS_SHA256"
        )
        # A pilot is an agent-development run, not release evidence.  It must
        # not depend on the release-only control precommitment.  Producer,
        # control, and ordinary release evaluations retain the exact binding.
        if not pilot and (
            not isinstance(supplied_quality_controls_sha256, str)
            or supplied_quality_controls_sha256 != sha256(QUALITY_CONTROLS_PATH)
        ):
            raise ValueError("private quality-control manifest identity differs")
        if calibration or quality_control:
            methodology = quality_controls.get("methodology")
            if (
                quality_controls.get("status") != "PRECOMMITTED"
                or quality_controls.get("task_id") != TASK_ID
                or quality_controls.get("profile") != WORKLOAD_PROFILE
                or not isinstance(methodology, dict)
                or methodology.get("metric")
                != "producer_relative_min_retention_ppm_v1"
                or methodology.get("threshold_derivation")
                != "midpoint_min_positive_max_negative_v1"
                or isinstance(methodology.get("minimum_separation_ppm"), bool)
                or not isinstance(methodology.get("minimum_separation_ppm"), int)
                or methodology["minimum_separation_ppm"] <= 0
                or isinstance(methodology.get("control_measurement_floor_ppm"), bool)
                or not isinstance(methodology.get("control_measurement_floor_ppm"), int)
                or methodology["control_measurement_floor_ppm"] <= 0
            ):
                raise ValueError("producer/control run lacks a frozen precommitment")
            control_measurement_floor_ppm = int(
                methodology["control_measurement_floor_ppm"]
            )
        if quality_control:
            expected_entries = [
                item
                for group in (
                    quality_controls.get("positive_controls"),
                    quality_controls.get("negative_controls"),
                )
                if isinstance(group, list)
                for item in group
                if isinstance(item, dict) and item.get("id") == control_id
            ]
            if (
                len(expected_entries) != 1
                or expected_entries[0].get("expected_classification")
                != expected_classification
            ):
                raise ValueError("quality-control identity is not precommitted")
        evaluation_context = context_from_metadata(
            trusted_metadata,
            cohort_secret,
            calibration=calibration or quality_control or qualification,
        )
        hidden = json.loads(HIDDEN_PATH.read_text(encoding="utf-8"))
        selection = select_cohort(hidden, evaluation_context)
        if selection.mode != "final":
            raise ValueError("every scored round requires the complete frozen final workload")
        runtime_provenance = runtime_provenance_from_private(private_context)
        # The quality reference is comparable only on the same GPU model and
        # compute capability under the same software ABI, so it is cached per
        # reference profile and selected on-device.  Driver version and image
        # ids are recorded provenance, never a selection key.  The
        # single-profile install remains the fallback for a Judge image built
        # for exactly one device (and for the pending placeholder).
        reference_inventory = hardware_inventory()
        reference_device_profile_key = measurement_profile_key(
            device_profile(reference_inventory)
        )
        observed_reference_profile = reference_profile(
            device_profile(reference_inventory)
        )
        observed_reference_profile_key = measurement_profile_key(
            observed_reference_profile
        )
        observed_software_abi = software_abi()
        cached_reference = (
            REFERENCE_CACHE_DIR / f"{observed_reference_profile_key}.json"
        )
        reference_path = (
            cached_reference
            if cached_reference.is_file() and not cached_reference.is_symlink()
            else REFERENCE_PATH
        )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        hidden_sha256 = sha256(HIDDEN_PATH)
        reference_sha256 = sha256(reference_path)
        public_cases = json.loads(PUBLIC_CASES_PATH.read_text(encoding="utf-8"))
        # Pilot quality is computed directly from the selected hidden cases and
        # is diagnostic-only.  A calibrated reference and its producer chain
        # are release concerns and must never block agent feedback.
        require_calibrated_reference = (
            quality_control
            or (not calibration and not qualification and not pilot)
        ) and not bootstrap
        if require_calibrated_reference:
            if reference.get("schema_version") != 4:
                return fail(args.log_dir, "quality reference schema mismatch", source_ok=True)
            if (
                reference.get("task_id") != TASK_ID
                or reference.get("workload_profile") != WORKLOAD_PROFILE
                or reference.get("status") != "CALIBRATED_STRATIFIED520"
            ):
                return fail(args.log_dir, "quality reference identity mismatch", source_ok=True)
            if reference.get("isolation") != "root-controller/gold-free-agent-worker":
                return fail(args.log_dir, "quality reference isolation provenance mismatch", source_ok=True)
            source_report_sha256 = reference.get("source_private_report_sha256")
            if (
                not isinstance(source_report_sha256, str)
                or len(source_report_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in source_report_sha256)
            ):
                return fail(args.log_dir, "quality reference source provenance is invalid", source_ok=True)
            if not reference.get("calibrated"):
                return fail(args.log_dir, "quality reference is not calibrated", source_ok=True)
            calibration_provenance = reference.get("calibration_provenance")
            if not valid_calibration_provenance(
                calibration_provenance, workload_policy, runtime_provenance
            ):
                return fail(args.log_dir, "quality reference calibration provenance is invalid", source_ok=True)
            if reference.get("dataset_manifest_sha256") != hidden_sha256:
                return fail(args.log_dir, "quality reference dataset provenance mismatch", source_ok=True)
            if reference.get("quality_runtime_sha256") != quality_runtime_sha256:
                return fail(args.log_dir, "quality reference runtime provenance mismatch", source_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        inventory = reference_inventory
        gpu_name = inventory["name"]
        # Peak memory is gated against this device's own capacity, never
        # against a cap measured on the qualification GPU.
        max_peak_gib = (
            None
            if memory_reserve_mib is None
            else device_peak_cap_gib(inventory, memory_reserve_mib)
        )
        if require_calibrated_reference:
            observed_profile = measurement_profile(
                inventory, reference["calibration_provenance"]
            )
            observed_profile_key = measurement_profile_key(observed_profile)
            recorded_profile = reference.get("measurement_profile")
            try:
                recorded_reference_profile_key = measurement_profile_key(
                    reference_profile(recorded_profile)
                )
            except (KeyError, TypeError):
                recorded_reference_profile_key = None
            if recorded_reference_profile_key != observed_reference_profile_key:
                return fail(
                    args.log_dir,
                    "quality reference was produced under a different measurement "
                    f"profile (GPU model / compute capability): expected "
                    f"{observed_reference_profile_key}, reference carries "
                    f"{recorded_reference_profile_key!r}",
                    source_ok=True,
                )
            if observed_software_abi != REFERENCE_SOFTWARE_ABI:
                return fail(
                    args.log_dir,
                    "quality reference was produced under a different measurement "
                    f"profile (software ABI): expected {REFERENCE_SOFTWARE_ABI}, "
                    f"this Judge runs {observed_software_abi}",
                    source_ok=True,
                )
        memory_sentinel = admission_state_test(
            worker,
            tokenizer,
            public_cases,
            max_peak_gib,
            enforce_memory_stability=not pilot,
        )
        if worker.integrity() != worker.weight_fingerprint:
            raise ValueError("candidate mutated sampled model weights during build/admission")

        row_commitments: list[dict[str, Any]] = []
        if selection.mode == "final":
            worker.close()
            worker = None
            identity = final_evaluation_identity(trusted_metadata, private_context)
            final_policy = workload_policy["final"]
            case_timeout = (
                _positive_env("LLADA_QUALIFICATION_CASE_TIMEOUT")
                if qualification
                else _positive_env("LLADA_BOOTSTRAP_CASE_TIMEOUT")
                if bootstrap
                else _positive(
                    final_policy.get("case_timeout_seconds"), "final case timeout"
                )
            )
            row_timeout = (
                _positive_env("LLADA_QUALIFICATION_ROW_TIMEOUT")
                if qualification
                else _positive_env("LLADA_BOOTSTRAP_ROW_TIMEOUT")
                if bootstrap
                else _positive_env("LLADA_PILOT_FINAL_ROW_TIMEOUT")
                if pilot
                else _positive(final_policy.get("row_timeout_seconds"), "final row timeout")
            )
            outer_timeout = (
                _positive_env("LLADA_QUALIFICATION_OUTER_TIMEOUT")
                if qualification
                else _positive_env("LLADA_BOOTSTRAP_OUTER_TIMEOUT")
                if bootstrap
                else _positive_env("LLADA_PILOT_FINAL_OUTER_TIMEOUT")
                if pilot
                else _positive(final_policy.get("outer_timeout_seconds"), "final outer timeout")
            )
            if pilot and not (0 < case_timeout < row_timeout < outer_timeout <= 7_000):
                raise ValueError("pilot final deadlines are invalid or exceed 7,000 seconds")
            if (qualification or bootstrap) and not (
                0 < case_timeout < row_timeout < outer_timeout <= 7_200
            ):
                raise ValueError("non-release deadlines are invalid or exceed the task ceiling")
            if private_context.get("CHECKPOINT_DIR") != str(CHECKPOINT_ROOT):
                raise ValueError("trusted checkpoint directory context is missing")
            if CHECKPOINT_ROOT.is_symlink() or not CHECKPOINT_ROOT.is_dir():
                raise ValueError("trusted checkpoint directory is unsafe")
            checkpoint_stat = CHECKPOINT_ROOT.stat()
            if checkpoint_stat.st_uid != 0 or checkpoint_stat.st_mode & 0o077:
                raise ValueError("trusted checkpoint directory ownership is invalid")
            checkpoint_root = evaluation_checkpoint_root(CHECKPOINT_ROOT, trusted_metadata)
            persistent_deadline = load_or_create_absolute_deadline(
                checkpoint_root,
                identity=identity,
                outer_timeout_seconds=outer_timeout,
            )
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for case in selection.cases:
                buckets[str(case["row_id"])].append(dict(case))
            records = []
            for row_index, row_id in enumerate(ROW_PLAN):
                row_cases = buckets.get(row_id, [])
                if len(row_cases) != ROW_PLAN[row_id].canonical_count:
                    raise ValueError(f"final row {row_id} is incomplete")

                def run_attempt(
                    _attempt: int,
                ) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
                    row_worker: CandidateWorker | None = None
                    started = time.monotonic()
                    try:
                        row_worker = CandidateWorker(
                            lab_root, model_path, int(admission["final_version"])
                        )
                        if row_worker.integrity() != row_worker.weight_fingerprint:
                            raise CandidateFailure("row worker failed initial integrity")
                        sentinel_case = next(
                            item
                            for item in public_cases
                            if item["benchmark"] == "gsm8k"
                        )
                        start_sentinel = fixed_shape_row_probe(
                            row_worker,
                            tokenizer,
                            sentinel_case,
                            max_peak_gib,
                            timeout=min(60.0, persistent_deadline.remaining_seconds()),
                        )
                        order = list(row_cases)
                        row_seed = int.from_bytes(
                            hashlib.sha256(
                                f"stratified520-row-order-v1:{row_id}".encode()
                            ).digest()[:8],
                            "big",
                        )
                        random.Random(row_seed).shuffle(order)
                        output = []
                        row_deadline = started + row_timeout
                        for case in order:
                            if time.monotonic() >= row_deadline:
                                raise CandidateFailure("candidate exceeded the row deadline")
                            if persistent_deadline.remaining_seconds() <= 0:
                                raise CandidateFailure("candidate exceeded the outer deadline")
                            output.append(
                                run_one(
                                    row_worker,
                                    tokenizer,
                                    case,
                                    timeout=min(
                                        case_timeout,
                                        row_deadline - time.monotonic(),
                                        persistent_deadline.remaining_seconds(),
                                    ),
                                    max_peak_gib=max_peak_gib,
                                )
                            )
                        if row_worker.integrity() != row_worker.weight_fingerprint:
                            raise CandidateFailure("candidate mutated weights within a row")
                        end_sentinel = fixed_shape_row_probe(
                            row_worker,
                            tokenizer,
                            sentinel_case,
                            max_peak_gib,
                            timeout=min(
                                60.0,
                                row_deadline - time.monotonic(),
                                persistent_deadline.remaining_seconds(),
                            ),
                        )
                        repeat_sentinel = fixed_shape_row_probe(
                            row_worker,
                            tokenizer,
                            sentinel_case,
                            max_peak_gib,
                            timeout=min(
                                60.0,
                                row_deadline - time.monotonic(),
                                persistent_deadline.remaining_seconds(),
                            ),
                        )
                        row_sentinel = row_memory_sentinel_evidence(
                            start_sentinel,
                            end_sentinel,
                            repeat_sentinel,
                            max_material_growth_bytes=row_sentinel_growth_bound,
                            max_repeat_growth_bytes=row_sentinel_repeat_growth_bound,
                            enforce_stability=not pilot,
                        )
                        return output, time.monotonic() - started, row_sentinel
                    except NvmlSamplerError as exc:
                        raise InfrastructureFailure(
                            "trusted NVML sampler failed during row evaluation"
                        ) from exc
                    finally:
                        if row_worker is not None:
                            row_worker.close()

                row_records, commitment = run_row_with_retry(
                    artifact_root=checkpoint_root,
                    identity=identity,
                    row_id=row_id,
                    row_index=row_index,
                    run_attempt=run_attempt,
                    deadline=persistent_deadline,
                    max_infrastructure_retries=int(
                        final_policy["max_infrastructure_retries_per_row"]
                    ),
                )
                records.extend(row_records)
                row_commitments.append(commitment)
            validate_completed_rows(
                checkpoint_root,
                identity=identity,
                row_order=list(ROW_PLAN),
                deadline_commitment_sha256=persistent_deadline.commitment_sha256,
            )
        else:
            raise ValueError("complete frozen final row lifecycle is required")
        rows = aggregate_rows(records)
        checkpoints = memory_checkpoints(records)
        memory_leak_free = bool(memory_sentinel["stable"]) and all(
            commitment.get("row_memory_sentinel", {}).get("stable") is True
            for commitment in row_commitments
        )
        throughputs = [row["throughput_tok_s"] for row in rows.values()]
        performance_ok = bool(throughputs) and all(
            value > 0 and math.isfinite(value) for value in throughputs
        ) and (memory_leak_free or pilot)
        geomean = (
            math.exp(statistics.fmean(math.log(value) for value in throughputs))
            if performance_ok
            else 0.0
        )
        quality_measurement = None
        quality_gate = None
        if calibration or qualification or bootstrap or pilot:
            quality_ok, quality_failures = calibration_quality_diagnostic(rows)
        else:
            if quality_control:
                assert control_measurement_floor_ppm is not None
                quality_threshold_ppm = control_measurement_floor_ppm
            else:
                thresholds = workload_policy.get("quality_policy", {}).get(
                    "thresholds"
                )
                if (
                    not isinstance(thresholds, dict)
                    or thresholds.get("metric")
                    != "producer_relative_min_retention_ppm_v1"
                    or isinstance(thresholds.get("minimum_score_ppm"), bool)
                    or not isinstance(thresholds.get("minimum_score_ppm"), int)
                    or thresholds["minimum_score_ppm"] <= 0
                ):
                    raise ValueError("frozen quality threshold is invalid")
                quality_threshold_ppm = int(thresholds["minimum_score_ppm"])
            quality_ok, quality_failures, quality_measurement, quality_gate = quality_diagnostic(
                records,
                rows,
                reference,
                threshold_ppm=quality_threshold_ppm,
            )
        final_quality_hard_gate = bool(
            selection.mode == "final"
            and not calibration
            and not bootstrap
            and not qualification
            and not pilot
            and (
                quality_control
                or workload_policy.get("quality_policy", {}).get("status") == "QUALIFIED"
            )
        )
        # A control run must return its measured ACCEPT/REJECT classification
        # to the trusted producer even for a deliberately regressed candidate.
        # It is explicitly non-release evidence and does not turn REJECT into a
        # release-valid score.
        valid = performance_ok and (
            quality_control or quality_ok or not final_quality_hard_gate
        )
        detail = {
            "schema_version": 3,
            "calibration": calibration,
            "quality_control": quality_control,
            "qualification": qualification,
            "control_id": control_id,
            "expected_classification": expected_classification,
            "observed_classification": (
                "ACCEPT" if quality_ok else "REJECT"
            ) if quality_control else None,
            "release_evidence": not (
                calibration or quality_control or qualification or bootstrap or pilot
            ),
            "bootstrap": bootstrap,
            "pilot": pilot,
            "isolation": "root-controller/gold-free-agent-worker",
            "source_ok": True,
            "integrity_ok": True,
            "quality_ok": quality_ok,
            "performance_ok": performance_ok,
            "quality_failures": quality_failures,
            "quality_measurement": quality_measurement,
            "quality_gate": quality_gate,
            "quality_hard_gate": final_quality_hard_gate,
            "final_version": admission["final_version"],
            "num_candidate_versions": len(admission["versions"]),
            "source_bytes": admission["source_bytes"],
            "geomean_throughput_tok_s": geomean,
            "rows": rows,
            "records": records,
            "row_commitments": row_commitments,
            "run_deadline": (
                persistent_deadline.report_summary()
                if selection.mode == "final"
                else None
            ),
            "memory_checkpoints": checkpoints,
            "fixed_shape_memory_sentinel": memory_sentinel,
            "memory_leak_free": memory_leak_free,
            "dataset_manifest_sha256": hidden_sha256,
            "reference_manifest_sha256": reference_sha256,
            "reference_source": reference_path.name,
            "device_profile_key": reference_device_profile_key,
            "reference_profile_key": observed_reference_profile_key,
            "software_abi": observed_software_abi,
            "quality_runtime_sha256": quality_runtime_sha256,
            "workload_profile": WORKLOAD_PROFILE,
            "generation_count": GENERATION_COUNT,
            "requested_slots": REQUESTED_SLOTS,
            "selected_requested_slots": sum(
                int(case["gen_length"]) for case in selection.cases
            ),
            "workload_policy_sha256": sha256(WORKLOAD_POLICY_PATH),
            "memory_policy_sha256": sha256(MEMORY_POLICY_PATH),
            "quality_controls_sha256": (
                identity.quality_controls_sha256
                if selection.mode == "final"
                else None
            ),
            "runtime_provenance": runtime_provenance,
            "memory_fixture": memory_fixture,
            "workload_policy_status": workload_policy["release_status"],
            "memory_policy_status": memory_policy["release_status"],
            "visible_device_count": torch.cuda.device_count(),
            "gpu_name": gpu_name,
            "hardware": inventory,
            "measurement_profile_key": (
                observed_profile_key if require_calibrated_reference else None
            ),
            "device_peak_cap_gib": max_peak_gib,
            "memory_reserve_mib": memory_reserve_mib,
            "evaluation_mode": selection.mode,
            "round_id": trusted_metadata.get("round_id"),
            "evaluation_phase": "interactive" if str(trusted_metadata.get("round_id", "")).startswith(("agent-", "auto-")) else "final",
            "submission_number": selection.submission_number,
            "cohort_cycle": selection.cycle,
            "cohort_slot": selection.slot,
            "cohort_row_windows": [
                {
                    "row_id": row_id,
                    "start": start,
                    "quota": quota,
                    "cycle": cycle,
                }
                for row_id, start, quota, cycle in selection.row_windows
            ],
            "cohort_case_count": len(selection.cases),
            "cohort_row_counts": dict(selection.row_counts),
            "cohort_category_definition": CATEGORY_DEFINITION,
            "cohort_category_counts": dict(selection.category_counts),
            "cohort_bonus_categories": list(selection.bonus_categories),
            "cohort_case_set_sha256": hashlib.sha256(
                "\n".join(
                    sorted(
                        hashlib.sha256(str(case["case_id"]).encode("utf-8")).hexdigest()
                        for case in selection.cases
                    )
                ).encode("utf-8")
            ).hexdigest(),
        }
        write_results(args.log_dir, detail, geomean if valid else 0.0)
        return 0 if valid else 1
    except Exception as exc:
        return fail(
            args.log_dir,
            f"{type(exc).__name__}: {exc}",
            source_ok=isinstance(exc, (RuntimeError, ValueError))
            and not isinstance(exc, AdmissionError),
            calibration=calibration,
        )
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
