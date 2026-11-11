#!/usr/bin/env python3
"""Bind timeout-boundary archive bytes to one independent final report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_eval_artifacts import (
    runtime_identity,
    selected_reference_path,
    validate_private as validate_eval_private,
    validate_private_runtime,
    validate_runtime_snapshot,
)
from h100_policy_contract import (
    MIN_COMPUTE_CAPABILITY,
    MIN_DEVICE_MEMORY_MIB,
    capability_rank,
    validate_device_envelope,
    CATEGORY_COUNTS,
    CATEGORY_DEFINITION,
    GENERATION_COUNT,
    INTERACTIVE_BONUS_CATEGORIES,
    INTERACTIVE_CATEGORY_BASE_QUOTA,
    INTERACTIVE_MINIMUM_PER_ROW,
    INTERACTIVE_SELECTION,
    INTERACTIVE_TOTAL_CASES,
    PROFILE,
    REQUESTED_SLOTS,
    ROW_COUNTS,
    canonical_hidden_bytes,
    canonical_hidden_sha256,
    private_data_dir,
)


HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
# Scheduling modes recorded by the launcher.  The legacy mode runs Work and
# Judge on two distinct cards and evaluates the frozen archive through a
# separate local `sforge eval`; the single-GPU mode shares one whole-card
# lease serially and evaluates the frozen archive through the still-alive
# handoff coordinator (`sforge eval --standalone-final-handoff`).
EVALUATION_MODE = "dual_gpu_interactive_then_frozen_final"
SINGLE_GPU_EVALUATION_MODE = (
    "single_gpu_lease_manual_judge_then_standalone_final_handoff"
)
EVALUATION_MODES = (EVALUATION_MODE, SINGLE_GPU_EVALUATION_MODE)
EXPECTED_LIFECYCLES = {
    EVALUATION_MODE: {
        "interactive_evaluation": "concurrent_distinct_gpus",
        "final_evaluation": "after_work_stopped_and_archive_frozen",
        "final_work_lock": "held_until_cleanup",
        "judge_checkpoint_dir": "judge-checkpoints",
        "judge_checkpoint_mount": "/logs/verifier/checkpoints",
        "checkpoint_ownership": "root:root-0700",
    },
    SINGLE_GPU_EVALUATION_MODE: {
        "interactive_evaluation": "serial_single_gpu_lease_handoff",
        "final_evaluation": (
            "coordinator_standalone_final_after_work_retired_and_archive_frozen"
        ),
        "final_work_lock": "held_until_cleanup",
        "judge_checkpoint_dir": "judge-checkpoints",
        "judge_checkpoint_mount": "/logs/verifier/checkpoints",
        "checkpoint_ownership": "root:root-0700",
    },
}
EXPECTED_LOCK_SCHEMES = {
    EVALUATION_MODE: "two_uuid_flocks_deterministic_order",
    SINGLE_GPU_EVALUATION_MODE: "single_uuid_flock_work_judge_final",
}
FINAL_HANDOFF_SUBMISSION_ID = re.compile(r"^final-[0-9a-f]{12}$")

# The harness freezes the Work container at the deadline. In the dual-GPU mode
# the container is stopped before the separate final `sforge eval`, so the
# snapshot must prove every Work process stopped. In the single-GPU mode the
# harness pauses the container with never-resume persisted (the agent's work
# boundary), drains any queued Judge, and only then strictly retires it; the
# snapshot proves the processes were frozen (SIGSTOP via docker pause), and
# the strict retirement plus NVML idle attestation are validated separately.
SINGLE_GPU_FREEZE_MECHANISM = "docker_pause_never_resume_final"


def validate_freeze_boundary(freeze: object, mode: str) -> None:
    if not isinstance(freeze, dict):
        raise ValueError("final snapshot did not prove all Work processes stopped")
    if mode == SINGLE_GPU_EVALUATION_MODE:
        if freeze.get("mechanism") != SINGLE_GPU_FREEZE_MECHANISM:
            raise ValueError("single-GPU final snapshot did not use the never-resume pause")
        if freeze.get("all_work_processes_frozen") is not True:
            raise ValueError("final snapshot did not prove all Work processes frozen")
        if freeze.get("container_status") != "paused":
            raise ValueError("single-GPU final snapshot did not leave the Work container paused")
        return
    if freeze.get("all_work_processes_stopped") is not True:
        raise ValueError("final snapshot did not prove all Work processes stopped")
SINGLE_GPU_RETIREMENT_WINDOW_SECONDS = 120.0
SINGLE_GPU_MAX_JUDGE_DRAIN_SECONDS = 7200.0
PRIVATE_ARTIFACT_MAX_BYTES = 64 * 1024 * 1024
TASK_ID = "llada_h100_decoder_optimization_v2"
WORKLOAD_PROFILE = PROFILE
SFORGE_REQUIRED_FILES = {
    "sforge/cli.py",
    "sforge/harness/run_agent.py",
    "sforge/harness/backend/docker_backend.py",
    "sforge/harness/run_evaluation.py",
    "sforge/harness/grading.py",
    "sforge/harness/judge_server.py",
}
FINAL_ROW_COUNTS = ROW_COUNTS


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_at(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def exact_private_payload(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("private Judge artifact is missing or unsafe")
    try:
        with tarfile.open(path, "r:gz") as bundle:
            members = bundle.getmembers()
            if (
                len(members) != 1
                or not members[0].isfile()
                or members[0].issym()
                or members[0].islnk()
                or members[0].name != "decoder_private.json"
                or members[0].size <= 0
                or members[0].size > PRIVATE_ARTIFACT_MAX_BYTES
            ):
                raise ValueError("unexpected private artifact contents")
            stream = bundle.extractfile(members[0])
            if stream is None:
                raise ValueError("private artifact is unreadable")
            payload = stream.read(PRIVATE_ARTIFACT_MAX_BYTES + 1)
            if len(payload) != members[0].size:
                raise ValueError("private artifact payload is truncated")
            return payload
    except (OSError, tarfile.TarError) as exc:
        raise ValueError("private Judge artifact is unreadable") from exc


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def require_zero_integer(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise ValueError(f"{label} must be integer zero")


def nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def require_zero_number(value: object, label: str) -> None:
    if finite_number(value, label) != 0.0:
        raise ValueError(f"{label} must be zero")


def require_digest(value: object, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} has an invalid digest")
    return value


def validate_cohort_secret(run_root: Path, expected_commitment: str) -> None:
    secret_path = run_root / ".cohort-secret"
    if secret_path.is_symlink() or not secret_path.is_file():
        raise ValueError("private cohort secret file is missing or invalid")
    if stat.S_IMODE(secret_path.stat().st_mode) != 0o600:
        raise ValueError("private cohort secret file permissions are invalid")
    secret_text = secret_path.read_text(encoding="ascii").strip()
    if HEX_SHA256.fullmatch(secret_text) is None:
        raise ValueError("private cohort secret contents are invalid")
    observed_commitment = hashlib.sha256(secret_text.encode("ascii")).hexdigest()
    if observed_commitment != expected_commitment:
        raise ValueError("private cohort secret commitment mismatch")


def validate_final_workload(
    private: dict, canonical_cases: list[dict] | None = None
) -> float:
    if (
        private.get("evaluation_mode") != "final"
        or private.get("workload_profile") != WORKLOAD_PROFILE
        or private.get("generation_count") != GENERATION_COUNT
        or private.get("requested_slots") != REQUESTED_SLOTS
        or private.get("cohort_case_count") != GENERATION_COUNT
        or private.get("cohort_row_counts") != FINAL_ROW_COUNTS
        or private.get("cohort_category_definition") != CATEGORY_DEFINITION
        or private.get("cohort_category_counts") != CATEGORY_COUNTS
        or private.get("cohort_bonus_categories") != []
        or private.get("submission_number") is not None
        or private.get("cohort_cycle") is not None
        or private.get("cohort_slot") is not None
    ):
        raise ValueError("formal Judge did not select the complete stratified-520 workload")
    rows = private.get("rows")
    if not isinstance(rows, dict) or set(rows) != set(FINAL_ROW_COUNTS):
        raise ValueError("formal Judge row results are incomplete")
    for row_id, expected in FINAL_ROW_COUNTS.items():
        row = rows.get(row_id)
        if not isinstance(row, dict) or row.get("cases") != expected:
            raise ValueError(f"formal Judge row count mismatch: {row_id}")
    records = private.get("records")
    if not isinstance(records, list) or len(records) != GENERATION_COUNT:
        raise ValueError("formal Judge did not report 520 executed records")
    observed_rows = {row_id: 0 for row_id in FINAL_ROW_COUNTS}
    case_hashes = []
    canonical_rows: dict[str, str] | None = None
    if canonical_cases is not None:
        if len(canonical_cases) != GENERATION_COUNT:
            raise ValueError("canonical task workload is not 520 cases")
        canonical_rows = {}
        for case in canonical_cases:
            if (
                not isinstance(case, dict)
                or not isinstance(case.get("case_id"), str)
                or case.get("row_id") not in FINAL_ROW_COUNTS
            ):
                raise ValueError("canonical task workload contains an invalid case")
            case_hash = hashlib.sha256(case["case_id"].encode("utf-8")).hexdigest()
            if case_hash in canonical_rows:
                raise ValueError("canonical task workload repeats a case")
            canonical_rows[case_hash] = case["row_id"]
        canonical_counts = {row_id: 0 for row_id in FINAL_ROW_COUNTS}
        for row_id in canonical_rows.values():
            canonical_counts[row_id] += 1
        if canonical_counts != FINAL_ROW_COUNTS:
            raise ValueError("canonical task workload row mapping is invalid")
    aggregates: dict[str, dict[str, int | float]] = {
        row_id: {
            "cases": 0,
            "correct": 0,
            "parsed": 0,
            "useful_tokens": 0,
            "seconds": 0.0,
        }
        for row_id in FINAL_ROW_COUNTS
    }
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("formal Judge contains a malformed record")
        row_id = record.get("row_id")
        if row_id not in observed_rows:
            raise ValueError("formal Judge record has an unknown row")
        case_hash = require_digest(
            record.get("case_sha256"), "formal record case", HEX_SHA256
        )
        if canonical_rows is not None and canonical_rows.get(case_hash) != row_id:
            raise ValueError("formal record is assigned to the wrong canonical row")
        correct = record.get("correct")
        parsed = record.get("parsed")
        useful_tokens = record.get("useful_tokens")
        if not isinstance(correct, bool) or not isinstance(parsed, bool):
            raise ValueError("formal record quality flags are malformed")
        if (
            isinstance(useful_tokens, bool)
            or not isinstance(useful_tokens, int)
            or useful_tokens < 0
        ):
            raise ValueError("formal record useful-token count is malformed")
        seconds = finite_number(record.get("seconds"), "formal record seconds")
        if seconds <= 0:
            raise ValueError("formal record seconds must be positive")
        observed_rows[row_id] += 1
        case_hashes.append(case_hash)
        aggregate = aggregates[row_id]
        aggregate["cases"] = int(aggregate["cases"]) + 1
        aggregate["correct"] = int(aggregate["correct"]) + int(correct)
        aggregate["parsed"] = int(aggregate["parsed"]) + int(parsed)
        aggregate["useful_tokens"] = int(aggregate["useful_tokens"]) + useful_tokens
        aggregate["seconds"] = float(aggregate["seconds"]) + seconds
    if observed_rows != FINAL_ROW_COUNTS or len(set(case_hashes)) != GENERATION_COUNT:
        raise ValueError("formal Judge records are repeated or row-incomplete")
    commitment = hashlib.sha256(
        "\n".join(sorted(case_hashes)).encode("utf-8")
    ).hexdigest()
    if private.get("cohort_case_set_sha256") != commitment:
        raise ValueError("formal Judge cohort commitment does not match records")
    if canonical_rows is not None:
        if set(case_hashes) != set(canonical_rows):
            raise ValueError("formal Judge records do not match the canonical workload")
    throughputs = []
    for row_id, expected in FINAL_ROW_COUNTS.items():
        row = rows[row_id]
        aggregate = aggregates[row_id]
        for field in ("correct", "parsed", "useful_tokens"):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != aggregate[field]
            ):
                raise ValueError(f"formal Judge row {field} mismatch: {row_id}")
        row_seconds = finite_number(row.get("seconds"), f"formal row seconds: {row_id}")
        if row_seconds <= 0 or not math.isclose(
            row_seconds,
            float(aggregate["seconds"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"formal Judge row seconds mismatch: {row_id}")
        expected_throughput = int(aggregate["useful_tokens"]) / row_seconds
        throughput = finite_number(
            row.get("throughput_tok_s"), f"formal row throughput: {row_id}"
        )
        if throughput <= 0 or not math.isclose(
            throughput, expected_throughput, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"formal Judge row throughput mismatch: {row_id}")
        throughputs.append(throughput)
    return math.exp(sum(math.log(value) for value in throughputs) / len(throughputs))


def validate_interactive_history(result: dict, history: dict, run_id: str) -> dict:
    if history.get("run_id") != run_id:
        raise ValueError("run history identity mismatch")
    agent_count = nonnegative_integer(
        history.get("agent_submissions"), "run_history.agent_submissions"
    )
    require_zero_integer(history.get("auto_submissions"), "run_history.auto_submissions")
    require_zero_integer(history.get("public_submissions"), "run_history.public_submissions")
    entries = history.get("entries")
    if not isinstance(entries, list) or len(entries) != agent_count:
        raise ValueError("run history/agent submission count mismatch")
    expected_rounds = [f"agent-{index}" for index in range(1, agent_count + 1)]
    actual_rounds: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "submission":
            raise ValueError("run history contains a non-agent evaluation")
        if entry.get("status") not in ("completed", "error"):
            raise ValueError("run history contains a nonterminal evaluation")
        round_id = entry.get("round")
        if not isinstance(round_id, str):
            raise ValueError("run history contains an invalid round")
        actual_rounds.append(round_id)
    if actual_rounds != expected_rounds:
        raise ValueError("run history rounds are not contiguous")

    result_rounds = nonnegative_integer(result.get("total_rounds"), "final_result.total_rounds")
    result_agent = nonnegative_integer(result.get("agent_submissions"), "final_result.agent_submissions")
    require_zero_integer(result.get("auto_submissions"), "final_result.auto_submissions")
    if result_rounds != agent_count or result_agent != agent_count:
        raise ValueError("final result/history submission count mismatch")
    for key in ("best_score", "best_round", "best_pass_rate"):
        if result.get(key) != history.get(key):
            raise ValueError(f"final result/history {key} mismatch")
    return {"agent": agent_count, "auto": 0, "public": 0}


def validate_image_provenance(run_root: Path, experiment: dict) -> dict:
    local_manifest = run_root / "image-build-manifest.json"
    manifest = object_at(local_manifest)
    images = experiment.get("images")
    if not isinstance(images, dict):
        raise ValueError("experiment image provenance is missing")
    build_manifest = images.get("build_manifest")
    if not isinstance(build_manifest, dict):
        raise ValueError("experiment build-manifest provenance is missing")
    if build_manifest.get("path") != "image-build-manifest.json":
        raise ValueError("experiment build-manifest path mismatch")
    manifest_sha = require_digest(
        build_manifest.get("sha256"),
        "experiment image manifest",
        HEX_SHA256,
    )
    if digest(local_manifest) != manifest_sha:
        raise ValueError("run-local image manifest digest mismatch")

    source_digest = require_digest(
        manifest.get("source_digest"), "image source", HEX_SHA256
    )
    if images.get("source_digest") != source_digest:
        raise ValueError("experiment/image source digest mismatch")
    for role in ("work", "judge"):
        recorded = manifest.get(role)
        declared = images.get(role)
        if not isinstance(recorded, dict) or not isinstance(declared, dict):
            raise ValueError(f"{role} image provenance is missing")
        name = recorded.get("name")
        image_id = require_digest(recorded.get("id"), f"{role} image", IMAGE_ID)
        if not isinstance(name, str) or not name:
            raise ValueError(f"{role} image name is invalid")
        if declared != {"name": name, "id": image_id}:
            raise ValueError(f"experiment/{role} image provenance mismatch")

    runtime = experiment.get("sforge_runtime")
    if not isinstance(runtime, dict):
        raise ValueError("SForge runtime provenance is missing")
    require_digest(runtime.get("sha256"), "SForge runtime", HEX_SHA256)
    runtime_files = runtime.get("files")
    if (
        runtime.get("scope") != "all_sforge_python"
        or not isinstance(runtime_files, list)
        or not runtime_files
        or runtime_files != sorted(runtime_files)
        or len(runtime_files) != len(set(runtime_files))
        or runtime.get("file_count") != len(runtime_files)
        or any(
            not isinstance(path, str)
            or not path.startswith("sforge/")
            or not path.endswith(".py")
            or "/__pycache__/" in path
            or path.startswith("/")
            or ".." in Path(path).parts
            for path in runtime_files
        )
        or not SFORGE_REQUIRED_FILES.issubset(runtime_files)
    ):
        raise ValueError("SForge runtime file set/order mismatch")
    return {
        "build_manifest": {
            "path": str(local_manifest),
            "sha256": manifest_sha,
        },
        "source_digest": source_digest,
        "work": images["work"],
        "judge": images["judge"],
    }


def evaluation_mode(args: argparse.Namespace) -> str:
    mode = getattr(args, "evaluation_mode", EVALUATION_MODE)
    if mode not in EVALUATION_MODES:
        raise ValueError("unknown evaluation mode")
    return mode


def check_private_cohort(private_cohort: object) -> None:
    """The experiment manifest's private_cohort follows the interactive contract."""

    if (
        not isinstance(private_cohort, dict)
        or private_cohort.get("selection") != INTERACTIVE_SELECTION
        or private_cohort.get("category_definition") != CATEGORY_DEFINITION
        or private_cohort.get("total_cases") != INTERACTIVE_TOTAL_CASES
        or private_cohort.get("category_base_quota")
        != INTERACTIVE_CATEGORY_BASE_QUOTA
        or private_cohort.get("bonus_categories") != INTERACTIVE_BONUS_CATEGORIES
        or private_cohort.get("minimum_per_row") != INTERACTIVE_MINIMUM_PER_ROW
        or "per_row_quotas" in private_cohort
        or "per_category_quotas" in private_cohort
        or private_cohort.get("final_generations") != GENERATION_COUNT
        or private_cohort.get("final_requested_slots") != REQUESTED_SLOTS
    ):
        raise ValueError("experiment private-cohort contract mismatch")


def pre(args: argparse.Namespace) -> dict:
    root = args.task_root.resolve()
    run_root = root.parent
    mode = evaluation_mode(args)
    archive = root / "final_archive.tar.gz"
    result = object_at(root / "final_result.json")
    snapshot = object_at(root / "final_snapshot.json")
    history = object_at(root / "run_history.json")
    experiment_path = run_root / "experiment-manifest.json"
    experiment = object_at(experiment_path)
    task_source = Path(__file__).resolve().parents[1]
    if run_root.name != args.run_id or root.name != args.task_id:
        raise ValueError("formal run directory identity mismatch")
    if result.get("task") != args.task_id or result.get("run_id") != args.run_id:
        raise ValueError("final result identity mismatch")
    if result.get("model") != args.model or result.get("timed_out") is not True:
        raise ValueError("formal Work run did not reach its fixed deadline")
    runtime = finite_number(result.get("runtime_seconds"), "final_result.runtime_seconds")
    if mode == EVALUATION_MODE and (
        runtime < args.budget_seconds or runtime > args.budget_seconds + 90
    ):
        raise ValueError("formal Work runtime is outside the fixed deadline window")
    work_evaluations = validate_interactive_history(result, history, args.run_id)

    if (
        experiment.get("schema_version") != 3
        or experiment.get("run_id") != args.run_id
        or experiment.get("task_id") != args.task_id
        or experiment.get("model") != args.model
        or args.task_id != TASK_ID
    ):
        raise ValueError("experiment manifest identity mismatch")
    if finite_number(experiment.get("timeout_seconds"), "experiment timeout") != args.budget_seconds:
        raise ValueError("experiment timeout provenance mismatch")
    if finite_number(experiment.get("eval_interval_seconds"), "experiment interval") != args.interval_seconds:
        raise ValueError("experiment interval provenance mismatch")
    if experiment.get("auto_eval_enabled") is not False:
        raise ValueError("experiment enabled Work-time auto evaluation")
    if experiment.get("max_agent_submissions") is not None:
        raise ValueError("experiment did not declare unlimited manual submissions")
    cooldown = finite_number(
        experiment.get("manual_submission_cooldown_seconds"),
        "manual submission cooldown",
    )
    if cooldown <= 0:
        raise ValueError("experiment did not declare a positive frozen cooldown")
    private_cohort = experiment.get("private_cohort")
    check_private_cohort(private_cohort)
    secret_commitment = require_digest(
        private_cohort.get("secret_sha256"),
        "experiment private-cohort secret commitment",
        HEX_SHA256,
    )
    validate_cohort_secret(run_root, secret_commitment)
    lifecycle = experiment.get("lifecycle")
    if lifecycle != EXPECTED_LIFECYCLES[mode]:
        raise ValueError("experiment lifecycle provenance mismatch")
    if experiment.get("scheduling") != mode:
        raise ValueError("experiment scheduling mode mismatch")
    gpu_admission = experiment.get("gpu_admission")
    if (
        not isinstance(gpu_admission, dict)
        or gpu_admission.get("locks") != EXPECTED_LOCK_SCHEMES[mode]
        or gpu_admission.get("process_check") is not True
        or gpu_admission.get("running_container_assignment_check") is not True
    ):
        raise ValueError("experiment GPU admission provenance mismatch")
    checkpoint_dir = run_root / "judge-checkpoints"
    checkpoint_stat = checkpoint_dir.stat() if checkpoint_dir.is_dir() else None
    if (
        checkpoint_dir.is_symlink()
        or checkpoint_stat is None
        or checkpoint_dir.resolve().parent != run_root
        or checkpoint_stat.st_uid != 0
        or stat.S_IMODE(checkpoint_stat.st_mode) != 0o700
    ):
        raise ValueError("formal Judge checkpoint directory ownership/path is invalid")
    gpu_records = {}
    for role in ("work_gpu", "judge_gpu"):
        gpu = experiment.get(role)
        if (
            not isinstance(gpu, dict)
            or not isinstance(gpu.get("index"), int)
            or re.fullmatch(r"GPU-[0-9A-Fa-f-]+", str(gpu.get("uuid", ""))) is None
            or not isinstance(gpu.get("name"), str)
            or not gpu["name"]
            or finite_number(gpu.get("total_memory_mib"), f"{role} total memory")
            < MIN_DEVICE_MEMORY_MIB
            or finite_number(
                gpu.get("free_memory_mib_at_admission"), f"{role} free memory"
            ) <= 0
            or capability_rank(gpu.get("compute_capability"))
            < capability_rank(MIN_COMPUTE_CAPABILITY)
            or not isinstance(gpu.get("driver_version"), str)
            or not gpu["driver_version"]
            or gpu.get("container_cuda_visible_devices") != "0"
        ):
            raise ValueError(f"experiment {role} provenance is invalid")
        gpu_records[role] = gpu
    if "gpu" in experiment:
        raise ValueError("experiment carries an ambiguous role-less GPU record")
    same_device = (
        gpu_records["work_gpu"]["index"] == gpu_records["judge_gpu"]["index"]
        and gpu_records["work_gpu"]["uuid"] == gpu_records["judge_gpu"]["uuid"]
    )
    if mode == EVALUATION_MODE:
        if (
            gpu_records["work_gpu"]["index"] == gpu_records["judge_gpu"]["index"]
            or gpu_records["work_gpu"]["uuid"] == gpu_records["judge_gpu"]["uuid"]
        ):
            raise ValueError("Work and Judge GPU provenance is not distinct")
    else:
        if not same_device:
            raise ValueError("single GPU lease provenance names two devices")
        validate_single_gpu_handoff_provenance(experiment)
    policies = experiment.get("policies")
    if not isinstance(policies, dict):
        raise ValueError("experiment policy provenance is missing")
    for key, filename in (
        ("workload", "workload-policy.json"),
        ("memory", "memory-policy.json"),
        ("quality_controls", "quality-controls-manifest.json"),
    ):
        declared = policies.get(key)
        path = run_root / filename
        if (
            not isinstance(declared, dict)
            or declared.get("path") != filename
            or require_digest(declared.get("sha256"), f"{key} policy", HEX_SHA256)
            != digest(path)
        ):
            raise ValueError(f"experiment {key} policy provenance mismatch")
    image_provenance = validate_image_provenance(run_root, experiment)
    runtime_declared = experiment.get("runtime_provenance")
    runtime_path = run_root / "runtime-provenance.json"
    if (
        not isinstance(runtime_declared, dict)
        or runtime_declared.get("path") != "runtime-provenance.json"
        or runtime_declared.get("sha256") != digest(runtime_path)
    ):
        raise ValueError("experiment runtime-provenance snapshot binding is invalid")
    runtime_snapshot, runtime_snapshot_sha256 = validate_runtime_snapshot(
        runtime_path, run_root=run_root, task_root=task_source
    )
    if (
        experiment.get("sforge_runtime") != runtime_snapshot["sforge_runtime"]
        or experiment.get("judge_invariant_core")
        != runtime_snapshot["judge_invariant_core"]
    ):
        raise ValueError("experiment complete runtime provenance differs from snapshot")

    if snapshot.get("schema_version") != 2:
        raise ValueError("final snapshot schema mismatch")
    if snapshot.get("snapshot_boundary") != "whole_container_frozen":
        raise ValueError("final snapshot did not freeze the whole Work container")
    freeze = snapshot.get("freeze")
    capture = snapshot.get("capture")
    snapshot_archive = snapshot.get("archive")
    validate_freeze_boundary(freeze, mode)
    if not isinstance(capture, dict) or capture.get("work_container_running_during_capture") is not False:
        raise ValueError("final snapshot capture overlapped a running Work container")
    if not isinstance(snapshot_archive, dict):
        raise ValueError("final snapshot archive metadata is missing")
    elapsed = finite_number(snapshot.get("snapshot_elapsed_seconds"), "snapshot elapsed")
    if mode == SINGLE_GPU_EVALUATION_MODE:
        # The seal (admission closed, Work paused, never-resume persisted) is
        # the agent's work boundary; a queued Judge may be drained after it and
        # the Work container is strictly retired before the agent exec returns.
        seal = finite_number(freeze.get("seal_requested_elapsed_seconds"), "snapshot seal elapsed")
        if seal < args.budget_seconds or seal > args.budget_seconds + 90:
            raise ValueError("single-GPU seal was not requested at the timeout boundary")
        if elapsed < seal - 1.0 or elapsed - seal > SINGLE_GPU_MAX_JUDGE_DRAIN_SECONDS:
            raise ValueError("final snapshot does not follow the seal within the drain ceiling")
        if runtime < elapsed - 1.0 or runtime - elapsed > SINGLE_GPU_RETIREMENT_WINDOW_SECONDS:
            raise ValueError("final snapshot boundary does not bracket the retired Work runtime")
    elif elapsed < args.budget_seconds or elapsed > args.budget_seconds + 90:
        raise ValueError("snapshot was not captured at the timeout boundary")
    frozen_elapsed = finite_number(
        freeze.get("frozen_elapsed_seconds"), "snapshot frozen elapsed"
    )
    if not math.isclose(frozen_elapsed, elapsed, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("snapshot/freeze boundary mismatch")
    archive_sha = digest(archive)
    archive_size = archive.stat().st_size
    if archive_size <= 0:
        raise ValueError("final archive is empty")
    if snapshot.get("archive_sha256") != archive_sha:
        raise ValueError("snapshot/archive digest mismatch")
    if snapshot_archive.get("sha256") != archive_sha or snapshot_archive.get("size_bytes") != archive_size:
        raise ValueError("nested snapshot/archive provenance mismatch")
    if snapshot_archive.get("path") != "final_archive.tar.gz":
        raise ValueError("snapshot archive path mismatch")
    if result.get("archive_size_bytes") != archive_size:
        raise ValueError("final result/archive size mismatch")
    with tarfile.open(archive, "r:gz") as bundle:
        files = [item for item in bundle.getmembers() if item.isfile()]
        if not files:
            raise ValueError("final archive has no regular files")
        for member in bundle.getmembers():
            if member.issym() or member.islnk() or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError("unsafe final archive member")
    return {
        "schema_version": 1,
        "phase": "pre",
        "evaluation_mode": mode,
        "run_id": args.run_id,
        "task_id": args.task_id,
        "snapshot_elapsed_seconds": elapsed,
        "snapshot_captured_at": snapshot["captured_at"],
        "experiment_manifest": {
            "path": str(experiment_path),
            "sha256": digest(experiment_path),
        },
        "images": image_provenance,
        "sforge_runtime": experiment["sforge_runtime"],
        "runtime_provenance": {
            "path": str(runtime_path),
            "sha256": runtime_snapshot_sha256,
        },
        "judge_invariant_core": runtime_snapshot["judge_invariant_core"],
        "work_evaluations": work_evaluations,
        "final_archive": {
            "path": str(archive),
            "size_bytes": archive_size,
            "sha256": archive_sha,
            "regular_file_count": len(files),
        },
    }


def validate_single_gpu_handoff_provenance(experiment: dict) -> None:
    """The single-GPU lease run records the handoff binding and the two
    deliberately separate identities (deployment attestation with the UUID
    and observed clocks; measurement profile without them)."""
    handoff = experiment.get("resource_handoff")
    if (
        not isinstance(handoff, dict)
        or handoff.get("mode") != "single_allocation_serial_v1"
        or re.fullmatch(
            r"allocation-sha256:[0-9a-f]{64}", str(handoff.get("allocation_key", ""))
        )
        is None
        or not isinstance(handoff.get("trusted_lease_id"), str)
        or not handoff["trusted_lease_id"]
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(handoff.get("evaluation_contract_sha256", "")),
        )
        is None
        or handoff.get("execution_profile_id") != experiment.get("execution_profile_id")
        or not isinstance(handoff.get("final_judge"), dict)
        or handoff["final_judge"].get("path")
        != "coordinator_standalone_final_handoff"
    ):
        raise ValueError("experiment single-GPU handoff provenance is invalid")
    attestation = experiment.get("deployment_attestation")
    profile = experiment.get("measurement_profile")
    comparison = experiment.get("raw_score_comparison")
    if (
        not isinstance(attestation, dict)
        or attestation.get("accelerator_uuid") != experiment["judge_gpu"]["uuid"]
        or not isinstance(profile, dict)
        or "accelerator_uuid" in profile
        or "observed_sm_clock_mhz" in profile
        or not isinstance(profile.get("accelerator"), dict)
        or "uuid" in profile["accelerator"]
        or profile["accelerator"].get("model") != experiment["judge_gpu"]["name"]
        or profile.get("images")
        != {
            "work": experiment["images"]["work"]["id"],
            "judge": experiment["images"]["judge"]["id"],
        }
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(profile.get("comparability_key_sha256", ""))
        )
        is None
        or not isinstance(comparison, dict)
        or comparison.get("evaluation_contract_sha256")
        != handoff["evaluation_contract_sha256"]
        or comparison.get("measurement_profile_sha256")
        != profile["comparability_key_sha256"]
        or comparison.get("policy")
        != "same_evaluation_contract_and_measurement_profile_only"
    ):
        raise ValueError("experiment measurement identities are invalid")


def validate_final_handoff_lease(
    task_root: Path,
    *,
    final_eval_id: str,
    task_id: str,
    expected_archive_sha256: str,
    report_path: Path,
    submission_id: object,
) -> None:
    """Bind the persisted final report to the coordinator's durable lease.

    The standalone final Judge is named by the coordinator (``final-<12 hex>``),
    so the report's submission id cannot equal the final eval id as it does on
    the legacy path.  Instead the durable lease record of the source run must
    be CLOSED, name exactly this final run id and archive digest, and record
    the SHA-256 of the report bytes that were persisted.
    """
    if not isinstance(submission_id, str) or (
        FINAL_HANDOFF_SUBMISSION_ID.fullmatch(submission_id) is None
    ):
        raise ValueError("coordinator final report submission identity is invalid")
    lease_path = task_root / "resource_lease.json"
    if lease_path.is_symlink() or not lease_path.is_file():
        raise ValueError("durable resource lease record is missing")
    lease = object_at(lease_path)
    identity = lease.get("identity")
    final_judge = lease.get("final_judge")
    if (
        lease.get("state") != "CLOSED"
        or lease.get("recovery_policy") != "never_resume"
        or not isinstance(identity, dict)
        or identity.get("task_id") != task_id
        or identity.get("run_id") != task_root.parent.name
        or not isinstance(final_judge, dict)
        or not isinstance(final_judge.get("request"), dict)
        or not isinstance(final_judge.get("artifacts"), dict)
    ):
        raise ValueError("durable resource lease did not close on this final Judge")
    request = final_judge["request"]
    artifacts = final_judge["artifacts"]
    if (
        request.get("final_run_id") != final_eval_id
        or request.get("source_run_id") != task_root.parent.name
        or request.get("task_id") != task_id
        or request.get("archive_sha256") != "sha256:" + expected_archive_sha256
        or not isinstance(artifacts.get("report"), dict)
        or artifacts["report"].get("sha256") != "sha256:" + digest(report_path)
        or not isinstance(artifacts.get("submission"), dict)
        or artifacts["submission"].get("sha256")
        != "sha256:" + expected_archive_sha256
    ):
        raise ValueError("durable resource lease does not bind this final report")


def post(args: argparse.Namespace) -> dict:
    mode = evaluation_mode(args)
    source = args.task_root.resolve() / "final_archive.tar.gz"
    submission = args.final_task_root.resolve() / "submissions/manual-1"
    evaluated = submission / "submission.tar.gz"
    report_path = submission / "report.json"
    private_path = submission / "private_judge_artifacts.tar.gz"
    checkpoint_dir = args.final_task_root.resolve().parent / "judge-checkpoints"
    checkpoint_stat = checkpoint_dir.stat() if checkpoint_dir.is_dir() else None
    if (
        args.final_task_root.resolve().name != args.task_id
        or args.final_task_root.resolve().parent.name != args.final_eval_id
        or submission.name != "manual-1"
        or checkpoint_dir.is_symlink()
        or checkpoint_stat is None
        or checkpoint_dir.resolve().parent != args.final_task_root.resolve().parent
        or checkpoint_stat.st_uid != 0
        or stat.S_IMODE(checkpoint_stat.st_mode) != 0o700
    ):
        raise ValueError("independent final run directory identity mismatch")
    if digest(source) != args.expected_archive_sha256 or digest(evaluated) != args.expected_archive_sha256:
        raise ValueError("independent evaluation did not use the frozen archive")
    report = object_at(report_path)
    if report.get("task_id") != args.task_id:
        raise ValueError("independent report identity mismatch")
    if mode == EVALUATION_MODE:
        if report.get("submission_id") != args.final_eval_id:
            raise ValueError("independent report identity mismatch")
    else:
        validate_final_handoff_lease(
            args.task_root.resolve(),
            final_eval_id=args.final_eval_id,
            task_id=args.task_id,
            expected_archive_sha256=args.expected_archive_sha256,
            report_path=report_path,
            submission_id=report.get("submission_id"),
        )
    if report.get("timed_out") is not False:
        raise ValueError("independent Judge timed out")
    if report.get("valid") is not True:
        raise ValueError("independent Judge did not accept the frozen submission")
    require_zero_integer(report.get("errors"), "independent report errors")
    score = finite_number(report.get("score"), "independent report score")
    if score <= 0:
        raise ValueError("independent Judge returned no positive accepted score")
    payload = exact_private_payload(private_path)
    private = json.loads(payload)
    if not isinstance(private, dict):
        raise ValueError("private Judge artifact is not an object")
    if private.get("calibration") is not False:
        raise ValueError("formal Judge accidentally ran in calibration mode")
    if private.get("schema_version") != 3:
        raise ValueError("formal private report schema mismatch")
    if private.get("isolation") != "root-controller/gold-free-agent-worker":
        raise ValueError("formal Judge did not use the isolated worker protocol")
    task_source = Path(__file__).resolve().parents[1]
    final_run_root = args.final_task_root.resolve().parent
    final_runtime_path = final_run_root / "runtime-provenance.json"
    final_runtime, final_runtime_sha256 = validate_runtime_snapshot(
        final_runtime_path, run_root=final_run_root, task_root=task_source
    )
    validate_private_runtime(private, final_runtime, final_runtime_sha256)
    hidden_path = task_source / "images/judge/private_data/hidden_cases.json"
    reference_path = selected_reference_path(private_data_dir(task_source), private)
    quality_path = task_source / "images/judge/verifier/quality.py"
    canonical_cases = json.loads(canonical_hidden_bytes(task_source))
    if not isinstance(canonical_cases, list):
        raise ValueError("canonical workload is not a list")
    recomputed_score = validate_final_workload(private, canonical_cases)
    validation_mode = "pilot" if args.pilot else "final"
    validated_score, row_identity = validate_eval_private(
        private,
        mode=validation_mode,
        hidden_path=hidden_path,
        reference_path=reference_path,
        quality_path=quality_path,
        workload_policy_path=task_source / "workload_policy.json",
        memory_policy_path=task_source / "memory_policy.json",
    )
    if not math.isclose(
        validated_score, recomputed_score, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("formal and evaluation validators disagree on score")
    if private.get("dataset_manifest_sha256") != canonical_hidden_sha256(task_source):
        raise ValueError("formal Judge dataset provenance mismatch")
    if private.get("reference_manifest_sha256") != digest(reference_path):
        raise ValueError("formal Judge reference provenance mismatch")
    if private.get("quality_runtime_sha256") != digest(quality_path):
        raise ValueError("formal Judge quality-runtime provenance mismatch")
    required_gates = [
        "source_ok",
        "integrity_ok",
        "performance_ok",
    ]
    if not args.pilot:
        required_gates.extend(
            ("memory_leak_free", "quality_ok", "quality_hard_gate")
        )
    for gate in required_gates:
        if private.get(gate) is not True:
            raise ValueError(f"formal private gate failed: {gate}")
    if (
        private.get("workload_policy_status") != ("PROVISIONAL" if args.pilot else "FROZEN")
        or private.get("memory_policy_status") != "FROZEN"
        or private.get("visible_device_count") != 1
    ):
        raise ValueError("formal private report lacks frozen policy provenance")
    validate_device_envelope(private, "formal private report")
    precheck = object_at(args.task_root.resolve() / "formal-precheck.json")
    pre_images = precheck.get("images")
    run_root = args.task_root.resolve().parent
    pre_runtime = precheck.get("runtime_provenance")
    if (
        not isinstance(pre_runtime, dict)
        or pre_runtime.get("sha256") != final_runtime_sha256
        or precheck.get("sforge_runtime") != final_runtime["sforge_runtime"]
        or precheck.get("judge_invariant_core")
        != final_runtime["judge_invariant_core"]
    ):
        raise ValueError("formal final runtime snapshot differs from the frozen run")
    expected_identity = {
        "run_id": args.final_eval_id,
        "archive_sha256": args.expected_archive_sha256,
        "work_image_id": pre_images.get("work", {}).get("id") if isinstance(pre_images, dict) else None,
        "judge_image_id": pre_images.get("judge", {}).get("id") if isinstance(pre_images, dict) else None,
        "model_validation_sha256": digest(run_root / "full-model-validation.json"),
        "runtime_entry_sha256": digest(task_source / "scripts/sforge_h100_entry.py"),
        "source_digest": pre_images.get("source_digest") if isinstance(pre_images, dict) else None,
        "workload_policy_sha256": digest(task_source / "workload_policy.json"),
        "memory_policy_sha256": digest(task_source / "memory_policy.json"),
        "quality_controls_sha256": digest(task_source / "quality_controls.json"),
        "hidden_manifest_sha256": canonical_hidden_sha256(task_source),
        "row_order_sha256": hashlib.sha256(
            ("\n".join(FINAL_ROW_COUNTS) + "\n").encode()
        ).hexdigest(),
        **runtime_identity(final_runtime, final_runtime_sha256),
    }
    if row_identity != expected_identity:
        raise ValueError("formal row commitments differ from frozen host identity")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("independent report metrics are missing")
    score_views = {
        "private.reward": finite_number(private.get("reward"), "private reward"),
        "private.geomean": finite_number(
            private.get("geomean_throughput_tok_s"), "private geomean"
        ),
        "metrics.reward": finite_number(metrics.get("reward"), "metrics reward"),
    }
    if any(
        value <= 0
        or not math.isclose(value, recomputed_score, rel_tol=1e-12, abs_tol=1e-12)
        for value in (*score_views.values(), score)
    ):
        raise ValueError("report/private/metrics score mismatch")
    return {
        "schema_version": 1,
        "phase": "post",
        "evaluation_mode": mode,
        "task_id": args.task_id,
        "final_eval_id": args.final_eval_id,
        "evaluated_archive": {"path": str(evaluated), "sha256": digest(evaluated)},
        "final_report": {"path": str(report_path), "sha256": digest(report_path)},
        "private_judge_artifact": {
            "path": str(private_path),
            "sha256": digest(private_path),
            "member_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "validated_report": report,
        "row_atomic_identity": row_identity,
    }


INVALID_FINAL_STATUS = "invalid_final"
INVALID_FINAL_EXIT_CODE = 3


def invalid_final_record(args: argparse.Namespace, error: BaseException) -> dict:
    """Describe a rejected final as a durable, machine-readable attestation.

    The record deliberately carries no `evaluated_archive`, `final_report` or
    `private_judge_artifact` binding: nothing downstream may treat it as a
    pass.  It only preserves *why* the fail-closed validator refused, so the
    reason survives the launcher's stdout.
    """
    return {
        "schema_version": 1,
        "evaluation_mode": args.evaluation_mode,
        "status": INVALID_FINAL_STATUS,
        "phase": args.phase,
        "final_eval_id": args.final_eval_id,
        "task_id": args.task_id,
        "expected_archive_sha256": args.expected_archive_sha256,
        "verify_suite": args.verify_suite,
        "pilot": bool(args.pilot),
        "final_task_root": str(args.final_task_root),
        "reason": str(error) or type(error).__name__,
        "error_type": type(error).__name__,
    }


def main() -> int:
    """Validate one phase and emit its JSON attestation.

    A rejected final (post phase) still fails closed: the process exits
    non-zero, but it first prints a complete `invalid_final` record on stdout
    so the launcher can persist `formal-postcheck.json` instead of discarding
    the temporary file.
    """
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    first = sub.add_parser("pre")
    first.add_argument("--task-root", type=Path, required=True)
    first.add_argument("--run-id", required=True)
    first.add_argument("--task-id", required=True)
    first.add_argument("--model", required=True)
    first.add_argument("--budget-seconds", type=float, required=True)
    first.add_argument("--interval-seconds", type=float, required=True)
    first.add_argument(
        "--evaluation-mode",
        choices=EVALUATION_MODES,
        default=EVALUATION_MODE,
        help="scheduling mode recorded by the launcher (legacy dual GPU or single-GPU lease)",
    )
    second = sub.add_parser("post")
    second.add_argument("--task-root", type=Path, required=True)
    second.add_argument("--final-task-root", type=Path, required=True)
    second.add_argument("--final-eval-id", required=True)
    second.add_argument("--task-id", required=True)
    second.add_argument("--expected-archive-sha256", required=True)
    second.add_argument("--verify-suite")
    second.add_argument("--pilot", action="store_true")
    second.add_argument(
        "--evaluation-mode",
        choices=EVALUATION_MODES,
        default=EVALUATION_MODE,
        help="scheduling mode recorded by the launcher (legacy dual GPU or single-GPU lease)",
    )
    args = parser.parse_args()
    if args.phase == "pre":
        value = pre(args)
    else:
        try:
            value = post(args)
        except Exception as exc:  # noqa: BLE001 - every rejection is recorded, none is accepted
            record = invalid_final_record(args, exc)
            print(json.dumps(record, sort_keys=True, indent=2))
            print(f"error: {record['reason']}", file=sys.stderr)
            return INVALID_FINAL_EXIT_CODE
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
