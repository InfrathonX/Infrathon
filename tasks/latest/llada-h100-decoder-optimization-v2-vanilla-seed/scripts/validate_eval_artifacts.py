#!/usr/bin/env python3
"""Fail-closed validation and attestation of one manual SForge evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "images/judge/verifier")
)
from h100_policy_contract import (
    validate_device_envelope,
    CATEGORY_COUNTS,
    CATEGORY_DEFINITION,
    GENERATION_COUNT,
    JUDGE_INVARIANT_CORE_HASH_FIELDS,
    PROFILE,
    REQUESTED_SLOTS,
    RUNTIME_ATTESTATION_HASH_FIELDS,
    ROW_COUNTS,
    canonical_hidden_bytes,
    canonical_hidden_sha256,
    is_sha256,
    validate_runtime_provenance,
    validate_memory_policy,
    validate_workload_policy,
    private_data_dir,
)
from quality_threshold import (
    AGGREGATE_FLOOR_PPM,
    BENCHMARK_FLOOR_PPM,
    MAX_USEFUL_TOKENS_RETENTION_PPM,
    QUALITY_GATE,
    benchmark_retention,
    fixed_gate_failures,
    measure_quality_retention,
    validate_measurement,
)


TASK_ID = "llada_h100_decoder_optimization_v2"
WORKLOAD_PROFILE = PROFILE
PRIVATE_ARTIFACT_MAX_BYTES = 64 * 1024 * 1024
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
EXPECTED_ROWS = ROW_COUNTS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Freeze the executing validator bytes at module initialization. If this file
# changes while validation is running, the later calibration/layout comparison
# fails instead of attributing the run to code that was never executed.
EXECUTING_VALIDATOR_SHA256 = sha256(Path(__file__).resolve())


def object_at(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or unsafe regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def canonical_commitment_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def validate_row_memory_sentinel(value: object) -> int:
    fields = (
        "torch_peak_allocated_bytes",
        "torch_peak_reserved_bytes",
        "nvml_peak_device_used_bytes",
    )
    if not isinstance(value, dict):
        raise ValueError("row memory sentinel is missing")
    start = value.get("start")
    end = value.get("end")
    repeat = value.get("repeat")
    deltas = value.get("deltas_bytes")
    repeat_deltas = value.get("repeat_deltas_bytes")
    bound = value.get("max_material_growth_bytes")
    repeat_bound = value.get("max_repeat_growth_bytes")
    if (
        value.get("schema_version") != 2
        or not isinstance(value.get("stable"), bool)
        or not isinstance(value.get("repeat_stable"), bool)
        or not isinstance(start, dict)
        or not isinstance(end, dict)
        or not isinstance(repeat, dict)
        or not isinstance(deltas, dict)
        or not isinstance(repeat_deltas, dict)
        or set(start) != set(fields)
        or set(end) != set(fields)
        or set(repeat) != set(fields)
        or set(deltas) != set(fields)
        or set(repeat_deltas) != set(fields)
        or isinstance(bound, bool)
        or not isinstance(bound, int)
        or bound <= 0
        or isinstance(repeat_bound, bool)
        or not isinstance(repeat_bound, int)
        or repeat_bound != 16 * 1024 * 1024
        or any(
            HEX_SHA256.fullmatch(str(value.get(field, ""))) is None
            for field in (
                "start_output_sha256",
                "end_output_sha256",
                "repeat_output_sha256",
            )
        )
    ):
        raise ValueError("row memory sentinel schema is invalid")
    expected = {}
    expected_repeat = {}
    for field in fields:
        before = start[field]
        after = end[field]
        repeated = repeat[field]
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (before, after, repeated)
        ):
            raise ValueError("row memory sentinel metrics are invalid")
        expected[field] = after - before
        expected_repeat[field] = repeated - after
    observed = max(0, *expected.values())
    observed_repeat = max(0, *expected_repeat.values())
    if (
        deltas != expected
        or repeat_deltas != expected_repeat
        or value.get("observed_max_positive_growth_bytes") != observed
        or value.get("observed_max_positive_repeat_growth_bytes")
        != observed_repeat
        or value.get("repeat_stable") is not (observed_repeat <= repeat_bound)
        or value.get("stable")
        is not (
            observed <= bound
            and observed_repeat <= repeat_bound
            and value.get("start_output_sha256")
            == value.get("end_output_sha256")
            == value.get("repeat_output_sha256")
        )
    ):
        raise ValueError("row memory sentinel evidence is internally inconsistent")
    return observed


def exact_private_payload(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("private Judge artifact is missing or unsafe")
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
            raise ValueError(
                "private Judge artifact must contain exactly one bounded regular decoder_private.json"
            )
        stream = bundle.extractfile(members[0])
        if stream is None:
            raise ValueError("private Judge artifact payload is unreadable")
        payload = stream.read(PRIVATE_ARTIFACT_MAX_BYTES + 1)
        if len(payload) != members[0].size:
            raise ValueError("private Judge artifact payload is truncated")
        return payload


def validate_model_attestation(model: dict[str, Any]) -> None:
    if (
        model.get("status") != "PASS"
        or model.get("full") is not True
        or model.get("hashes_checked") != 16
        or model.get("logical_size_bytes") != 16037449215
        or model.get("revision") != "6059b30c9531767ce47e0ca6a8161a6bb0c67693"
    ):
        raise ValueError("full model validation attestation is invalid")


def validate_image_manifest(manifest: dict[str, Any]) -> None:
    if (
        HEX_SHA256.fullmatch(str(manifest.get("source_digest", ""))) is None
        or any(
            not isinstance(manifest.get(role), dict)
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(manifest[role].get("id", ""))
            )
            is None
            for role in ("work", "judge")
        )
    ):
        raise ValueError("evaluation image manifest is invalid")


def repository_root(task_root: Path) -> Path:
    for candidate in (task_root, *task_root.parents):
        if (candidate / "sforge").is_dir():
            return candidate
    raise ValueError("repository SForge runtime tree is missing")


def sforge_runtime_from_source(task_root: Path) -> dict[str, Any]:
    repo = repository_root(task_root)
    paths = sorted(
        (path.resolve() for path in (repo / "sforge").rglob("*.py")),
        key=lambda path: str(path),
    )
    files = [path.relative_to(repo).as_posix() for path in paths]
    if not files or any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("repository SForge runtime file set is unsafe")
    file_set_sha256 = hashlib.sha256(
        ("\n".join(files) + "\n").encode()
    ).hexdigest()
    # This is exactly `find ... -print0 | sort -z | xargs -0 sha256sum | sha256sum`.
    digest_input = "".join(
        f"{sha256(path)}  {name}\n" for path, name in zip(paths, files, strict=True)
    ).encode()
    return {
        "scope": "all_sforge_python",
        "sha256": hashlib.sha256(digest_input).hexdigest(),
        "files": files,
        "file_count": len(files),
        "file_set_sha256": file_set_sha256,
    }


def validate_runtime_snapshot(
    path: Path, *, run_root: Path, task_root: Path
) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    if (
        path.is_symlink()
        or not resolved.is_file()
        or resolved.parent != run_root
        or resolved.stat().st_mode & 0o222
    ):
        raise ValueError("runtime provenance snapshot is not immutable and run-local")
    snapshot = validate_runtime_provenance(object_at(resolved))
    if snapshot["sforge_runtime"] != sforge_runtime_from_source(task_root):
        raise ValueError("runtime provenance SForge bytes/file set differ from source")
    expected_core = {
        "verify_decoder_sha256": sha256(
            task_root / "images/judge/verifier/verify_decoder.py"
        ),
        "candidate_worker_sha256": sha256(
            task_root / "images/judge/verifier/candidate_worker.py"
        ),
        "cohort_sha256": sha256(task_root / "images/judge/verifier/cohort.py"),
        "quality_sha256": sha256(task_root / "images/judge/verifier/quality.py"),
        "source_admission_sha256": sha256(
            task_root / "images/judge/verifier/source_admission.py"
        ),
    }
    if snapshot["judge_invariant_core"] != expected_core:
        raise ValueError("runtime provenance Judge invariant core differs from source")
    return snapshot, sha256(resolved)


def runtime_identity(snapshot: dict[str, Any], snapshot_sha256: str) -> dict[str, str]:
    sforge = snapshot["sforge_runtime"]
    core = snapshot["judge_invariant_core"]
    return {
        "runtime_provenance_sha256": snapshot_sha256,
        "sforge_runtime_sha256": sforge["sha256"],
        "sforge_runtime_file_set_sha256": sforge["file_set_sha256"],
        **core,
    }


def validate_private_runtime(
    private: dict[str, Any], snapshot: dict[str, Any], snapshot_sha256: str
) -> None:
    value = private.get("runtime_provenance")
    expected = {
        "schema_version": 1,
        "runtime_provenance_sha256": snapshot_sha256,
        "sforge_runtime": {
            key: snapshot["sforge_runtime"][key]
            for key in ("scope", "sha256", "file_set_sha256", "file_count")
        },
        "judge_invariant_core": snapshot["judge_invariant_core"],
    }
    if value != expected:
        raise ValueError("private runtime provenance differs from the host snapshot")


def validate_memory_fixture_snapshot(
    path: Path,
    *,
    run_root: Path,
    task_root: Path,
    kind: str,
    archive_sha256: str,
    archive_size_bytes: int,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    resolved = path.resolve()
    if (
        path.is_symlink()
        or not resolved.is_file()
        or resolved.parent != run_root
        or resolved.stat().st_mode & 0o222
    ):
        raise ValueError("memory fixture snapshot is not immutable and run-local")
    manifest = object_at(resolved)
    fixtures = manifest.get("fixtures")
    workload_policy = object_at(task_root / "workload_policy.json")
    approval = manifest.get("approval_id")
    selected = fixtures.get(kind) if isinstance(fixtures, dict) else None
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "PRECOMMITTED"
        or manifest.get("task_id") != TASK_ID
        or manifest.get("profile") != WORKLOAD_PROFILE
        or manifest.get("workload_policy_sha256")
        != sha256(task_root / "workload_policy.json")
        or manifest.get("memory_policy_sha256")
        != sha256(task_root / "memory_policy.json")
        or not isinstance(approval, str)
        or SAFE_ID.fullmatch(approval) is None
        or not isinstance(fixtures, dict)
        or set(fixtures) != {"corrected_reference", "near_cap_stress"}
        or not isinstance(selected, dict)
        or selected.get("kind") != kind
        or selected.get("archive_sha256") != archive_sha256
        or selected.get("archive_size_bytes") != archive_size_bytes
        or workload_policy.get("profile") != WORKLOAD_PROFILE
    ):
        raise ValueError("memory fixture snapshot does not select the evaluated archive")
    return manifest, sha256(resolved), selected


REFERENCE_PROFILE_KEY = re.compile(r"[0-9a-f]{64}")


def selected_reference_path(private_data: Path, private: dict[str, Any]) -> Path:
    """The quality reference the Judge scored against.

    The verifier prefers reference_cache/<reference_profile_key>.json for the GPU
    it observed and falls back to the installed reference_quality.json, so one
    Judge image serves every calibrated GPU model.  Resolve the same file here;
    the report's reference_source must name it.
    """
    key = private.get("reference_profile_key")
    selected = private_data / "reference_quality.json"
    if isinstance(key, str) and REFERENCE_PROFILE_KEY.fullmatch(key):
        cached = private_data / "reference_cache" / f"{key}.json"
        if cached.is_file() and not cached.is_symlink():
            selected = cached
    source = private.get("reference_source")
    if source is not None and source != selected.name:
        raise ValueError("private report quality-reference source mismatch")
    return selected


def validate_private(
    private: dict[str, Any],
    *,
    mode: str,
    hidden_path: Path,
    reference_path: Path,
    quality_path: Path,
    workload_policy_path: Path,
    memory_policy_path: Path,
) -> tuple[float, dict[str, str]]:
    calibration = mode == "calibration"
    quality_control = mode == "quality_control"
    qualification = mode == "qualification"
    pilot = mode == "pilot"
    if private.get("schema_version") != 3:
        raise ValueError("private report schema mismatch")
    if private.get("calibration") is not calibration:
        raise ValueError("private report calibration mode mismatch")
    if private.get("quality_control") is not quality_control:
        raise ValueError("private report quality-control mode mismatch")
    if private.get("qualification") is not qualification:
        raise ValueError("private report qualification mode mismatch")
    if private.get("isolation") != "root-controller/gold-free-agent-worker":
        raise ValueError("private report isolation protocol mismatch")
    for gate in ("source_ok", "integrity_ok", "performance_ok"):
        if private.get(gate) is not True:
            raise ValueError(f"private hard gate failed: {gate}")
    if not isinstance(private.get("quality_ok"), bool):
        raise ValueError("private quality diagnostic is not boolean")
    if private.get("release_evidence") is not (mode == "final"):
        raise ValueError("private report release-evidence classification is invalid")
    if quality_control:
        observed = "ACCEPT" if private["quality_ok"] else "REJECT"
        if (
            private.get("quality_hard_gate") is not True
            or private.get("observed_classification") != observed
            or private.get("expected_classification") not in {"ACCEPT", "REJECT"}
            or not isinstance(private.get("control_id"), str)
            or not private["control_id"]
        ):
            raise ValueError("private quality-control classification is invalid")
    elif mode == "final" and (
        private.get("quality_hard_gate") is not True
        or private.get("quality_ok") is not True
    ):
        raise ValueError("release evaluation did not pass the frozen quality hard gate")
    if (
        private.get("evaluation_mode") != "final"
        or private.get("workload_profile") != WORKLOAD_PROFILE
        or private.get("generation_count") != GENERATION_COUNT
        or private.get("requested_slots") != REQUESTED_SLOTS
        or private.get("cohort_case_count") != GENERATION_COUNT
        or private.get("cohort_row_counts") != EXPECTED_ROWS
        or private.get("cohort_category_definition") != CATEGORY_DEFINITION
        or private.get("cohort_category_counts") != CATEGORY_COUNTS
        or private.get("cohort_bonus_categories") != []
        or private.get("submission_number") is not None
        or private.get("cohort_cycle") is not None
        or private.get("cohort_slot") is not None
    ):
        raise ValueError("evaluation did not select the canonical stratified-520 workload")
    task_root = hidden_path.parents[3]
    hidden_sha = canonical_hidden_sha256(task_root)
    if private.get("dataset_manifest_sha256") != hidden_sha:
        raise ValueError("private report hidden-workload provenance mismatch")
    if private.get("reference_manifest_sha256") != sha256(reference_path):
        raise ValueError("private report quality-reference provenance mismatch")
    if private.get("quality_runtime_sha256") != sha256(quality_path):
        raise ValueError("private report quality-runtime provenance mismatch")
    if private.get("workload_policy_sha256") != sha256(workload_policy_path):
        raise ValueError("private report workload-policy provenance mismatch")
    if private.get("memory_policy_sha256") != sha256(memory_policy_path):
        raise ValueError("private report memory-policy provenance mismatch")
    if (
        private.get("workload_policy_status")
        != (
            "PENDING"
            if qualification
            else "PROVISIONAL"
            if mode in {"calibration", "quality_control"}
            else "PROVISIONAL"
            if pilot
            else "FROZEN"
        )
        or private.get("memory_policy_status")
        != ("PENDING" if qualification else "FROZEN")
        or private.get("visible_device_count") != 1
    ):
        raise ValueError("private report lacks the expected frozen policy provenance")
    validate_device_envelope(private, "private report")

    canonical = json.loads(canonical_hidden_bytes(task_root))
    if not isinstance(canonical, list) or len(canonical) != GENERATION_COUNT:
        raise ValueError("canonical hidden workload is invalid")
    canonical_rows = {
        hashlib.sha256(str(case["case_id"]).encode("utf-8")).hexdigest(): case[
            "row_id"
        ]
        for case in canonical
    }
    if (
        len(canonical_rows) != GENERATION_COUNT
        or dict(Counter(canonical_rows.values())) != EXPECTED_ROWS
    ):
        raise ValueError("canonical hidden workload row mapping is invalid")
    rows = private.get("rows")
    records = private.get("records")
    if not isinstance(rows, dict) or set(rows) != set(EXPECTED_ROWS):
        raise ValueError("private report row set is incomplete")
    if not isinstance(records, list) or len(records) != GENERATION_COUNT:
        raise ValueError("private report must contain exactly 520 records")
    observed_rows: Counter[str] = Counter()
    observed_hashes: list[str] = []
    aggregates: dict[str, dict[str, int | float]] = {
        row_id: {
            "cases": 0,
            "correct": 0,
            "parsed": 0,
            "useful_tokens": 0,
            "seconds": 0.0,
        }
        for row_id in EXPECTED_ROWS
    }
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("private report contains a malformed record")
        row_id = record.get("row_id")
        case_hash = record.get("case_sha256")
        if row_id not in EXPECTED_ROWS or HEX_SHA256.fullmatch(str(case_hash)) is None:
            raise ValueError("private report contains invalid record provenance")
        if canonical_rows.get(str(case_hash)) != row_id:
            raise ValueError("private record is assigned to the wrong canonical row")
        correct = record.get("correct")
        parsed = record.get("parsed")
        useful_tokens = record.get("useful_tokens")
        if not isinstance(correct, bool) or not isinstance(parsed, bool):
            raise ValueError("private record quality flags are malformed")
        if (
            isinstance(useful_tokens, bool)
            or not isinstance(useful_tokens, int)
            or useful_tokens < 0
        ):
            raise ValueError("private record useful-token count is malformed")
        seconds = finite(record.get("seconds"), "private record seconds")
        if seconds <= 0:
            raise ValueError("private record seconds must be positive")
        observed_rows[str(row_id)] += 1
        observed_hashes.append(str(case_hash))
        aggregate = aggregates[str(row_id)]
        aggregate["cases"] = int(aggregate["cases"]) + 1
        aggregate["correct"] = int(aggregate["correct"]) + int(correct)
        aggregate["parsed"] = int(aggregate["parsed"]) + int(parsed)
        aggregate["useful_tokens"] = int(aggregate["useful_tokens"]) + useful_tokens
        aggregate["seconds"] = float(aggregate["seconds"]) + seconds
    if dict(observed_rows) != EXPECTED_ROWS or set(observed_hashes) != set(canonical_rows):
        raise ValueError("private records do not equal the canonical workload")
    if len(set(observed_hashes)) != GENERATION_COUNT:
        raise ValueError("private report repeats canonical records")
    commitment = hashlib.sha256(
        "\n".join(sorted(observed_hashes)).encode("utf-8")
    ).hexdigest()
    if private.get("cohort_case_set_sha256") != commitment:
        raise ValueError("private cohort commitment mismatch")
    row_commitments = private.get("row_commitments")
    if not isinstance(row_commitments, list) or len(row_commitments) != len(EXPECTED_ROWS):
        raise ValueError("private report lacks all row-atomic commitments")
    row_identity: dict[str, str] | None = None
    run_deadline = private.get("run_deadline")
    if not isinstance(run_deadline, dict):
        raise ValueError("private report lacks its persistent outer deadline")
    deadline_frozen = dict(run_deadline)
    deadline_commitment = deadline_frozen.pop("commitment_sha256", None)
    deadline_started = finite(
        run_deadline.get("started_at_unix_seconds"), "deadline start"
    )
    deadline_end = finite(
        run_deadline.get("outer_deadline_unix_seconds"), "deadline end"
    )
    deadline_timeout = finite(
        run_deadline.get("outer_timeout_seconds"), "deadline timeout"
    )
    if (
        run_deadline.get("schema_version") != 1
        or not isinstance(run_deadline.get("identity"), dict)
        or HEX_SHA256.fullmatch(str(deadline_commitment or "")) is None
        or deadline_commitment
        != hashlib.sha256(canonical_commitment_bytes(deadline_frozen)).hexdigest()
        or not 0 < deadline_timeout <= 7_200
        or not math.isclose(
            deadline_end - deadline_started,
            deadline_timeout,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("persistent outer deadline commitment is invalid")
    by_row_records: dict[str, list[dict[str, Any]]] = {
        row_id: [record for record in records if record["row_id"] == row_id]
        for row_id in EXPECTED_ROWS
    }
    for row_index, (row_id, expected_count) in enumerate(EXPECTED_ROWS.items()):
        row_commit = row_commitments[row_index]
        if not isinstance(row_commit, dict):
            raise ValueError("private row commitment is malformed")
        identity = row_commit.get("identity")
        members = by_row_records[row_id]
        case_hashes = [str(record["case_sha256"]) for record in members]
        records_sha = hashlib.sha256(
            json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        case_set_sha = hashlib.sha256(
            ("\n".join(sorted(case_hashes)) + "\n").encode()
        ).hexdigest()
        if (
            row_commit.get("schema_version") != 2
            or row_commit.get("status") != "COMPLETED"
            or row_commit.get("row_id") != row_id
            or row_commit.get("row_index") != row_index
            or row_commit.get("attempt") not in {0, 1}
            or row_commit.get("record_count") != expected_count
            or row_commit.get("records_sha256") != records_sha
            or row_commit.get("case_set_sha256") != case_set_sha
            or row_commit.get("deadline_commitment_sha256")
            != deadline_commitment
            or HEX_SHA256.fullmatch(str(row_commit.get("commitment_sha256", ""))) is None
            or not isinstance(identity, dict)
        ):
            raise ValueError(f"private row commitment mismatch: {row_id}")
        elapsed = finite(
            row_commit.get("elapsed_seconds"), f"row commitment elapsed: {row_id}"
        )
        if elapsed <= 0 or elapsed < sum(float(item["seconds"]) for item in members):
            raise ValueError(f"private row commitment timing is invalid: {row_id}")
        validate_row_memory_sentinel(row_commit.get("row_memory_sentinel"))
        reconstructed = {**row_commit, "records": members}
        committed_sha = reconstructed.pop("commitment_sha256")
        if committed_sha != hashlib.sha256(
            canonical_commitment_bytes(reconstructed)
        ).hexdigest():
            raise ValueError(f"private row commitment hash mismatch: {row_id}")
        normalized_identity = {str(key): str(value) for key, value in identity.items()}
        if row_identity is None:
            row_identity = normalized_identity
        elif normalized_identity != row_identity:
            raise ValueError("private row commitments have different evaluation identities")
    if row_identity is None:
        raise ValueError("private row commitment identity is missing")
    if run_deadline.get("identity") != row_identity:
        raise ValueError("persistent deadline and row identity differ")
    sha_fields = (
        "archive_sha256",
        "model_validation_sha256",
        "runtime_entry_sha256",
        "source_digest",
        "workload_policy_sha256",
        "memory_policy_sha256",
        "quality_controls_sha256",
        "hidden_manifest_sha256",
        "row_order_sha256",
        *RUNTIME_ATTESTATION_HASH_FIELDS,
    )
    if (
        not row_identity.get("run_id")
        or any(HEX_SHA256.fullmatch(row_identity.get(field, "")) is None for field in sha_fields)
        or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", row_identity.get(field, "")) is None
            for field in ("work_image_id", "judge_image_id")
        )
        or row_identity["workload_policy_sha256"] != sha256(workload_policy_path)
        or row_identity["memory_policy_sha256"] != sha256(memory_policy_path)
        or private.get("quality_controls_sha256")
        != row_identity["quality_controls_sha256"]
        or row_identity["hidden_manifest_sha256"] != hidden_sha
        or row_identity["row_order_sha256"]
        != hashlib.sha256(("\n".join(EXPECTED_ROWS) + "\n").encode()).hexdigest()
    ):
        raise ValueError("private row commitment identity tuple is invalid")
    throughputs: list[float] = []
    for row_id, count in EXPECTED_ROWS.items():
        row = rows[row_id]
        if not isinstance(row, dict) or row.get("cases") != count:
            raise ValueError(f"private row count mismatch: {row_id}")
        aggregate = aggregates[row_id]
        for field in ("correct", "parsed", "useful_tokens"):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != aggregate[field]
            ):
                raise ValueError(f"private row {field} aggregate mismatch: {row_id}")
        row_seconds = finite(row.get("seconds"), f"private row seconds: {row_id}")
        if row_seconds <= 0 or not math.isclose(
            row_seconds,
            float(aggregate["seconds"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"private row seconds aggregate mismatch: {row_id}")
        expected_throughput = int(aggregate["useful_tokens"]) / row_seconds
        throughput = finite(
            row.get("throughput_tok_s"), f"private row throughput: {row_id}"
        )
        if throughput <= 0 or not math.isclose(
            throughput, expected_throughput, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"private row throughput mismatch: {row_id}")
        throughputs.append(throughput)
    recomputed_geomean = math.exp(
        sum(math.log(value) for value in throughputs) / len(throughputs)
    )
    reward = finite(private.get("reward"), "private reward")
    geomean = finite(
        private.get("geomean_throughput_tok_s"), "private geomean throughput"
    )
    if reward <= 0 or any(
        not math.isclose(value, recomputed_geomean, rel_tol=1e-12, abs_tol=1e-12)
        for value in (reward, geomean)
    ):
        raise ValueError("private score views do not match")
    return reward, row_identity


def validate_reference_replay(
    reference: dict[str, Any], mode: str, archive_sha256: str
) -> None:
    if mode != "final" or reference.get("calibrated") is not True:
        raise ValueError("reference replay requires final mode and a calibrated reference")
    provenance = reference.get("calibration_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("candidate_archive_sha256") != archive_sha256
    ):
        raise ValueError(
            "reference replay archive does not match frozen calibration provenance"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--input-archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-archive-size", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("qualification", "calibration", "quality_control", "final"),
        required=True,
    )
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--model-validation", type=Path, required=True)
    parser.add_argument("--runtime-provenance", type=Path, required=True)
    parser.add_argument("--quality-controls-manifest", type=Path)
    parser.add_argument("--memory-fixture-manifest", type=Path)
    parser.add_argument("--reference-replay", action="store_true")
    parser.add_argument(
        "--judge-gpu-uuid", default=os.environ.get("LLADA_JUDGE_GPU_UUID")
    )
    parser.add_argument(
        "--memory-fixture-kind", choices=("corrected_reference", "near_cap_stress")
    )
    parser.add_argument("--nvml-total-bytes", type=int)
    parser.add_argument("--nvml-clean-idle-free-bytes", type=int)
    parser.add_argument("--torch-device-total-bytes", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()

    if HEX_SHA256.fullmatch(args.expected_archive_sha256) is None:
        raise SystemExit("expected archive SHA-256 is invalid")
    if args.expected_archive_size <= 0:
        raise SystemExit("expected archive size is invalid")
    root = args.task_root.resolve()
    workload_policy = object_at(root / "workload_policy.json")
    validate_workload_policy(workload_policy, require_frozen=args.mode == "final")
    if args.mode in {"calibration", "quality_control"} and workload_policy.get("release_status") != "PROVISIONAL":
        raise SystemExit("non-release producer evaluation requires provisional workload policy")
    if args.mode == "qualification" and (
        workload_policy.get("qualification_status") != "UNQUALIFIED"
        or workload_policy.get("release_status") != "PENDING"
    ):
        raise SystemExit("qualification producer requires the initial unqualified workload policy")
    memory_policy = object_at(root / "memory_policy.json")
    validate_memory_policy(memory_policy, require_frozen=args.mode != "qualification")
    if args.mode != "calibration" and args.mode != "qualification":
        frozen_reference = object_at(
            private_data_dir(root) / "reference_quality.json"
        )
        if (
            frozen_reference.get("schema_version") != 4
            or frozen_reference.get("calibrated") is not True
            or frozen_reference.get("status") != "CALIBRATED_STRATIFIED520"
            or frozen_reference.get("task_id") != TASK_ID
            or frozen_reference.get("workload_profile") != WORKLOAD_PROFILE
            or frozen_reference.get("generation_count") != GENERATION_COUNT
            or frozen_reference.get("requested_slots") != REQUESTED_SLOTS
        ):
            raise SystemExit("quality/final evaluation requires the calibrated H100 reference")
    submission = args.submission_root.resolve()
    if args.submission_root.is_symlink() or not submission.is_dir():
        raise SystemExit("submission root is missing or unsafe")
    if (
        re.fullmatch(r"manual-[1-9][0-9]*", submission.name) is None
        or submission.parent.name != "submissions"
        or submission.parent.parent.name != TASK_ID
    ):
        raise SystemExit("submission root has an unexpected run layout")
    run_root = submission.parents[2]
    if run_root.name != args.run_id:
        raise SystemExit("run directory identity does not match --run-id")
    if args.image_manifest.resolve().parent != run_root:
        raise SystemExit("image manifest is not adjacent to the evaluated run")
    if args.model_validation.resolve() != run_root / "full-model-validation.json":
        raise SystemExit("model attestation is not adjacent to the evaluated run")
    if args.output.resolve() != run_root / "evaluation-attestation.json":
        raise SystemExit("evaluation attestation output is not adjacent to the evaluated run")
    try:
        runtime_snapshot, runtime_snapshot_sha256 = validate_runtime_snapshot(
            args.runtime_provenance, run_root=run_root, task_root=root
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    runtime_hashes = runtime_identity(runtime_snapshot, runtime_snapshot_sha256)
    controls_manifest: dict[str, Any] | None = None
    controls_manifest_sha256: str | None = None
    if args.mode in {"calibration", "quality_control"}:
        if args.quality_controls_manifest is None:
            raise SystemExit("producer/control evaluation requires a run-local controls snapshot")
        controls_path = args.quality_controls_manifest.resolve()
        if (
            args.quality_controls_manifest.is_symlink()
            or not controls_path.is_file()
            or controls_path.parent != run_root
            or controls_path.stat().st_mode & 0o222
        ):
            raise SystemExit("quality-control snapshot is not immutable and run-local")
        controls_manifest = object_at(controls_path)
        controls_manifest_sha256 = sha256(controls_path)
        if (
            controls_manifest.get("status") != "PRECOMMITTED"
            or controls_manifest.get("task_id") != TASK_ID
            or controls_manifest.get("profile") != WORKLOAD_PROFILE
            or controls_manifest.get("workload_policy_sha256")
            != sha256(root / "workload_policy.json")
            or controls_manifest.get("hidden_cases_sha256")
            != workload_policy["manifest"]["hidden_cases_sha256"]
        ):
            raise SystemExit(
                "quality-control snapshot is not bound to the provisional policy"
            )
        if args.mode == "calibration":
            producer = controls_manifest.get("producer_calibration")
            if (
                not isinstance(producer, dict)
                or producer.get("expected_mode") != "calibration"
                or producer.get("archive_sha256") != args.expected_archive_sha256
            ):
                raise SystemExit(
                    "calibration run does not match its precommitted producer archive"
                )
    input_archive = args.input_archive
    persisted_archive = submission / "submission.tar.gz"
    report_path = submission / "report.json"
    private_path = submission / "private_judge_artifacts.tar.gz"
    for path, label in (
        (input_archive, "input archive"),
        (persisted_archive, "persisted archive"),
    ):
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"{label} is missing or unsafe")
        if (
            path.stat().st_size != args.expected_archive_size
            or sha256(path) != args.expected_archive_sha256
        ):
            raise SystemExit(f"{label} changed or differs from the pre-evaluation bytes")
    if input_archive.read_bytes() != persisted_archive.read_bytes():
        raise SystemExit("input and SForge-persisted archives are not byte-identical")

    report = object_at(report_path)
    payload = exact_private_payload(private_path)
    private = json.loads(payload)
    if not isinstance(private, dict):
        raise SystemExit("private Judge payload is not an object")
    if (
        report.get("task_id") != TASK_ID
        or report.get("submission_id") != args.run_id
    ):
        raise SystemExit("SForge report identity is invalid")
    timed_out = report.get("timed_out")
    errors = report.get("errors")
    if (
        not isinstance(timed_out, bool)
        or isinstance(errors, bool)
        or not isinstance(errors, int)
        or errors < 0
    ):
        raise SystemExit("SForge report acceptance fields are invalid")
    if report.get("valid") is not True or timed_out or errors != 0:
        private_error = private.get("error")
        if isinstance(private_error, str) and private_error.strip():
            # The artifact is produced by the root-owned Judge, but candidate
            # exceptions can contribute text to this field. Keep diagnostics
            # single-line and bounded before returning them to the host.
            diagnostic = " ".join(
                "".join(
                    character if character.isprintable() else " "
                    for character in private_error
                ).split()
            )[:500]
            if diagnostic:
                raise SystemExit(f"Judge rejected result: {diagnostic}")
        raise SystemExit("SForge report was rejected without a private failure reason")
    try:
        validate_private_runtime(
            private, runtime_snapshot, runtime_snapshot_sha256
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.mode == "quality_control":
        assert controls_manifest is not None
        controls = {
            item.get("id"): item
            for group in (
                controls_manifest.get("positive_controls"),
                controls_manifest.get("negative_controls"),
            )
            if isinstance(group, list)
            for item in group
            if isinstance(item, dict)
        }
        control = controls.get(private.get("control_id"))
        if (
            not isinstance(control, dict)
            or control.get("expected_classification")
            != private.get("expected_classification")
            or control.get("archive_sha256") != args.expected_archive_sha256
        ):
            raise SystemExit("quality-control run does not match its precommitted archive")
    score = finite(report.get("score"), "SForge report score")
    metrics = report.get("metrics")
    if score <= 0 or not isinstance(metrics, dict):
        raise SystemExit("SForge report has no positive accepted score")
    sforge_wall_seconds = finite(
        report.get("runtime_seconds"), "SForge report runtime_seconds"
    )
    if sforge_wall_seconds <= 0:
        raise SystemExit("SForge report wall runtime is invalid")
    reference_path = selected_reference_path(private_data_dir(root), private)
    private_score, row_identity = validate_private(
        private,
        mode=args.mode,
        hidden_path=root / "images/judge/private_data/hidden_cases.json",
        reference_path=reference_path,
        quality_path=root / "images/judge/verifier/quality.py",
        workload_policy_path=root / "workload_policy.json",
        memory_policy_path=root / "memory_policy.json",
    )
    quality_measurement = None
    if args.mode in {"quality_control", "final"}:
        reference_rows = frozen_reference.get("rows")
        if not isinstance(reference_rows, dict):
            raise SystemExit("calibrated reference rows are missing")
        quality_measurement = measure_quality_retention(
            private["rows"], reference_rows
        )
        validate_measurement(private.get("quality_measurement"))
        if private["quality_measurement"] != quality_measurement:
            raise SystemExit("private quality measurement does not match raw row metrics")
        benchmark_ppm = benchmark_retention(private["rows"], reference_rows)
        gate = private.get("quality_gate")
        if (
            not isinstance(gate, dict)
            or gate.get("gate") != QUALITY_GATE
            or gate.get("aggregate_floor_ppm") != AGGREGATE_FLOOR_PPM
            or gate.get("benchmark_floor_ppm") != BENCHMARK_FLOOR_PPM
            or gate.get("max_useful_tokens_retention_ppm")
            != MAX_USEFUL_TOKENS_RETENTION_PPM
            or gate.get("benchmark_components_ppm") != benchmark_ppm
        ):
            raise SystemExit("private quality gate does not match raw row metrics")
        fixed_failures = fixed_gate_failures(quality_measurement, benchmark_ppm)
        if any(item not in private.get("quality_failures", []) for item in fixed_failures):
            raise SystemExit("private quality failures omit a fixed-gate failure")
        if fixed_failures and private.get("quality_ok") is not False:
            raise SystemExit("private quality verdict ignores a fixed-gate failure")
    metrics_reward = finite(metrics.get("reward"), "SForge metrics reward")
    if not all(
        math.isclose(value, score, rel_tol=1e-12, abs_tol=1e-12)
        for value in (private_score, metrics_reward)
    ):
        raise SystemExit("SForge and private score views differ")

    image_manifest = object_at(args.image_manifest)
    model_validation = object_at(args.model_validation)
    validate_image_manifest(image_manifest)
    validate_model_attestation(model_validation)
    judge_gpu_uuid = args.judge_gpu_uuid
    if re.fullmatch(r"GPU-[0-9A-Fa-f-]+", str(judge_gpu_uuid or "")) is None:
        raise SystemExit("Judge GPU UUID provenance is missing or invalid")
    expected_identity = {
        "run_id": args.run_id,
        "archive_sha256": args.expected_archive_sha256,
        "work_image_id": str(image_manifest["work"]["id"]),
        "judge_image_id": str(image_manifest["judge"]["id"]),
        "model_validation_sha256": sha256(args.model_validation),
        "runtime_entry_sha256": sha256(root / "scripts/sforge_h100_entry.py"),
        "source_digest": str(image_manifest["source_digest"]),
        "workload_policy_sha256": sha256(root / "workload_policy.json"),
        "memory_policy_sha256": sha256(root / "memory_policy.json"),
        "quality_controls_sha256": (
            controls_manifest_sha256
            if controls_manifest_sha256 is not None
            else sha256(root / "quality_controls.json")
        ),
        "hidden_manifest_sha256": canonical_hidden_sha256(root),
        "row_order_sha256": hashlib.sha256(
            ("\n".join(EXPECTED_ROWS) + "\n").encode()
        ).hexdigest(),
        **runtime_hashes,
    }
    if row_identity != expected_identity:
        raise SystemExit("row-atomic commitment identity differs from host provenance")
    reference = object_at(private_data_dir(root) / "reference_quality.json")
    if args.reference_replay:
        validate_reference_replay(
            reference, args.mode, args.expected_archive_sha256
        )

    attestation = {
        "schema_version": 1,
        "status": "PASS",
        "task_id": TASK_ID,
        "hardware": validate_device_envelope(private, "private report"),
        "visible_device_count": 1,
        "profile": WORKLOAD_PROFILE,
        "run_id": args.run_id,
        "submission_id": args.run_id,
        "round_id": submission.name,
        "mode": args.mode,
        "reference_replay": args.reference_replay,
        "release_evidence": args.mode == "final",
        "score": score,
        "archive_sha256": args.expected_archive_sha256,
        "archive_size_bytes": args.expected_archive_size,
        "sforge_report_sha256": sha256(report_path),
        "private_artifact_sha256": sha256(private_path),
        "private_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "image_manifest_sha256": sha256(args.image_manifest),
        "model_validation_sha256": sha256(args.model_validation),
        "reference_manifest_sha256": sha256(reference_path),
        "workload_profile": WORKLOAD_PROFILE,
        "generation_count": GENERATION_COUNT,
        "requested_slots": REQUESTED_SLOTS,
        "row_counts": EXPECTED_ROWS,
        "row_seconds": {
            str(commitment["row_id"]): float(commitment["elapsed_seconds"])
            for commitment in private["row_commitments"]
        },
        "elapsed_seconds": sforge_wall_seconds,
        "case_generation_seconds": sum(
            float(private["rows"][row_id]["seconds"])
            for row_id in EXPECTED_ROWS
        ),
        "max_case_seconds": max(
            float(record["seconds"]) for record in private["records"]
        ),
        "hidden_cases_sha256": private["dataset_manifest_sha256"],
        "dataset_manifest_sha256": workload_policy["manifest"]["dataset_manifest_sha256"],
        "workload_policy_sha256": sha256(root / "workload_policy.json"),
        "memory_policy_sha256": sha256(root / "memory_policy.json"),
        "work_image_id": image_manifest["work"]["id"],
        "judge_image_id": image_manifest["judge"]["id"],
        "judge_gpu_uuid": judge_gpu_uuid,
        "runtime_entry_sha256": row_identity["runtime_entry_sha256"],
        "source_digest": row_identity["source_digest"],
        "control_id": private.get("control_id"),
        "expected_classification": private.get("expected_classification"),
        "observed_classification": private.get("observed_classification"),
        "quality_measurement": quality_measurement,
        "quality_score_ppm": (
            quality_measurement["score_ppm"]
            if quality_measurement is not None
            else None
        ),
        "validator_sha256": EXECUTING_VALIDATOR_SHA256,
        "runtime_provenance_sha256": runtime_snapshot_sha256,
        "sforge_runtime_sha256": runtime_snapshot["sforge_runtime"]["sha256"],
        "sforge_runtime_file_set_sha256": runtime_snapshot["sforge_runtime"][
            "file_set_sha256"
        ],
        "sforge_runtime": runtime_snapshot["sforge_runtime"],
        "judge_invariant_core": runtime_snapshot["judge_invariant_core"],
        **runtime_snapshot["judge_invariant_core"],
        "quality_controls_sha256": (
            controls_manifest_sha256
            if controls_manifest_sha256 is not None
            else sha256(root / "quality_controls.json")
        ),
        "row_sentinel_observed_max_growth_bytes": max(
            int(commitment["row_memory_sentinel"][
                "observed_max_positive_growth_bytes"
            ])
            for commitment in private["row_commitments"]
        ),
        "row_sentinel_observed_max_repeat_growth_bytes": max(
            int(commitment["row_memory_sentinel"][
                "observed_max_positive_repeat_growth_bytes"
            ])
            for commitment in private["row_commitments"]
        ),
        "row_sentinel_all_stable": all(
            commitment["row_memory_sentinel"].get("stable") is True
            for commitment in private["row_commitments"]
        ),
        "row_sentinel_repeat_all_stable": all(
            commitment["row_memory_sentinel"].get("repeat_stable") is True
            for commitment in private["row_commitments"]
        ),
    }
    committed_row_seconds = sum(attestation["row_seconds"].values())
    if sforge_wall_seconds < committed_row_seconds:
        raise SystemExit("SForge wall runtime is shorter than committed row runtimes")
    memory_args = (
        args.memory_fixture_kind,
        args.memory_fixture_manifest,
        args.nvml_total_bytes,
        args.nvml_clean_idle_free_bytes,
        args.torch_device_total_bytes,
    )
    if any(value is not None for value in memory_args):
        if args.mode != "qualification":
            raise SystemExit("memory qualification fixtures require qualification mode")
        if any(value is None for value in memory_args):
            raise SystemExit("memory qualification evidence requires all capacity fields")
        try:
            fixture_manifest, fixture_manifest_sha256, selected_fixture = (
                validate_memory_fixture_snapshot(
                    args.memory_fixture_manifest,
                    run_root=run_root,
                    task_root=root,
                    kind=args.memory_fixture_kind,
                    archive_sha256=args.expected_archive_sha256,
                    archive_size_bytes=args.expected_archive_size,
                )
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        expected_private_fixture = {
            "schema_version": 1,
            "manifest_sha256": fixture_manifest_sha256,
            "approval_id": fixture_manifest["approval_id"],
            "kind": args.memory_fixture_kind,
            "archive_sha256": args.expected_archive_sha256,
            "archive_size_bytes": args.expected_archive_size,
        }
        if private.get("memory_fixture") != expected_private_fixture:
            raise SystemExit("private memory fixture identity differs from its snapshot")
        peak_allocated = max(int(record["torch_peak_allocated_bytes"]) for record in private["records"])
        peak_reserved = max(int(record["torch_peak_reserved_bytes"]) for record in private["records"])
        peak_nvml = max(
            int(record["nvml_peak_device_used_bytes"])
            for record in private["records"]
        )
        if not (
            0 < peak_allocated <= peak_reserved <= peak_nvml < int(args.nvml_total_bytes)
            and 0 < int(args.nvml_clean_idle_free_bytes) <= int(args.nvml_total_bytes)
            and peak_allocated < int(args.torch_device_total_bytes) <= int(args.nvml_total_bytes)
        ):
            raise SystemExit("memory qualification capacity/peak evidence is inconsistent")
        attestation.update(
            {
                "fixture_kind": args.memory_fixture_kind,
                "torch_peak_allocated_bytes": peak_allocated,
                "torch_peak_reserved_bytes": peak_reserved,
                "nvml_peak_device_used_bytes": peak_nvml,
                "nvml_total_bytes": args.nvml_total_bytes,
                "nvml_clean_idle_free_bytes": args.nvml_clean_idle_free_bytes,
                "torch_device_total_bytes": args.torch_device_total_bytes,
                "completed_without_oom": True,
                "memory_fixture_manifest_sha256": fixture_manifest_sha256,
                "memory_fixture_approval_id": fixture_manifest["approval_id"],
                "memory_fixture": expected_private_fixture,
                "memory_fixture_precommit": {
                    "schema_version": fixture_manifest["schema_version"],
                    "status": fixture_manifest["status"],
                    "task_id": fixture_manifest["task_id"],
                    "profile": fixture_manifest["profile"],
                    "workload_policy_sha256": fixture_manifest[
                        "workload_policy_sha256"
                    ],
                    "memory_policy_sha256": fixture_manifest[
                        "memory_policy_sha256"
                    ],
                    "approval_id": fixture_manifest["approval_id"],
                    "selected_fixture": selected_fixture,
                },
            }
        )
    elif private.get("memory_fixture") is not None:
        raise SystemExit("private memory fixture evidence is present outside memory qualification")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(attestation, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.chmod(0o444)
    os.replace(temporary, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
