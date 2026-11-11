#!/usr/bin/env python3
"""Freeze a paired quality-diagnostic baseline from a corrected private report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h100_policy_contract import (
    validate_device_envelope,
    GENERATION_COUNT,
    PROFILE,
    REQUESTED_SLOTS,
    RUNTIME_ATTESTATION_HASH_FIELDS,
    ROW_COUNTS,
    is_sha256,
    validate_attested_runtime,
    validate_memory_policy,
    validate_workload_policy,
)


EXPECTED_ROWS = ROW_COUNTS
TASK_ID = "llada_h100_decoder_optimization_v2"
WORKLOAD_PROFILE = PROFILE
PRIVATE_ARTIFACT_MAX_BYTES = 64 * 1024 * 1024
REFERENCE_CACHE_DIRNAME = "reference_cache"


def profile_key(profile: dict) -> str:
    """Hash one measurement/device profile exactly as the Judge does."""
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Bind the bytes that initialized this process, rather than re-reading a tool
# that could have changed after Python loaded it.
EXECUTING_CALIBRATION_TOOL_SHA256 = sha256(Path(__file__).resolve())


def object_at(path: Path, label: str) -> tuple[bytes, dict]:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} is missing or unsafe")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise SystemExit(f"{label} is not a JSON object")
    return raw, value


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise SystemExit(f"{label} is not finite")
    return number


def private_payload(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise SystemExit("private Judge artifact is missing or unsafe")
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
                raise SystemExit(
                    "private Judge artifact must contain exactly one bounded regular decoder_private.json"
                )
            stream = bundle.extractfile(members[0])
            if stream is None:
                raise SystemExit("private Judge payload is unreadable")
            payload = stream.read(PRIVATE_ARTIFACT_MAX_BYTES + 1)
            if len(payload) != members[0].size:
                raise SystemExit("private Judge payload is truncated")
            return payload
    except (OSError, tarfile.TarError) as exc:
        raise SystemExit("private Judge artifact is unreadable") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_root", type=Path)
    parser.add_argument("--producer-image-manifest", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument(
        "--canonical-candidate-archive",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts/corrected-decoder-reference.tar.gz",
    )
    parser.add_argument("--model-validation", type=Path, required=True)
    parser.add_argument("--evaluation-attestation", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "images/judge/private_data/reference_quality.json",
    )
    args = parser.parse_args()
    task_root = Path(__file__).resolve().parents[1]
    workload_policy_path = task_root / "workload_policy.json"
    memory_policy_path = task_root / "memory_policy.json"
    quality_controls_path = task_root / "quality_controls.json"
    workload_policy = object_at(workload_policy_path, "workload policy")[1]
    validate_workload_policy(workload_policy)
    if (
        workload_policy.get("qualification_status") != "QUALIFIED"
        or workload_policy.get("release_status") != "PROVISIONAL"
    ):
        raise SystemExit("producer calibration requires provisional qualified workload policy")
    validate_memory_policy(
        object_at(memory_policy_path, "memory policy")[1], require_frozen=True
    )
    submission_root = args.submission_root.resolve()
    if args.submission_root.is_symlink() or not submission_root.is_dir():
        raise SystemExit("trusted submission root is missing or unsafe")
    if (
        re.fullmatch(r"manual-[1-9][0-9]*", submission_root.name) is None
        or submission_root.parent.name != "submissions"
        or submission_root.parent.parent.name
        != TASK_ID
    ):
        raise SystemExit("trusted submission root has an unexpected run layout")
    run_root = submission_root.parents[2]
    if args.evaluation_attestation.resolve() != run_root / "evaluation-attestation.json":
        raise SystemExit("evaluation attestation is not adjacent to the trusted run")
    if args.model_validation.resolve() != run_root / "full-model-validation.json":
        raise SystemExit("model attestation is not adjacent to the trusted run")
    if (
        args.producer_image_manifest.resolve().parent != run_root
        or args.producer_image_manifest.name
        not in {"evaluation-image-manifest.json", "calibration-image-manifest.json"}
    ):
        raise SystemExit("producer image manifest is not adjacent to the trusted run")
    candidate_archive = submission_root / "submission.tar.gz"
    private_artifact = submission_root / "private_judge_artifacts.tar.gz"
    sforge_report_path = submission_root / "report.json"
    raw = private_payload(private_artifact)
    report = json.loads(raw)
    if not isinstance(report, dict):
        raise SystemExit("private Judge report is not an object")
    sforge_report_raw, sforge_report = object_at(sforge_report_path, "SForge report")
    image_manifest_raw, image_manifest = object_at(
        args.producer_image_manifest, "producer image manifest"
    )
    model_validation_raw, model_validation = object_at(
        args.model_validation, "model validation"
    )
    evaluation_attestation_raw, evaluation_attestation = object_at(
        args.evaluation_attestation, "evaluation attestation"
    )
    try:
        runtime_provenance = validate_attested_runtime(evaluation_attestation)
    except ValueError as exc:
        raise SystemExit("evaluation attestation runtime provenance is invalid") from exc
    hex_sha = re.compile(r"[0-9a-f]{64}")
    image_id = re.compile(r"sha256:[0-9a-f]{64}")
    calibration_files = (
        image_manifest.get("calibration_files")
        if isinstance(image_manifest, dict)
        else None
    )
    if (
        not isinstance(image_manifest, dict)
        or hex_sha.fullmatch(str(image_manifest.get("source_digest", ""))) is None
        or hex_sha.fullmatch(
            str(image_manifest.get("judge_runtime_tree_sha256", ""))
        )
        is None
        or not isinstance(image_manifest.get("work"), dict)
        or not isinstance(image_manifest.get("judge"), dict)
        or image_id.fullmatch(str(image_manifest["work"].get("id", ""))) is None
        or image_id.fullmatch(str(image_manifest["judge"].get("id", ""))) is None
        or not isinstance(calibration_files, dict)
        or any(
            hex_sha.fullmatch(str(calibration_files.get(key, ""))) is None
            for key in (
                "hidden_cases_sha256",
                "quality_runtime_sha256",
                "verifier_sha256",
                "candidate_worker_sha256",
                "cohort_sha256",
                "workload_policy_sha256",
                "memory_policy_sha256",
            )
        )
    ):
        raise SystemExit("producer image manifest is invalid")
    if (
        not isinstance(model_validation, dict)
        or model_validation.get("status") != "PASS"
        or model_validation.get("full") is not True
        or model_validation.get("hashes_checked") != 16
        or model_validation.get("logical_size_bytes") != 16037449215
        or model_validation.get("revision")
        != "6059b30c9531767ce47e0ca6a8161a6bb0c67693"
    ):
        raise SystemExit("full model validation provenance is invalid")
    if not candidate_archive.is_file() or candidate_archive.is_symlink():
        raise SystemExit("candidate archive provenance is missing or invalid")
    if (
        not args.canonical_candidate_archive.is_file()
        or args.canonical_candidate_archive.is_symlink()
        or candidate_archive.read_bytes()
        != args.canonical_candidate_archive.read_bytes()
    ):
        raise SystemExit(
            "SForge-persisted archive is not byte-identical to the canonical corrected archive"
        )
    if not args.candidate_source.is_file() or args.candidate_source.is_symlink():
        raise SystemExit("candidate source provenance is missing or invalid")
    candidate_source_bytes = args.candidate_source.read_bytes()
    try:
        with tarfile.open(candidate_archive, "r:gz") as bundle:
            members = bundle.getmembers()
            expected_members = {
                "FINAL_VERSION",
                "notes.md",
                "solutions/decoder_v001/__init__.py",
                "solutions/decoder_v001/candidate.py",
            }
            if any(
                member.issym()
                or member.islnk()
                or member.name.startswith("/")
                or ".." in Path(member.name).parts
                for member in members
            ):
                raise SystemExit("candidate archive contains an unsafe member")
            if (
                len(members) != len(expected_members)
                or {member.name for member in members} != expected_members
                or any(not member.isfile() for member in members)
            ):
                raise SystemExit("candidate archive member set is not canonical")
            final_member = next(
                member for member in members if member.name == "FINAL_VERSION"
            )
            final_stream = bundle.extractfile(final_member)
            if final_stream is None or final_stream.read() != b"1\n":
                raise SystemExit("candidate archive does not select corrected v001")
            candidate_members = [
                member
                for member in members
                if member.isfile()
                and member.name == "solutions/decoder_v001/candidate.py"
            ]
            if len(candidate_members) != 1:
                raise SystemExit("candidate archive has no unique corrected source")
            candidate_stream = bundle.extractfile(candidate_members[0])
            if candidate_stream is None or candidate_stream.read() != candidate_source_bytes:
                raise SystemExit("candidate archive/source provenance mismatch")
    except (OSError, tarfile.TarError) as exc:
        raise SystemExit("candidate archive is unreadable") from exc
    run_id = evaluation_attestation.get("run_id")
    submission_id = evaluation_attestation.get("submission_id")
    validator_path = Path(__file__).resolve().with_name("validate_eval_artifacts.py")
    validator_sha256 = sha256(validator_path)
    calibration_tool_sha256 = EXECUTING_CALIBRATION_TOOL_SHA256
    archive_sha256 = sha256(candidate_archive)
    private_artifact_sha256 = sha256(private_artifact)
    sforge_report_sha256 = hashlib.sha256(sforge_report_raw).hexdigest()
    private_payload_sha256 = hashlib.sha256(raw).hexdigest()
    image_manifest_sha256 = hashlib.sha256(image_manifest_raw).hexdigest()
    model_validation_sha256 = hashlib.sha256(model_validation_raw).hexdigest()
    producer_reference_sha256 = sha256(
        Path(__file__).resolve().parents[1]
        / "images/judge/private_data/reference_quality.json"
    )
    if (
        evaluation_attestation.get("schema_version") != 1
        or evaluation_attestation.get("status") != "PASS"
        or evaluation_attestation.get("task_id")
        != TASK_ID
        or not isinstance(run_id, str)
        or not run_id
        or run_root.name != run_id
        or submission_id != run_id
        or evaluation_attestation.get("round_id") != submission_root.name
        or re.fullmatch(r"manual-[1-9][0-9]*", submission_root.name) is None
        or evaluation_attestation.get("mode") != "calibration"
        or evaluation_attestation.get("reference_replay") is not False
        or evaluation_attestation.get("archive_sha256") != archive_sha256
        or evaluation_attestation.get("archive_size_bytes")
        != candidate_archive.stat().st_size
        or evaluation_attestation.get("sforge_report_sha256")
        != sforge_report_sha256
        or evaluation_attestation.get("private_artifact_sha256")
        != private_artifact_sha256
        or evaluation_attestation.get("private_payload_sha256")
        != private_payload_sha256
        or evaluation_attestation.get("image_manifest_sha256")
        != image_manifest_sha256
        or evaluation_attestation.get("model_validation_sha256")
        != model_validation_sha256
        or evaluation_attestation.get("reference_manifest_sha256")
        != producer_reference_sha256
        or evaluation_attestation.get("validator_sha256") != validator_sha256
        or evaluation_attestation.get("quality_controls_sha256")
        != sha256(quality_controls_path)
        or any(
            not is_sha256(evaluation_attestation.get(field))
            for field in RUNTIME_ATTESTATION_HASH_FIELDS
        )
        or report.get("reference_manifest_sha256") != producer_reference_sha256
    ):
        raise SystemExit("evaluation attestation does not bind the calibration artifacts")
    if (
        sforge_report.get("task_id") != TASK_ID
        or sforge_report.get("submission_id") != run_id
        or sforge_report.get("valid") is not True
        or sforge_report.get("timed_out") is not False
        or isinstance(sforge_report.get("errors"), bool)
        or not isinstance(sforge_report.get("errors"), int)
        or sforge_report.get("errors") != 0
    ):
        raise SystemExit("SForge calibration report is not an accepted matching evaluation")
    sforge_score = finite(sforge_report.get("score"), "SForge calibration score")
    attested_score = finite(
        evaluation_attestation.get("score"), "attested calibration score"
    )
    private_score = finite(report.get("reward"), "private calibration score")
    private_geomean = finite(
        report.get("geomean_throughput_tok_s"), "private calibration geomean"
    )
    sforge_metrics = sforge_report.get("metrics")
    if not isinstance(sforge_metrics, dict):
        raise SystemExit("SForge calibration metrics are missing")
    metrics_score = finite(sforge_metrics.get("reward"), "SForge calibration reward")
    if sforge_score <= 0 or not all(
        math.isclose(value, sforge_score, rel_tol=1e-12, abs_tol=1e-12)
        for value in (attested_score, private_score, private_geomean, metrics_score)
    ):
        raise SystemExit("calibration score views do not agree")
    if report.get("calibration") is not True:
        raise SystemExit("input is not a calibration report")
    if report.get("schema_version") != 3:
        raise SystemExit("calibration report schema is not version 3")
    if report.get("isolation") != "root-controller/gold-free-agent-worker":
        raise SystemExit("calibration did not use the isolated worker protocol")
    if not all(report.get(key) is True for key in ("source_ok", "integrity_ok", "performance_ok")):
        raise SystemExit("calibration report did not pass all hard gates")
    if set(report.get("rows", {})) != set(EXPECTED_ROWS):
        raise SystemExit("calibration report has the wrong expected row set")
    quality_runtime_sha256 = report.get("quality_runtime_sha256")
    if not isinstance(quality_runtime_sha256, str) or len(quality_runtime_sha256) != 64:
        raise SystemExit("calibration report has no quality-runtime provenance")
    if (
        calibration_files["hidden_cases_sha256"]
        != report.get("dataset_manifest_sha256")
        or calibration_files["quality_runtime_sha256"] != quality_runtime_sha256
        or calibration_files["verifier_sha256"]
        != runtime_provenance["judge_invariant_core"]["verify_decoder_sha256"]
        or calibration_files["candidate_worker_sha256"]
        != runtime_provenance["judge_invariant_core"][
            "candidate_worker_sha256"
        ]
        or calibration_files["cohort_sha256"]
        != runtime_provenance["judge_invariant_core"]["cohort_sha256"]
        or calibration_files["quality_runtime_sha256"]
        != runtime_provenance["judge_invariant_core"]["quality_sha256"]
        or calibration_files["workload_policy_sha256"] != sha256(workload_policy_path)
        or calibration_files["memory_policy_sha256"] != sha256(memory_policy_path)
        or report.get("workload_policy_sha256") != sha256(workload_policy_path)
        or report.get("memory_policy_sha256") != sha256(memory_policy_path)
        or report.get("quality_controls_sha256") != sha256(quality_controls_path)
        or report.get("workload_profile") != WORKLOAD_PROFILE
        or report.get("generation_count") != sum(EXPECTED_ROWS.values())
        or report.get("requested_slots") != REQUESTED_SLOTS
    ):
        raise SystemExit("producer image contents do not match the calibration report")
    rows = {}
    for row_id, row in sorted(report["rows"].items()):
        cases = int(row["cases"])
        correct = int(row["correct"])
        parsed = int(row["parsed"])
        useful_tokens = int(row["useful_tokens"])
        if cases != EXPECTED_ROWS[row_id]:
            raise SystemExit(f"wrong case count for {row_id}")
        if not (0 <= correct <= parsed <= cases) or parsed <= 0:
            raise SystemExit(f"invalid quality reference for {row_id}")
        if useful_tokens <= 0:
            raise SystemExit(f"zero useful-token reference for {row_id}")
        rows[row_id] = {
            "cases": cases,
            "correct": correct,
            "parsed": parsed,
            "useful_tokens": useful_tokens,
        }
    raw_records = report.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != sum(EXPECTED_ROWS.values()):
        raise SystemExit("calibration report has the wrong record count")
    case_records = {}
    observed_row_counts = {row_id: 0 for row_id in EXPECTED_ROWS}
    for record in raw_records:
        if not isinstance(record, dict):
            raise SystemExit("calibration report contains a malformed record")
        case_sha256 = record.get("case_sha256")
        row_id = record.get("row_id")
        correct = record.get("correct")
        parsed = record.get("parsed")
        useful_tokens = record.get("useful_tokens")
        if (
            not isinstance(case_sha256, str)
            or len(case_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in case_sha256)
            or case_sha256 in case_records
        ):
            raise SystemExit("calibration report has invalid or duplicate case provenance")
        if row_id not in EXPECTED_ROWS:
            raise SystemExit("calibration record has an unknown row")
        if not isinstance(correct, bool) or not isinstance(parsed, bool):
            raise SystemExit("calibration record has invalid quality flags")
        if isinstance(useful_tokens, bool) or not isinstance(useful_tokens, int) or useful_tokens < 0:
            raise SystemExit("calibration record has invalid useful-token count")
        observed_row_counts[row_id] += 1
        case_records[case_sha256] = {
            "row_id": row_id,
            "correct": correct,
            "parsed": parsed,
            "useful_tokens": useful_tokens,
        }
    if observed_row_counts != EXPECTED_ROWS:
        raise SystemExit("calibration record rows do not match the frozen workload")
    for row_id, frozen_row in rows.items():
        members = [item for item in case_records.values() if item["row_id"] == row_id]
        if (
            sum(int(item["correct"]) for item in members) != frozen_row["correct"]
            or sum(int(item["parsed"]) for item in members) != frozen_row["parsed"]
            or sum(int(item["useful_tokens"]) for item in members)
            != frozen_row["useful_tokens"]
        ):
            raise SystemExit(f"record aggregates do not match row {row_id}")
    # Record the full measurement profile the producer ran under (device,
    # driver, producer images) as provenance.  The Judge selects and gates the
    # reference on GPU model + compute capability only (plus the software ABI
    # pinned in the verifier); driver and image ids never refuse it.
    inventory = report.get("hardware")
    if not isinstance(inventory, dict):
        raise SystemExit(
            "calibration report has no device inventory; re-run the producer on "
            "a Judge that records one"
        )
    hardware_name = validate_device_envelope(report, "calibration report")
    measurement_profile = {
        "gpu_name": hardware_name,
        "compute_capability": str(inventory.get("compute_capability", "")),
        "driver_version": str(inventory.get("driver_version", "")),
        "work_image_id": str(image_manifest["work"]["id"]),
        "judge_image_id": str(image_manifest["judge"]["id"]),
    }
    if any(not value for value in measurement_profile.values()):
        raise SystemExit("calibration report device/image profile is incomplete")
    device_profile = {
        key: measurement_profile[key]
        for key in ("compute_capability", "driver_version", "gpu_name")
    }
    output = {
        "schema_version": 4,
        "calibrated": True,
        "status": "CALIBRATED_STRATIFIED520",
        "task_id": TASK_ID,
        "workload_profile": WORKLOAD_PROFILE,
        "hardware": hardware_name,
        "measurement_profile": measurement_profile,
        "measurement_profile_key": profile_key(measurement_profile),
        "device_profile_key": profile_key(device_profile),
        "generation_count": GENERATION_COUNT,
        "requested_slots": REQUESTED_SLOTS,
        "reference": "corrected threshold + Prefix KV host submission",
        "source_private_report_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_manifest_sha256": report["dataset_manifest_sha256"],
        "quality_runtime_sha256": quality_runtime_sha256,
        "isolation": report["isolation"],
        "calibration_provenance": {
            "schema_version": 1,
            "task_id": TASK_ID,
            "workload_profile": WORKLOAD_PROFILE,
            "hardware": hardware_name,
            "candidate_archive_sha256": archive_sha256,
            "canonical_candidate_archive_sha256": sha256(
                args.canonical_candidate_archive
            ),
            "candidate_source_sha256": hashlib.sha256(
                candidate_source_bytes
            ).hexdigest(),
            "evaluation_validator_sha256": validator_sha256,
            "calibration_tool_sha256": calibration_tool_sha256,
            "image_manifest_sha256": image_manifest_sha256,
            "source_digest": image_manifest["source_digest"],
            "judge_runtime_tree_sha256": image_manifest[
                "judge_runtime_tree_sha256"
            ],
            "work_image": image_manifest["work"],
            "judge_image": image_manifest["judge"],
            "model_validation_sha256": model_validation_sha256,
            "model_revision": model_validation["revision"],
            "workload_policy_sha256": sha256(workload_policy_path),
            "memory_policy_sha256": sha256(memory_policy_path),
            "quality_controls_sha256": sha256(quality_controls_path),
            "runtime_provenance_sha256": evaluation_attestation[
                "runtime_provenance_sha256"
            ],
            "sforge_runtime_sha256": evaluation_attestation[
                "sforge_runtime_sha256"
            ],
            "sforge_runtime_file_set_sha256": evaluation_attestation[
                "sforge_runtime_file_set_sha256"
            ],
            "sforge_runtime": runtime_provenance["sforge_runtime"],
            "judge_invariant_core": runtime_provenance["judge_invariant_core"],
            **runtime_provenance["judge_invariant_core"],
            "evaluation_attestation_sha256": hashlib.sha256(
                evaluation_attestation_raw
            ).hexdigest(),
            "private_artifact_sha256": private_artifact_sha256,
            "sforge_report_sha256": sforge_report_sha256,
            "private_payload_sha256": private_payload_sha256,
            "run_id": run_id,
            "submission_id": submission_id,
        },
        "rows": rows,
        # Per-case baselines let an interactive cohort retain the same paired
        # quality diagnostic without revealing or approximating unseen cases.
        "case_records": dict(sorted(case_records.items())),
        "historical_a6000_reference": {
            "source_task_id": "llada_a6000_decoder_optimization_v3",
            "applicability": "NONE",
            "status": "UNCALIBRATED_FOR_H100_STRATIFIED520",
            "provenance_preserved_in_source_task": True,
        },
    }
    rendered = json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    # Cache the reference under its reference profile (GPU model + compute
    # capability) so a later run on the same GPU model can reuse it instead of
    # re-running the producer, and so a Judge image can carry references for
    # several GPU models at once.  A new producer run on the same GPU model
    # replaces the entry; it must reproduce the frozen numbers or be reviewed.
    cache_dir = args.output.parent / REFERENCE_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    reference_profile = {
        key: measurement_profile[key] for key in ("compute_capability", "gpu_name")
    }
    cached = cache_dir / f"{profile_key(reference_profile)}.json"
    cached.write_text(rendered, encoding="utf-8")
    print(args.output)
    print(cached)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
