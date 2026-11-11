#!/usr/bin/env python3
"""Atomically persist the fail-closed terminal state of a formal run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

from final_snapshot_contract import SnapshotContractError, validate_snapshot


ROOT = Path(__file__).resolve().parents[1]
# Keep this identical to the preflight's fixed repository-root binding.
EXPECTED_MLSYSFORGE_ROOT = str(ROOT.parents[2])
EXPECTED_MLSYSFORGE_GENERATED_EXCLUSIONS = [
    "directory-component:__pycache__",
    "file-suffix:.pyc",
]
EXPECTED_MLSYSFORGE_IMPORT_ORIGINS = {
    "sforge": "sforge/__init__.py",
    "sforge.cli": "sforge/cli.py",
    "sforge.harness.run_agent": "sforge/harness/run_agent.py",
    "sforge.harness.backend.docker_backend": (
        "sforge/harness/backend/docker_backend.py"
    ),
    "sforge.harness.judge_server": "sforge/harness/judge_server.py",
    "sforge.harness.run_evaluation": "sforge/harness/run_evaluation.py",
    "sforge.harness.task_spec": "sforge/harness/task_spec.py",
}
# A formal run that finished every stage but selected nothing (every hidden
# submission was authoritatively invalid) is a legitimate terminal outcome,
# distinct from an infrastructure failure. It still requires the same
# provenance, quiescence and trusted-source evidence as `passed`; only the
# selected-submission evidence is absent by construction.
NO_VALID_SUBMISSION_STATUS = "completed_no_valid_submission"
TERMINAL_STATUSES = frozenset({"passed", NO_VALID_SUBMISSION_STATUS})
HIDDEN_BASELINE_IDENTITY_FIELDS = frozenset(
    {
        "file_sha256",
        "valid_steps_per_min",
        "backend_serving_contract_sha256",
        "workload_nonce_sha256",
    }
)


def _empty_hidden_baseline_identity() -> dict[str, Any]:
    return {
        "file_sha256": None,
        "valid_steps_per_min": None,
        "backend_serving_contract_sha256": None,
        "workload_nonce_sha256": None,
    }


def _hidden_baseline_identity(
    value: object, label: str
) -> tuple[dict[str, Any], list[str]]:
    empty = _empty_hidden_baseline_identity()
    if not isinstance(value, dict) or set(value) != HIDDEN_BASELINE_IDENTITY_FIELDS:
        return empty, [f"{label}:schema"]
    file_sha256 = value.get("file_sha256")
    rate = _finite(value.get("valid_steps_per_min"))
    backend_sha256 = value.get("backend_serving_contract_sha256")
    nonce_sha256 = value.get("workload_nonce_sha256")
    errors: list[str] = []
    for name, digest in (
        ("file_sha256", file_sha256),
        ("backend_serving_contract_sha256", backend_sha256),
        ("workload_nonce_sha256", nonce_sha256),
    ):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append(f"{label}:{name}")
    if rate is None or rate <= 0.0:
        errors.append(f"{label}:valid_steps_per_min")
    if errors:
        return empty, errors
    return {
        "file_sha256": file_sha256,
        "valid_steps_per_min": rate,
        "backend_serving_contract_sha256": backend_sha256,
        "workload_nonce_sha256": nonce_sha256,
    }, []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "sha256": None, "size": None}
    record: dict[str, Any] = {
        "path": str(path.resolve(strict=False)),
        "exists": False,
        "sha256": None,
        "size": None,
    }
    try:
        if path.is_symlink() or not path.is_file():
            return record
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            return record
        record.update(exists=True, sha256=_sha256(path), size=info.st_size)
    except OSError:
        pass
    return record


def _load_object(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "path_not_set"
    record = _file_record(path)
    if not record["exists"]:
        return None, "file_missing_or_unsafe"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "invalid_json"
    if not isinstance(value, dict):
        return None, "root_not_object"
    return value, None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _validated_evaluation(
    path: Path | None,
    *,
    allowed_classifications: set[str],
) -> tuple[dict[str, Any], list[str]]:
    record = _file_record(path)
    payload, load_error = _load_object(path)
    errors: list[str] = []
    result: dict[str, Any] = {
        "validation": record,
        "hidden_baseline_identity": _empty_hidden_baseline_identity(),
        "classification": None,
        "authoritative": False,
        "submission_id": None,
        "eval_rc": None,
        "valid": None,
        "score": None,
        "report": {"path": None, "sha256": None},
        "private_artifact": {"path": None, "sha256": None},
        "evaluated_archive": {"path": None, "sha256": None},
        "expected_archive": {"path": None, "sha256": None},
    }
    if load_error is not None or payload is None:
        return result, [f"validation:{load_error}"]
    for key in result:
        if key != "validation" and key in payload:
            result[key] = payload[key]
    if payload.get("schema_version") != 1:
        errors.append("validation:schema_version")
    identity, identity_errors = _hidden_baseline_identity(
        payload.get("hidden_baseline_identity"), "validation:hidden_baseline_identity"
    )
    result["hidden_baseline_identity"] = identity
    errors.extend(identity_errors)
    if payload.get("classification") not in allowed_classifications:
        errors.append("validation:classification")
    if payload.get("authoritative") is not True:
        errors.append("validation:not_authoritative")
    if payload.get("retryable") is not False:
        errors.append("validation:retryable")
    if payload.get("eval_rc") != 0:
        errors.append("validation:eval_rc")
    if not isinstance(payload.get("valid"), bool):
        errors.append("validation:valid")
    if _finite(payload.get("score")) is None:
        errors.append("validation:score")
    for field in (
        "report",
        "private_artifact",
        "evaluated_archive",
        "expected_archive",
    ):
        nested = payload.get(field)
        if not isinstance(nested, dict):
            errors.append(f"validation:{field}_shape")
            continue
        nested_path = nested.get("path")
        expected_hash = nested.get("sha256")
        observed = _file_record(Path(nested_path)) if isinstance(nested_path, str) else _file_record(None)
        if not observed["exists"] or observed["sha256"] != expected_hash:
            errors.append(f"validation:{field}_hash")
    evaluated = result.get("evaluated_archive")
    expected = result.get("expected_archive")
    if (
        not isinstance(evaluated, dict)
        or not isinstance(expected, dict)
        or evaluated.get("sha256") != expected.get("sha256")
    ):
        errors.append("validation:archive_pair_hash")
    return result, errors


def _snapshot(
    path: Path | None,
    archive_path: Path | None,
    expected_seconds: float,
) -> tuple[dict[str, Any], list[str]]:
    record = _file_record(path)
    result = {
        **record,
        "captured_at": None,
        "elapsed_seconds": None,
        "boundary": None,
        "deadline": None,
        "freeze": None,
        "capture": None,
        "archive": {"path": None, "size": None, "sha256": None},
    }
    if path is None or archive_path is None:
        return result, ["final_snapshot:path_not_set"]
    try:
        validated = validate_snapshot(path, archive_path, expected_seconds)
    except (SnapshotContractError, OSError) as exc:
        return result, [f"final_snapshot:{exc}"]
    result.update(validated)
    return result, []


def _preflight(path: Path | None) -> tuple[dict[str, Any], list[str]]:
    record = _file_record(path)
    payload, load_error = _load_object(path)
    result = {
        **record,
        "status": None,
        "backend_serving_contract_sha256": None,
        "baseline_file_sha256": None,
        "baseline_valid_steps_per_min": None,
        "nonce_commitment_sha256": None,
        "hidden_baseline_identity": _empty_hidden_baseline_identity(),
        "matching_nonce_count": None,
        "public_baseline_file_sha256": None,
        "public_baseline_valid_steps_per_min": None,
        "public_nonce_commitment_sha256": None,
        "public_matching_nonce_count": None,
        "trusted_runtime_sources": None,
        "trusted_execution_sources": None,
        "mlsysforge_root": None,
        "mlsysforge_generated_exclusions": None,
        "mlsysforge_import_origins": None,
        "mlsysforge_harness_sources": None,
    }
    if load_error is not None or payload is None:
        return result, [f"preflight:{load_error}"]
    result.update(
        status=payload.get("status"),
        backend_serving_contract_sha256=payload.get(
            "backend_serving_contract_sha256"
        ),
        baseline_file_sha256=payload.get("baseline_file_sha256"),
        baseline_valid_steps_per_min=payload.get(
            "baseline_valid_steps_per_min"
        ),
        nonce_commitment_sha256=payload.get("nonce_commitment_sha256"),
        matching_nonce_count=payload.get("matching_nonce_count"),
        public_baseline_file_sha256=payload.get(
            "public_baseline_file_sha256"
        ),
        public_baseline_valid_steps_per_min=payload.get(
            "public_baseline_valid_steps_per_min"
        ),
        public_nonce_commitment_sha256=payload.get(
            "public_nonce_commitment_sha256"
        ),
        public_matching_nonce_count=payload.get(
            "public_matching_nonce_count"
        ),
        trusted_runtime_sources=payload.get("trusted_runtime_sources"),
        trusted_execution_sources=payload.get("trusted_execution_sources"),
        mlsysforge_root=payload.get("mlsysforge_root"),
        mlsysforge_generated_exclusions=payload.get(
            "mlsysforge_generated_exclusions"
        ),
        mlsysforge_import_origins=payload.get("mlsysforge_import_origins"),
        mlsysforge_harness_sources=payload.get("mlsysforge_harness_sources"),
    )
    errors = []
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 1
        or payload.get("status") != "passed"
    ):
        errors.append("preflight:not_passed")
    for key in (
        "backend_serving_contract_sha256",
        "baseline_file_sha256",
        "nonce_commitment_sha256",
        "public_baseline_file_sha256",
        "public_nonce_commitment_sha256",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"preflight:{key}")
    if payload.get("matching_nonce_count") != 1:
        errors.append("preflight:matching_nonce_count")
    hidden_rate = _finite(payload.get("baseline_valid_steps_per_min"))
    if hidden_rate is None or hidden_rate <= 0:
        errors.append("preflight:baseline_valid_steps_per_min")
    identity, identity_errors = _hidden_baseline_identity(
        {
            "file_sha256": payload.get("baseline_file_sha256"),
            "valid_steps_per_min": payload.get("baseline_valid_steps_per_min"),
            "backend_serving_contract_sha256": payload.get(
                "backend_serving_contract_sha256"
            ),
            "workload_nonce_sha256": payload.get("nonce_commitment_sha256"),
        },
        "preflight:hidden_baseline_identity",
    )
    result["hidden_baseline_identity"] = identity
    errors.extend(identity_errors)
    if payload.get("public_matching_nonce_count") != 1:
        errors.append("preflight:public_matching_nonce_count")
    public_rate = _finite(payload.get("public_baseline_valid_steps_per_min"))
    if public_rate is None or public_rate <= 0:
        errors.append("preflight:public_baseline_valid_steps_per_min")
    for key in (
        "trusted_runtime_sources",
        "trusted_execution_sources",
        "mlsysforge_harness_sources",
    ):
        attestation = payload.get(key)
        if not isinstance(attestation, dict) or set(attestation) != {
            "algorithm",
            "tree_sha256",
            "file_count",
            "total_bytes",
        }:
            errors.append(f"preflight:{key}_schema")
            continue
        if (
            attestation.get("algorithm")
            != "sha256-canonical-json-path-mode-size-content-v1"
            or not isinstance(attestation.get("tree_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", attestation["tree_sha256"])
            is None
            or isinstance(attestation.get("file_count"), bool)
            or not isinstance(attestation.get("file_count"), int)
            or attestation["file_count"] <= 0
            or isinstance(attestation.get("total_bytes"), bool)
            or not isinstance(attestation.get("total_bytes"), int)
            or attestation["total_bytes"] <= 0
        ):
            errors.append(f"preflight:{key}_values")
    if payload.get("mlsysforge_root") != EXPECTED_MLSYSFORGE_ROOT:
        errors.append("preflight:mlsysforge_root")
    if (
        payload.get("mlsysforge_generated_exclusions")
        != EXPECTED_MLSYSFORGE_GENERATED_EXCLUSIONS
    ):
        errors.append("preflight:mlsysforge_generated_exclusions")
    if payload.get("mlsysforge_import_origins") != EXPECTED_MLSYSFORGE_IMPORT_ORIGINS:
        errors.append("preflight:mlsysforge_import_origins")
    return result, errors


def _trusted_source_verification(
    path: Path | None,
    preflight: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    record = _file_record(path)
    payload, load_error = _load_object(path)
    result = {
        **record,
        "status": None,
        "attested_preflight_sha256": None,
        "trusted_runtime_sources": None,
        "trusted_execution_sources": None,
        "mlsysforge_root": None,
        "mlsysforge_generated_exclusions": None,
        "mlsysforge_import_origins": None,
        "mlsysforge_harness_sources": None,
    }
    if load_error is not None or payload is None:
        return result, [f"trusted_source_verification:{load_error}"]
    result.update(
        status=payload.get("status"),
        attested_preflight_sha256=payload.get("attested_preflight_sha256"),
        trusted_runtime_sources=payload.get("trusted_runtime_sources"),
        trusted_execution_sources=payload.get("trusted_execution_sources"),
        mlsysforge_root=payload.get("mlsysforge_root"),
        mlsysforge_generated_exclusions=payload.get(
            "mlsysforge_generated_exclusions"
        ),
        mlsysforge_import_origins=payload.get("mlsysforge_import_origins"),
        mlsysforge_harness_sources=payload.get("mlsysforge_harness_sources"),
    )
    errors: list[str] = []
    if set(payload) != {
        "schema_version",
        "status",
        "attested_preflight_sha256",
        "trusted_runtime_sources",
        "trusted_execution_sources",
        "mlsysforge_root",
        "mlsysforge_generated_exclusions",
        "mlsysforge_import_origins",
        "mlsysforge_harness_sources",
    }:
        errors.append("trusted_source_verification:schema")
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 1
        or payload.get("status") != "passed"
    ):
        errors.append("trusted_source_verification:not_passed")
    if payload.get("attested_preflight_sha256") != preflight.get("sha256"):
        errors.append("trusted_source_verification:preflight_hash")
    for key in (
        "trusted_runtime_sources",
        "trusted_execution_sources",
        "mlsysforge_harness_sources",
        "mlsysforge_root",
        "mlsysforge_generated_exclusions",
        "mlsysforge_import_origins",
    ):
        if payload.get(key) != preflight.get(key):
            errors.append(f"trusted_source_verification:{key}_mismatch")
    return result, errors


def _experiment_manifest(path: Path | None) -> tuple[dict[str, Any], list[str]]:
    record = _file_record(path)
    payload, load_error = _load_object(path)
    result = {
        **record,
        "preflight_sha256": None,
        "hidden_baseline_identity": _empty_hidden_baseline_identity(),
    }
    if load_error is not None or payload is None:
        return result, [f"experiment_manifest:{load_error}"]
    errors: list[str] = []
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 2
    ):
        errors.append("experiment_manifest:schema_version")
    preflight_sha256 = payload.get("preflight_sha256")
    if not isinstance(preflight_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", preflight_sha256
    ) is None:
        errors.append("experiment_manifest:preflight_sha256")
    identity, identity_errors = _hidden_baseline_identity(
        payload.get("hidden_baseline_identity"),
        "experiment_manifest:hidden_baseline_identity",
    )
    result.update(
        preflight_sha256=preflight_sha256,
        hidden_baseline_identity=identity,
    )
    errors.extend(identity_errors)
    return result, errors


def _selection(
    selection_path: Path | None,
    selected_validation_path: Path | None,
) -> tuple[dict[str, Any], list[str]]:
    selection_record = _file_record(selection_path)
    payload, load_error = _load_object(selection_path)
    selected_validation, selected_validation_errors = _validated_evaluation(
        selected_validation_path,
        allowed_classifications={"candidate_valid"},
    )
    result: dict[str, Any] = {
        "selection": selection_record,
        "hidden_baseline_identity": _empty_hidden_baseline_identity(),
        "expected_preflight_sha256": None,
        "discovered_candidate_count": None,
        "eligible_candidate_count": None,
        "round": None,
        "submission_id": None,
        "valid": None,
        "score": None,
        "report": {"path": None, "sha256": None},
        "private_artifact": {"path": None, "sha256": None},
        "archive": {"path": None, "sha256": None},
        "qualification_validation": {"path": None, "sha256": None},
        "qualification_hidden_baseline_identity": _empty_hidden_baseline_identity(),
        "selected_validation": selected_validation.get("validation"),
        "selected_hidden_baseline_identity": selected_validation.get(
            "hidden_baseline_identity"
        ),
    }
    errors = list(selected_validation_errors)
    if load_error is not None or payload is None:
        errors.append(f"selection:{load_error}")
        return result, errors
    best = payload.get("best")
    expected_top_fields = {
        "schema_version",
        "selection",
        "score_direction",
        "tie_break",
        "discovered_candidate_count",
        "eligible_candidate_count",
        "rejections",
        "hidden_baseline_identity",
        "expected_preflight_sha256",
        "best_validation_sha256",
        "best",
        "preserved_archive",
        "preserved_report",
        "preserved_private_artifact",
        "preserved_qualification_validation",
    }
    expected_best_fields = {
        "round",
        "submission_id",
        "score",
        "pass_rate",
        "submitted_at",
        "archive_sha256",
        "report_sha256",
        "private_artifact_sha256",
        "qualification_validation_sha256",
    }
    if (
        payload.get("schema_version") != 2
        or set(payload) != expected_top_fields
        or not isinstance(best, dict)
        or set(best) != expected_best_fields
        or payload.get("selection") != "valid_then_score"
        or payload.get("score_direction") != "maximize"
        or payload.get("tie_break")
        != "highest_raw_score_then_earliest_submitted_at"
    ):
        errors.append("selection:schema")
        return result, errors
    selection_identity, identity_errors = _hidden_baseline_identity(
        payload.get("hidden_baseline_identity"),
        "selection:hidden_baseline_identity",
    )
    errors.extend(identity_errors)
    preflight_sha256 = payload.get("expected_preflight_sha256")
    if not isinstance(preflight_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", preflight_sha256
    ) is None:
        errors.append("selection:expected_preflight_sha256")
    discovered = payload.get("discovered_candidate_count")
    eligible = payload.get("eligible_candidate_count")
    rejections = payload.get("rejections")
    if (
        isinstance(discovered, bool)
        or not isinstance(discovered, int)
        or discovered < 1
        or isinstance(eligible, bool)
        or not isinstance(eligible, int)
        or eligible < 1
        or eligible > discovered
        or not isinstance(rejections, list)
        or len(rejections) != discovered - eligible
    ):
        errors.append("selection:candidate_counts")
    score = _finite(best.get("score"))
    round_name = best.get("round")
    submission_id = best.get("submission_id")
    report_path = payload.get("preserved_report")
    private_path = payload.get("preserved_private_artifact")
    archive_path = payload.get("preserved_archive")
    qualification_path = payload.get("preserved_qualification_validation")
    report_record = _file_record(Path(report_path)) if isinstance(report_path, str) else _file_record(None)
    private_record = _file_record(Path(private_path)) if isinstance(private_path, str) else _file_record(None)
    archive_record = _file_record(Path(archive_path)) if isinstance(archive_path, str) else _file_record(None)
    qualification_record = _file_record(Path(qualification_path)) if isinstance(qualification_path, str) else _file_record(None)
    qualification, qualification_errors = _validated_evaluation(
        Path(qualification_path) if isinstance(qualification_path, str) else None,
        allowed_classifications={"candidate_valid"},
    )
    errors.extend(f"qualification_{item}" for item in qualification_errors)
    result.update(
        hidden_baseline_identity=selection_identity,
        expected_preflight_sha256=preflight_sha256,
        discovered_candidate_count=discovered,
        eligible_candidate_count=eligible,
        round=round_name,
        submission_id=submission_id,
        valid=selected_validation.get("valid"),
        score=score,
        report={"path": report_record["path"], "sha256": report_record["sha256"]},
        private_artifact={"path": private_record["path"], "sha256": private_record["sha256"]},
        archive={"path": archive_record["path"], "sha256": archive_record["sha256"]},
        qualification_validation={
            "path": qualification_record["path"],
            "sha256": qualification_record["sha256"],
        },
        qualification_hidden_baseline_identity=qualification.get(
            "hidden_baseline_identity"
        ),
    )
    if not isinstance(round_name, str) or not round_name:
        errors.append("selection:round")
    if not isinstance(submission_id, str) or not submission_id:
        errors.append("selection:submission_id")
    if score is None or score <= 0.0:
        errors.append("selection:score")
    if selected_validation.get("valid") is not True or qualification.get("valid") is not True:
        errors.append("selection:not_valid")
    if (
        _finite(selected_validation.get("score")) != score
        or _finite(qualification.get("score")) != score
    ):
        errors.append("selection:score_mismatch")
    if report_record["sha256"] != best.get("report_sha256"):
        errors.append("selection:report_hash")
    if private_record["sha256"] != best.get("private_artifact_sha256"):
        errors.append("selection:private_artifact_hash")
    if archive_record["sha256"] != best.get("archive_sha256"):
        errors.append("selection:archive_hash")
    if (
        qualification_record["sha256"]
        != best.get("qualification_validation_sha256")
        or qualification_record["sha256"] != payload.get("best_validation_sha256")
        or not isinstance(qualification_path, str)
        or qualification_record["path"]
        != str(Path(qualification_path).resolve(strict=False))
    ):
        errors.append("selection:qualification_validation_hash")
    for label, validation in (
        ("qualification", qualification),
        ("selected", selected_validation),
    ):
        for field, record in (
            ("report", report_record),
            ("private_artifact", private_record),
            ("evaluated_archive", archive_record),
            ("expected_archive", archive_record),
        ):
            observed = validation.get(field)
            if not isinstance(observed, dict) or observed.get("sha256") != record["sha256"]:
                errors.append(f"selection:{label}_{field}_hash")
        if validation.get("submission_id") != submission_id:
            errors.append(f"selection:{label}_submission_id")
        if validation.get("hidden_baseline_identity") != selection_identity:
            errors.append(f"selection:{label}_hidden_baseline_identity")
    return result, errors


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    parent = path.parent.resolve(strict=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
            raise RuntimeError("completion target is unsafe")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = -1
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _optional_int(value: str | None) -> int | None:
    return None if value in (None, "") else int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--requested-status",
        required=True,
        choices=("failed", "passed", NO_VALID_SUBMISSION_STATUS),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--finished-at", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--codex-cli-version", required=True)
    parser.add_argument("--task-timeout", required=True, type=float)
    parser.add_argument("--submission-cooldown", required=True, type=int)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--reason", default="")
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--run-rc")
    parser.add_argument("--final-rc")
    parser.add_argument("--final-attempt", type=int)
    parser.add_argument("--final-eval-id")
    parser.add_argument("--cleanup-status", required=True)
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--task-spec", required=True, type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--experiment-manifest", type=Path)
    parser.add_argument("--trusted-source-verification", type=Path)
    parser.add_argument("--final-archive", type=Path)
    parser.add_argument("--final-snapshot", type=Path)
    parser.add_argument("--final-validation", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selected-validation", type=Path)
    parser.add_argument("--dedicated-nonce-source", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_rc = _optional_int(args.run_rc)
    final_rc = _optional_int(args.final_rc)
    final_archive = _file_record(args.final_archive)
    preflight, preflight_errors = _preflight(args.preflight)
    experiment_manifest, experiment_manifest_errors = _experiment_manifest(
        args.experiment_manifest
    )
    source_verification, source_verification_errors = _trusted_source_verification(
        args.trusted_source_verification, preflight
    )
    snapshot, snapshot_errors = _snapshot(
        args.final_snapshot, args.final_archive, args.task_timeout
    )
    final, final_errors = _validated_evaluation(
        args.final_validation,
        allowed_classifications={"candidate_valid", "candidate_invalid"},
    )
    selection, selection_errors = _selection(args.selection, args.selected_validation)
    launcher = _file_record(args.launcher)
    task_spec = _file_record(args.task_spec)
    nonce_removed = (
        args.dedicated_nonce_source is not None
        and not args.dedicated_nonce_source.exists()
        and not args.dedicated_nonce_source.is_symlink()
    )
    strict_errors: list[str] = []
    selection_required = args.requested_status == "passed"
    if args.requested_status in TERMINAL_STATUSES:
        strict_errors.extend(preflight_errors)
        strict_errors.extend(experiment_manifest_errors)
        strict_errors.extend(source_verification_errors)
        strict_errors.extend(snapshot_errors)
        strict_errors.extend(f"final_{item}" for item in final_errors)
        expected_identity = preflight.get("hidden_baseline_identity")
        identity_sources = [
            (
                "manifest_hidden_baseline_identity_mismatch",
                experiment_manifest.get("hidden_baseline_identity"),
            ),
            (
                "final_hidden_baseline_identity_mismatch",
                final.get("hidden_baseline_identity"),
            ),
        ]
        if selection_required:
            strict_errors.extend(f"selected_{item}" for item in selection_errors)
            identity_sources.extend(
                (
                    (
                        "selected_hidden_baseline_identity_mismatch",
                        selection.get("selected_hidden_baseline_identity"),
                    ),
                    (
                        "selection_hidden_baseline_identity_mismatch",
                        selection.get("hidden_baseline_identity"),
                    ),
                    (
                        "qualification_hidden_baseline_identity_mismatch",
                        selection.get("qualification_hidden_baseline_identity"),
                    ),
                )
            )
        else:
            # Zero-valid: no selection evidence may exist, and the outcome
            # must carry the selector's own reason.
            if selection["selection"]["exists"] or selection["selected_validation"]["exists"]:
                strict_errors.append("no_valid_submission_with_selection_evidence")
            if not args.reason.strip():
                strict_errors.append("no_valid_submission_reason")
        for label, observed_identity in identity_sources:
            if observed_identity != expected_identity:
                strict_errors.append(label)
        if experiment_manifest.get("preflight_sha256") != preflight.get("sha256"):
            strict_errors.append("experiment_manifest_preflight_hash_mismatch")
        if selection_required and selection.get("expected_preflight_sha256") != preflight.get("sha256"):
            strict_errors.append("selection_preflight_hash_mismatch")
        if run_rc != 0:
            strict_errors.append("run_rc")
        if final_rc != 0:
            strict_errors.append("final_rc")
        if final.get("eval_rc") != final_rc:
            strict_errors.append("final_validation_rc_mismatch")
        if final.get("submission_id") != args.final_eval_id:
            strict_errors.append("final_validation_run_id_mismatch")
        for archive_field in ("evaluated_archive", "expected_archive"):
            nested = final.get(archive_field)
            if (
                not isinstance(nested, dict)
                or nested.get("sha256") != final_archive.get("sha256")
            ):
                strict_errors.append(f"final_{archive_field}_hash_mismatch")
        snapshot_archive = snapshot.get("archive")
        if (
            not isinstance(snapshot_archive, dict)
            or snapshot_archive.get("sha256") != final_archive.get("sha256")
        ):
            strict_errors.append("final_snapshot_archive_hash_mismatch")
        if final.get("classification") == "candidate_valid" and final.get("valid") is not True:
            strict_errors.append("final_validity_mismatch")
        if final.get("classification") == "candidate_invalid" and (
            final.get("valid") is not False or _finite(final.get("score")) != 0.0
        ):
            strict_errors.append("final_invalidity_mismatch")
        if args.exit_code != 0:
            strict_errors.append("exit_code")
        if args.cleanup_status != "passed":
            strict_errors.append("cleanup_status")
        if not nonce_removed:
            strict_errors.append("dedicated_nonce_not_removed")
        if not final_archive["exists"] or not final_archive["sha256"]:
            strict_errors.append("final_archive")
        if args.final_attempt is None or args.final_attempt < 1:
            strict_errors.append("final_attempt")
        if not launcher["exists"] or not task_spec["exists"]:
            strict_errors.append("provenance_sources")
    actual_status = (
        args.requested_status
        if args.requested_status in TERMINAL_STATUSES and not strict_errors
        else "failed"
    )
    payload = {
        "schema_version": 2,
        "status": actual_status,
        "started_at": args.started_at,
        "finished_at": args.finished_at,
        "run_id": args.run_id,
        "task_id": args.task_id,
        "hidden_baseline_identity": preflight.get(
            "hidden_baseline_identity", _empty_hidden_baseline_identity()
        ),
        "stage": args.stage,
        "failure": None
        if actual_status in TERMINAL_STATUSES
        else {
            "reason": args.reason or "formal run did not meet strict completion conditions",
            "exit_code": args.exit_code,
            "strict_errors": strict_errors,
        },
        "no_valid_submission": {"reason": args.reason}
        if actual_status == NO_VALID_SUBMISSION_STATUS
        else None,
        "provenance": {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "codex_cli_version": args.codex_cli_version,
            "timeout_seconds": args.task_timeout,
            "submission_cooldown_seconds": args.submission_cooldown,
            "launcher": launcher,
            "task_spec": task_spec,
            "preflight": preflight,
            "experiment_manifest": experiment_manifest,
            "trusted_source_verification": source_verification,
        },
        "run": {"rc": run_rc},
        "final_snapshot": snapshot,
        "final_archive": final_archive,
        "final_evaluation": {
            "attempt": args.final_attempt,
            "run_id": args.final_eval_id,
            "rc": final_rc,
            **final,
        },
        "selected": selection,
        "cleanup": {
            "status": args.cleanup_status,
            "dedicated_nonce_removed": nonce_removed,
        },
    }
    _atomic_write(args.output, payload)
    return 0 if actual_status == args.requested_status else 2


if __name__ == "__main__":
    raise SystemExit(main())
