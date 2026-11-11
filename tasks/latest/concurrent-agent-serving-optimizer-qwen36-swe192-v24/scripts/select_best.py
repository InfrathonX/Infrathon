#!/usr/bin/env python3
"""Authoritatively qualify, select, and preserve the best hidden submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


HEX64 = re.compile(r"[0-9a-f]{64}")
IDENTITY_KEYS = {
    "file_sha256",
    "valid_steps_per_min",
    "backend_serving_contract_sha256",
    "workload_nonce_sha256",
}


class SelectionError(RuntimeError):
    """Raised when selection cannot be completed fail-closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SelectionError(f"{label} is not a safe regular file")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SelectionError(f"{label} is not valid strict JSON") from exc
    if not isinstance(payload, dict):
        raise SelectionError(f"{label} must be a JSON object")
    return payload


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def validate_hidden_baseline_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != IDENTITY_KEYS:
        raise SelectionError("hidden baseline identity has the wrong shape")
    rate = finite_number(value.get("valid_steps_per_min"))
    if rate is None or rate <= 0.0:
        raise SelectionError("hidden baseline identity rate is invalid")
    for key in (
        "file_sha256",
        "backend_serving_contract_sha256",
        "workload_nonce_sha256",
    ):
        digest = value.get(key)
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            raise SelectionError(f"hidden baseline identity {key} is invalid")
    return {
        "file_sha256": value["file_sha256"],
        "valid_steps_per_min": rate,
        "backend_serving_contract_sha256": value[
            "backend_serving_contract_sha256"
        ],
        "workload_nonce_sha256": value["workload_nonce_sha256"],
    }


def expected_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    if preflight.get("status") != "passed":
        raise SelectionError("expected preflight did not pass")
    nested = preflight.get("hidden_baseline_identity")
    nested_identity = (
        validate_hidden_baseline_identity(nested) if nested is not None else None
    )
    flat_names = {
        "file_sha256": "baseline_file_sha256",
        "valid_steps_per_min": "baseline_valid_steps_per_min",
        "backend_serving_contract_sha256": "backend_serving_contract_sha256",
        "workload_nonce_sha256": "nonce_commitment_sha256",
    }
    flat_present = [name in preflight for name in flat_names.values()]
    if any(flat_present) and not all(flat_present):
        raise SelectionError("expected preflight has an incomplete baseline identity")
    flat_identity = None
    if all(flat_present):
        flat_identity = validate_hidden_baseline_identity(
            {key: preflight[name] for key, name in flat_names.items()}
        )
    if nested_identity is None and flat_identity is None:
        raise SelectionError("expected preflight lacks a hidden baseline identity")
    if (
        nested_identity is not None
        and flat_identity is not None
        and nested_identity != flat_identity
    ):
        raise SelectionError("expected preflight baseline identities disagree")
    if nested_identity is not None:
        return nested_identity
    assert flat_identity is not None
    return flat_identity


def rejection(
    *,
    round_name: str,
    submission_id: str | None,
    reason: str,
    validator_returncode: int | None = None,
    validation: dict[str, Any] | None = None,
    validation_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "round": round_name,
        "submission_id": submission_id,
        "reason": reason,
        "validator_returncode": validator_returncode,
        "classification": validation.get("classification")
        if isinstance(validation, dict)
        else None,
        "authoritative": validation.get("authoritative")
        if isinstance(validation, dict)
        else None,
        "valid": validation.get("valid")
        if isinstance(validation, dict)
        else None,
        "qualification_validation_sha256": validation_sha256,
    }


def safe_record(path: Path, label: str) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise SelectionError(f"{label} is missing or unsafe")
    return {"path": str(path.resolve(strict=True)), "sha256": sha256(path)}


def qualification_rejection_reason(
    *,
    returncode: int,
    validation: dict[str, Any],
    task_id: str,
    submission_id: str,
    identity: dict[str, Any],
    report_record: dict[str, str],
    private_record: dict[str, str],
    archive_record: dict[str, str],
) -> str | None:
    if returncode != 0:
        return "validator_exit_nonzero"
    if validation.get("schema_version") != 1:
        return "qualification_schema_mismatch"
    if validation.get("classification") != "candidate_valid":
        return "classification_not_candidate_valid"
    if validation.get("authoritative") is not True:
        return "qualification_not_authoritative"
    if validation.get("valid") is not True:
        return "qualification_not_valid"
    if validation.get("task_id") != task_id:
        return "qualification_task_identity_mismatch"
    if validation.get("submission_id") != submission_id:
        return "qualification_submission_identity_mismatch"
    try:
        observed_identity = validate_hidden_baseline_identity(
            validation.get("hidden_baseline_identity")
        )
    except SelectionError:
        return "qualification_hidden_baseline_identity_invalid"
    if observed_identity != identity:
        return "hidden_baseline_identity_mismatch"
    records = (
        (validation.get("report"), report_record, "qualification_report_mismatch"),
        (
            validation.get("private_artifact"),
            private_record,
            "qualification_private_artifact_mismatch",
        ),
        (
            validation.get("evaluated_archive"),
            archive_record,
            "qualification_evaluated_archive_mismatch",
        ),
        (
            validation.get("expected_archive"),
            archive_record,
            "qualification_expected_archive_mismatch",
        ),
    )
    for observed, expected, reason in records:
        if (
            not isinstance(observed, dict)
            or observed.get("sha256") != expected["sha256"]
        ):
            return reason
    return None


def stage_copy(source: Path, output_dir: Path, prefix: str) -> tuple[Path, str]:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{prefix}.", suffix=".tmp", dir=output_dir
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as source_stream, os.fdopen(
            descriptor, "wb"
        ) as output_stream:
            descriptor = -1
            for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                output_stream.write(chunk)
                digest.update(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.chmod(temporary, 0o600)
        return temporary, digest.hexdigest()
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def publish_new(staged: Path, destination: Path) -> None:
    try:
        os.link(staged, destination)
    except FileExistsError as exc:
        raise SelectionError(f"refusing to overwrite: {destination}") from exc
    staged.unlink()


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    data = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        publish_new(temporary, path)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--source-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument("--expected-preflight", required=True, type=Path)
    args = parser.parse_args()

    validator = args.validator.resolve(strict=True)
    preflight_path = args.expected_preflight.resolve(strict=True)
    if args.validator.is_symlink() or not validator.is_file():
        raise SelectionError("validator is not a safe regular file")
    if args.expected_preflight.is_symlink() or not preflight_path.is_file():
        raise SelectionError("expected preflight is not a safe regular file")
    preflight = load_object(preflight_path, "expected preflight")
    identity = expected_identity(preflight)
    preflight_sha256 = sha256(preflight_path)

    discovered_count = 0
    eligible: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_reports: set[Path] = set()
    discovery_index = 0

    with tempfile.TemporaryDirectory(prefix="kvbench-selector-") as temporary_name:
        qualification_root = Path(temporary_name)
        for source_root_argument in args.source_root:
            source_root = source_root_argument.resolve(strict=True)
            if not source_root.is_dir():
                raise SelectionError(f"source root is not a directory: {source_root}")
            for report_path in sorted(source_root.glob("submissions/*/report.json")):
                if report_path.parent.name.startswith("public-"):
                    continue
                resolved_report = report_path.resolve(strict=False)
                if resolved_report in seen_reports:
                    continue
                seen_reports.add(resolved_report)
                discovered_count += 1
                discovery_index += 1
                round_name = report_path.parent.name
                submission_id: str | None = None
                try:
                    report = load_object(report_path, f"report for {round_name}")
                    raw_submission_id = report.get("submission_id")
                    if not isinstance(raw_submission_id, str) or not raw_submission_id:
                        raise SelectionError("report submission_id is invalid")
                    submission_id = raw_submission_id
                except SelectionError:
                    rejections.append(
                        rejection(
                            round_name=round_name,
                            submission_id=None,
                            reason="report_invalid",
                        )
                    )
                    continue

                private_path = report_path.with_name(
                    "private_judge_artifacts.tar.gz"
                )
                archive_path = report_path.with_name("submission.tar.gz")
                if (
                    not private_path.is_file()
                    or private_path.is_symlink()
                    or not archive_path.is_file()
                    or archive_path.is_symlink()
                ):
                    rejections.append(
                        rejection(
                            round_name=round_name,
                            submission_id=submission_id,
                            reason="candidate_artifact_missing_or_unsafe",
                        )
                    )
                    continue

                qualification_path = (
                    qualification_root / f"qualification-{discovery_index:06d}.json"
                )
                command = [
                    sys.executable,
                    str(validator),
                    "--report",
                    str(report_path),
                    "--private-artifact",
                    str(private_path),
                    "--expected-submission-id",
                    submission_id,
                    "--eval-rc",
                    "0",
                    "--evaluated-archive",
                    str(archive_path),
                    "--expected-archive",
                    str(archive_path),
                    "--expected-preflight",
                    str(preflight_path),
                    "--output",
                    str(qualification_path),
                ]
                try:
                    completed = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=300,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    rejections.append(
                        rejection(
                            round_name=round_name,
                            submission_id=submission_id,
                            reason="validator_timeout",
                        )
                    )
                    continue
                if not qualification_path.is_file() or qualification_path.is_symlink():
                    rejections.append(
                        rejection(
                            round_name=round_name,
                            submission_id=submission_id,
                            reason="qualification_output_missing_or_unsafe",
                            validator_returncode=completed.returncode,
                        )
                    )
                    continue
                validation_sha256 = sha256(qualification_path)
                try:
                    validation = load_object(
                        qualification_path, f"qualification for {round_name}"
                    )
                    report_record = safe_record(report_path, "candidate report")
                    private_record = safe_record(
                        private_path, "candidate private artifact"
                    )
                    archive_record = safe_record(archive_path, "candidate archive")
                except SelectionError:
                    rejections.append(
                        rejection(
                            round_name=round_name,
                            submission_id=submission_id,
                            reason="qualification_or_candidate_record_invalid",
                            validator_returncode=completed.returncode,
                            validation_sha256=validation_sha256,
                        )
                    )
                    continue
                reason = qualification_rejection_reason(
                    returncode=completed.returncode,
                    validation=validation,
                    task_id=args.task_id,
                    submission_id=submission_id,
                    identity=identity,
                    report_record=report_record,
                    private_record=private_record,
                    archive_record=archive_record,
                )
                score = finite_number(validation.get("score"))
                report_score = finite_number(report.get("score"))
                submitted_at = finite_number(report.get("submitted_at"))
                if reason is None and (
                    report.get("task_id") != args.task_id
                    or report.get("valid") is not True
                    or score is None
                    or report_score is None
                    or score != report_score
                    or submitted_at is None
                ):
                    reason = "validated_report_selection_fields_invalid"
                if reason is not None:
                    rejections.append(
                        rejection(
                            round_name=round_name,
                            submission_id=submission_id,
                            reason=reason,
                            validator_returncode=completed.returncode,
                            validation=validation,
                            validation_sha256=validation_sha256,
                        )
                    )
                    continue
                pass_rate = finite_number(report.get("pass_rate"))
                eligible.append(
                    {
                        "round": round_name,
                        "submission_id": submission_id,
                        "score": score,
                        "pass_rate": pass_rate,
                        "submitted_at": submitted_at,
                        "discovery_index": discovery_index,
                        "report_source": Path(report_record["path"]),
                        "private_source": Path(private_record["path"]),
                        "archive_source": Path(archive_record["path"]),
                        "qualification_source": qualification_path,
                        "report_sha256": report_record["sha256"],
                        "private_artifact_sha256": private_record["sha256"],
                        "archive_sha256": archive_record["sha256"],
                        "qualification_validation_sha256": validation_sha256,
                    }
                )

        if not eligible:
            raise SelectionError(
                f"no authoritatively eligible hidden submission was found "
                f"({discovered_count} discovered, {len(rejections)} rejected)"
            )

        best = min(
            eligible,
            key=lambda row: (
                -row["score"],
                row["submitted_at"],
                row["discovery_index"],
            ),
        )
        output_argument = args.output_dir
        if output_argument.is_symlink():
            raise SelectionError("output directory must not be a symlink")
        output_argument.mkdir(parents=True, exist_ok=True)
        output_dir = output_argument.resolve(strict=True)
        if not output_dir.is_dir():
            raise SelectionError("output path is not a directory")

        selection_destination = output_dir / "best_selection.json"
        if selection_destination.exists() or selection_destination.is_symlink():
            raise SelectionError(f"refusing to overwrite: {selection_destination}")

        best_dir = output_dir / "best"
        if best_dir.exists() or best_dir.is_symlink():
            raise SelectionError(
                f"refusing to reuse best evidence directory: {best_dir}"
            )
        try:
            best_dir.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise SelectionError(
                f"refusing to reuse best evidence directory: {best_dir}"
            ) from exc
        os.chmod(best_dir, 0o700)

        destinations = {
            "archive": best_dir / "submission.tar.gz",
            "report": best_dir / "report.json",
            "private": best_dir / "private_judge_artifacts.tar.gz",
            "qualification": best_dir / "qualification-validation.json",
            "selection": selection_destination,
        }
        for destination in destinations.values():
            if destination.exists() or destination.is_symlink():
                raise SelectionError(f"refusing to overwrite: {destination}")

        staged: list[tuple[Path, Path]] = []
        source_specs = (
            ("archive", best["archive_source"], best["archive_sha256"]),
            ("report", best["report_source"], best["report_sha256"]),
            (
                "private",
                best["private_source"],
                best["private_artifact_sha256"],
            ),
            (
                "qualification",
                best["qualification_source"],
                best["qualification_validation_sha256"],
            ),
        )
        try:
            for name, source, expected_sha256 in source_specs:
                temporary, copied_sha256 = stage_copy(source, best_dir, name)
                if copied_sha256 != expected_sha256:
                    temporary.unlink(missing_ok=True)
                    raise SelectionError(
                        f"selected {name} changed after authoritative validation"
                    )
                staged.append((temporary, destinations[name]))
            for temporary, destination in staged:
                publish_new(temporary, destination)
            staged.clear()
        finally:
            for temporary, _destination in staged:
                temporary.unlink(missing_ok=True)

        preserved_hashes = {
            name: sha256(destinations[name])
            for name in ("archive", "report", "private", "qualification")
        }
        if preserved_hashes != {
            "archive": best["archive_sha256"],
            "report": best["report_sha256"],
            "private": best["private_artifact_sha256"],
            "qualification": best["qualification_validation_sha256"],
        }:
            raise SelectionError("preserved best evidence failed on-site hashing")

        best_payload = {
            "round": best["round"],
            "submission_id": best["submission_id"],
            "score": best["score"],
            "pass_rate": best["pass_rate"],
            "submitted_at": best["submitted_at"],
            "archive_sha256": best["archive_sha256"],
            "report_sha256": best["report_sha256"],
            "private_artifact_sha256": best["private_artifact_sha256"],
            "qualification_validation_sha256": best[
                "qualification_validation_sha256"
            ],
        }
        payload = {
            "schema_version": 2,
            "selection": "valid_then_score",
            "score_direction": "maximize",
            "tie_break": "highest_raw_score_then_earliest_submitted_at",
            "discovered_candidate_count": discovered_count,
            "eligible_candidate_count": len(eligible),
            "rejections": rejections,
            "hidden_baseline_identity": identity,
            "expected_preflight_sha256": preflight_sha256,
            "best_validation_sha256": best[
                "qualification_validation_sha256"
            ],
            "best": best_payload,
            "preserved_archive": str(destinations["archive"]),
            "preserved_report": str(destinations["report"]),
            "preserved_private_artifact": str(destinations["private"]),
            "preserved_qualification_validation": str(
                destinations["qualification"]
            ),
        }
        write_new_json(destinations["selection"], payload)
        print(json.dumps(payload, allow_nan=False, sort_keys=True))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SelectionError as exc:
        raise SystemExit(str(exc)) from exc
