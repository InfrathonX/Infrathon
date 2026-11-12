#!/usr/bin/env python3
"""Validate one persisted SForge evaluation and its immutable archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
# The harness's own cap (SUBMISSION_ARCHIVE_MAX_MEMBERS, sforge/harness/run_evaluation.py):
# a run keeps every version, so archives that the Judge scored can hold thousands of files.
MAX_ARCHIVE_MEMBERS = 20_000
MAX_MEMBER_BYTES = 1024 * 1024
# Per selected version, as the Judge's source admission counts it (images/judge/verifier).
MAX_LOGICAL_BYTES = 16 * 1024 * 1024
ALLOWED_TOP_LEVEL = {"solutions", "FINAL_VERSION", "notes.md"}
ALLOWED_SOURCE_SUFFIXES = {
    ".py",
    ".cu",
    ".cuh",
    ".cpp",
    ".h",
    ".json",
    ".md",
    ".txt",
    ".toml",
}
VERSION_RE = re.compile(r"engine_v[0-9]{3}")
TASK_ID = "eagle3_spec_decode_engine_qwen3_8b_h100_v1"
ADAPTER_SOURCE = "eagle3_spec_decode_v1_verifier_adapter"
PRIVATE_ARTIFACT = "private_judge_artifacts.tar.gz"
PRIVATE_MEMBER = "eagle3_spec_decode_private.json"
REQUIRED_GATES = (
    "protected_ok",
    "source_ok",
    "worker_ok",
    "o1_ok",
    "o2_ok",
    "tiers_ok",
    "o3_o4_ok",
    "o5_ok",
    "wall_ok",
    "memory_ok",
    "idle_ok",
)


def sha256(path: Path) -> str:
    """Hash a file without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, label: str, maximum: int) -> Path:
    """Require one bounded, non-linked regular file."""
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or not 0 < metadata.st_size <= maximum
    ):
        raise ValueError(f"{label} violates the regular-file contract")
    return path


def reject_constant(value: str) -> None:
    """Reject non-finite JSON constants."""
    raise ValueError(f"non-finite JSON constant: {value}")


def finite_tree(value: Any) -> bool:
    """Return whether a JSON-compatible tree contains only finite numbers."""
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and finite_tree(item)
            for key, item in value.items()
        )
    return False


def json_object(path: Path, label: str) -> dict[str, Any]:
    """Load a bounded JSON object with finite numeric leaves."""
    regular_file(path, label, MAX_JSON_BYTES)
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict) or not finite_tree(value):
        raise ValueError(f"{label} must be a finite JSON object")
    return value


def canonical_member(member: tarfile.TarInfo) -> PurePosixPath:
    """Validate one portable submission archive member name and kind."""
    name = member.name
    while name.startswith("./"):
        name = name[2:]
    name = name.rstrip("/")
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != name
    ):
        raise ValueError(f"unsafe archive member: {member.name!r}")
    if not member.isfile() and not member.isdir():
        raise ValueError(f"unsupported archive member kind: {member.name!r}")
    if path.parts[0] not in ALLOWED_TOP_LEVEL:
        raise ValueError(f"unexpected top-level archive member: {member.name!r}")
    return path


def validate_archive(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Validate archive identity, paths, and required submission members."""
    regular_file(path, "submission.tar.gz", MAX_ARCHIVE_BYTES)
    observed = sha256(path)
    if observed != expected_sha256:
        raise ValueError("submission archive SHA-256 differs")
    files: set[str] = set()
    versions: set[str] = set()
    names: set[str] = set()
    logical_bytes = 0
    version_bytes: dict[str, int] = {}
    selector: bytes | None = None
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("submission archive is empty")
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("submission archive has too many members")
        for member in members:
            canonical = canonical_member(member)
            canonical_name = canonical.as_posix()
            if canonical_name in names:
                raise ValueError(
                    f"duplicate archive member: {canonical_name!r}"
                )
            names.add(canonical_name)
            if member.isdir():
                if canonical_name != "solutions" and not (
                    len(canonical.parts) >= 2
                    and canonical.parts[0] == "solutions"
                    and VERSION_RE.fullmatch(canonical.parts[1]) is not None
                ):
                    raise ValueError(
                        f"unexpected archive directory: {canonical_name!r}"
                    )
                continue
            if not 0 <= member.size <= MAX_MEMBER_BYTES:
                raise ValueError(
                    f"archive member exceeds its size limit: {canonical_name!r}"
                )
            logical_bytes += member.size
            allowed_root_file = canonical_name in {"FINAL_VERSION", "notes.md"}
            allowed_source = (
                len(canonical.parts) >= 3
                and canonical.parts[0] == "solutions"
                and VERSION_RE.fullmatch(canonical.parts[1]) is not None
                and canonical.suffix.lower() in ALLOWED_SOURCE_SUFFIXES
            )
            if not allowed_root_file and not allowed_source:
                raise ValueError(
                    f"unexpected archive file: {canonical_name!r}"
                )
            if member.isfile():
                files.add(canonical_name)
            if len(canonical.parts) >= 2 and canonical.parts[0] == "solutions":
                versions.add(canonical.parts[1])
                if member.isfile():
                    version_bytes[canonical.parts[1]] = (
                        version_bytes.get(canonical.parts[1], 0) + member.size
                    )
            if canonical_name == "FINAL_VERSION":
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("submission selector is unreadable")
                selector = stream.read(17)
    if selector is None or re.fullmatch(rb"[1-9][0-9]{0,2}\n?", selector) is None:
        raise ValueError("FINAL_VERSION must be a canonical integer 1..999")
    if "notes.md" not in files:
        raise ValueError("submission archive lacks notes.md")
    selected_version = f"engine_v{int(selector):03d}"
    selected_candidate = f"solutions/{selected_version}/candidate.py"
    if selected_version not in versions or selected_candidate not in files:
        raise ValueError("FINAL_VERSION selects a missing candidate.py")
    if version_bytes.get(selected_version, 0) > MAX_LOGICAL_BYTES:
        raise ValueError("the selected version exceeds its logical-size limit")
    return {
        "path": str(path.resolve()),
        "sha256": observed,
        "size": path.stat().st_size,
        "file_count": len(files),
        "logical_size": logical_bytes,
        "final_version": int(selector),
        "versions": sorted(versions),
    }


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate the persisted SForge report without exposing private details."""
    # Native SForge ``EvalReport.to_dict`` places validity and score at the top
    # level and deliberately has no free-form status field.
    if isinstance(report.get("valid"), bool):
        if report.get("task_id") != TASK_ID:
            raise ValueError("report task identity differs")
        submission_id = report.get("submission_id")
        if not isinstance(submission_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}",
            submission_id,
        ) is None:
            raise ValueError("report submission identity is unsafe")
        if not isinstance(report.get("timed_out"), bool):
            raise ValueError("report timed_out must be boolean")
        score = report.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("report score must be numeric")
        if not math.isfinite(float(score)) or float(score) < 0:
            raise ValueError("report score must be finite and nonnegative")
        for field in ("total_tests", "passed", "failed", "errors"):
            value = report.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"report {field} must be a nonnegative integer")
        total_tests = report["total_tests"]
        passed_tests = report["passed"]
        failed_tests = report["failed"]
        error_tests = report["errors"]
        if total_tests <= 0 or total_tests != (
            passed_tests + failed_tests + error_tests
        ):
            raise ValueError("report test counts are inconsistent")
        pass_rate = report.get("pass_rate")
        if (
            isinstance(pass_rate, bool)
            or not isinstance(pass_rate, (int, float))
            or not 0.0 <= float(pass_rate) <= 1.0
        ):
            raise ValueError("report pass_rate must lie inside [0, 1]")
        expected_pass_rate = passed_tests / total_tests
        if not math.isclose(
            float(pass_rate),
            expected_pass_rate,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("report test counts disagree with pass_rate")
        metrics = report.get("metrics")
        if not isinstance(metrics, dict) or metrics.get("source") != ADAPTER_SOURCE:
            raise ValueError("report lacks the trusted EAGLE adapter identity")
        valid = report["valid"] and not report["timed_out"]
        if valid and (float(score) <= 0 or float(pass_rate) != 1.0):
            raise ValueError("valid report has inconsistent score or pass rate")
        if not valid and float(score) != 0.0:
            raise ValueError("invalid report must have zero accepted score")
        if valid:
            expected_metrics = {
                "adapter_schema_version": 1,
                "candidate_output_suppressed": True,
                "verifier_completed": True,
                "verifier_returncode": 0,
                "result_protocol_records": 1,
                "primary_score_name": "score_tokens_per_s",
                "primary_score_unit": "tokens/s",
                "primary_score_direction": "maximize",
            }
            for name, expected in expected_metrics.items():
                if metrics.get(name) != expected:
                    raise ValueError(f"report metrics.{name} differs")
            metric_score = metrics.get("score_tokens_per_s")
            if (
                isinstance(metric_score, bool)
                or not isinstance(metric_score, (int, float))
                or not math.isclose(
                    float(metric_score),
                    float(score),
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                )
            ):
                raise ValueError("report score differs from raw tokens/s metric")
            for gate in REQUIRED_GATES:
                if metrics.get(gate) != 1.0:
                    raise ValueError(f"report metrics.{gate} must equal 1.0")
        return {
            "status": "PASS" if valid else "FAIL",
            "score": float(score),
            "valid": valid,
            "sha256": None,
        }

    # Retain compatibility with explicitly wrapped verifier records produced by
    # older SForge versions, but require a recognized status.
    status = report.get("status")
    if status not in {"PASS", "FAIL", "ERROR", "PENDING_QUALIFICATION", "passed", "failed"}:
        # Current SForge reports may wrap the verifier result under ``result``.
        nested = report.get("result")
        if isinstance(nested, dict):
            status = nested.get("status", nested.get("valid"))
        if status not in {"PASS", "FAIL", "ERROR", "PENDING_QUALIFICATION", True, False}:
            raise ValueError("report lacks a recognized verifier status")
    return {
        "status": status,
        "score": None,
        "valid": status == "PASS",
        "sha256": None,
    }


def locate_submission(root: Path) -> Path:
    """Resolve exactly one manual submission directory below a task run."""
    direct = root / "submissions/manual-1"
    if direct.is_dir() and not direct.is_symlink():
        return direct
    candidates = [
        path
        for path in root.glob("submissions/manual-*")
        if path.is_dir() and not path.is_symlink()
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one manual submission, found {len(candidates)}")
    return candidates[0]


def validate_private_artifact(
    path: Path,
    *,
    expected_score: float,
    qualification: bool = False,
) -> dict[str, Any]:
    """Validate the root-only private detail packaged by SForge."""
    regular_file(path, PRIVATE_ARTIFACT, MAX_ARCHIVE_BYTES)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("private Judge artifact must have mode 0600")
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        if (
            len(members) != 1
            or members[0].name != PRIVATE_MEMBER
            or not members[0].isfile()
            or stat.S_IMODE(members[0].mode) != 0o600
            or not 0 < members[0].size <= MAX_JSON_BYTES
        ):
            raise ValueError(
                f"private artifact must contain exactly one {PRIVATE_MEMBER}"
            )
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError("private Judge result is unreadable")
        payload = stream.read(MAX_JSON_BYTES + 1)
    if len(payload) != members[0].size:
        raise ValueError("private Judge result is truncated")
    detail = json.loads(
        payload.decode("utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(detail, dict) or not finite_tree(detail):
        raise ValueError("private Judge result must be a finite JSON object")
    score = detail.get("score_tokens_per_s")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isclose(
            float(score),
            expected_score,
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError("private score differs from public structured score")
    if detail.get("status") != "PASS":
        raise ValueError("private Judge result did not pass")
    expected_state = "PENDING_QUALIFICATION" if qualification else "QUALIFIED"
    if detail.get("qualification_state") != expected_state:
        raise ValueError("private Judge result has the wrong qualification state")
    if bool(detail.get("qualification_mode")) is not qualification:
        raise ValueError("private Judge result has the wrong qualification mode")
    gates = detail.get("gates")
    if not isinstance(gates, dict) or any(
        gates.get(name) is not True for name in REQUIRED_GATES
    ):
        raise ValueError("private Judge result lacks an all-pass gate vector")
    cells = detail.get("cell_tokens_per_s")
    if not isinstance(cells, dict) or not cells:
        raise ValueError("private Judge result lacks cell throughput")
    cell_values = list(cells.values())
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in cell_values
    ):
        raise ValueError("private cell throughput is malformed")
    recomputed = math.exp(
        math.fsum(math.log(float(value)) for value in cell_values)
        / len(cell_values)
    )
    if not math.isclose(recomputed, expected_score, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise ValueError("private cell geomean differs from accepted score")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size": path.stat().st_size,
        "member": PRIVATE_MEMBER,
        "member_sha256": hashlib.sha256(payload).hexdigest(),
        "cell_count": len(cells),
    }


def validate(
    submission_root: Path,
    input_archive: Path,
    expected_archive_sha256: str,
    *,
    qualification: bool = False,
) -> dict[str, Any]:
    """Cross-check input, persisted archive, and report identities."""
    if not submission_root.is_dir() or submission_root.is_symlink():
        submission_root = locate_submission(submission_root)
    persisted_archive = submission_root / "submission.tar.gz"
    report_path = submission_root / "report.json"
    input_record = validate_archive(input_archive, expected_archive_sha256)
    persisted_record = validate_archive(persisted_archive, expected_archive_sha256)
    if input_record["size"] != persisted_record["size"]:
        raise ValueError("persisted archive size differs from input")
    report = json_object(report_path, "report.json")
    report_record = validate_report(report)
    report_record["path"] = str(report_path.resolve())
    report_record["sha256"] = sha256(report_path)
    private_record = (
        validate_private_artifact(
            submission_root / PRIVATE_ARTIFACT,
            expected_score=float(report_record.get("score") or 0.0),
            qualification=qualification,
        )
        if report_record.get("status") == "PASS"
        else None
    )
    reward_path = submission_root / "reward.json"
    reward_record = None
    if reward_path.exists():
        reward = json_object(reward_path, "reward.json")
        reward_value = reward.get("reward")
        expected_reward = float(report_record.get("score") or 0.0)
        if (
            isinstance(reward_value, bool)
            or not isinstance(reward_value, (int, float))
            or not math.isclose(
                float(reward_value),
                expected_reward,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
        ):
            raise ValueError("reward.json differs from the accepted report score")
        reward_record = {
            "path": str(reward_path.resolve()),
            "sha256": sha256(reward_path),
            "reward": reward.get("reward"),
        }
    return {
        "schema_version": 1,
        "status": "PASS",
        "input_archive": input_record,
        "persisted_archive": persisted_record,
        "report": report_record,
        "reward": reward_record,
        "private_judge_artifact": private_record,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse standalone evaluation validation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--input-archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate and write or print one attestation."""
    args = parse_args(argv)
    if len(args.expected_archive_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in args.expected_archive_sha256
    ):
        raise ValueError("expected archive SHA-256 must be lowercase hexadecimal")
    result = validate(
        args.submission_root,
        args.input_archive,
        args.expected_archive_sha256,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
