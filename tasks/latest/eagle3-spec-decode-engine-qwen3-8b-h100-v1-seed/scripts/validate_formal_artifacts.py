#!/usr/bin/env python3
"""Validate pre-final, post-final, and standalone evaluation attestations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import sys
from pathlib import Path
from typing import Any

TASK_ID = "eagle3_spec_decode_engine_qwen3_8b_h100_v1"
EVALUATION_MODE = "dual_h100_manual_judge_then_independent_holdout_final"
SINGLE_GPU_EVALUATION_MODE = (
    "single_gpu_lease_manual_judge_then_standalone_final_handoff"
)
EVALUATION_MODES = (EVALUATION_MODE, SINGLE_GPU_EVALUATION_MODE)


# --- single-GPU lease handoff -------------------------------------------------
# Under the single-GPU lease the coordinator names the standalone final Judge
# submission ``final-<12 hex>`` and records the whole final in the source run's
# durable resource lease, so the final report is bound to that lease record
# (CLOSED, this final run id, this archive digest, this report digest) instead
# of the legacy submission-id equality.  The agent's work boundary is the seal
# (admission closed, Work paused, never-resume persisted): a queued Judge may be
# drained afterwards, so the frozen snapshot can legitimately be later than the
# seal, and the Work container is strictly retired (destroy + NVML attestation)
# before the agent exec returns, so the recorded runtime exceeds the snapshot.
SINGLE_GPU_RETIREMENT_WINDOW_SECONDS = 120.0
SINGLE_GPU_MAX_JUDGE_DRAIN_SECONDS = 7200.0
_FINAL_HANDOFF_SUBMISSION_ID = __import__("re").compile(r"^final-[0-9a-f]{12}$")


def _frozen_before_capture(snapshot: dict[str, Any], elapsed: float, runtime: float) -> bool:
    """A whole-container snapshot is taken from the stopped container: every Work
    process stopped at the freeze, and the recorded runtime runs on to the start
    of the capture.  Accept the gap between freeze and capture, but only when the
    record proves nothing ran in it (the same rule as the LMCache task)."""
    freeze = snapshot.get("freeze")
    capture = snapshot.get("capture")
    if (snapshot.get("snapshot_boundary") != "whole_container_frozen"
            or not isinstance(freeze, dict) or not isinstance(capture, dict)
            or freeze.get("all_work_processes_stopped") is not True
            or capture.get("work_container_running_during_capture") is not False):
        return False
    frozen = finite_number(freeze.get("frozen_elapsed_seconds"), "freeze.frozen_elapsed_seconds")
    started = finite_number(capture.get("started_elapsed_seconds"), "capture.started_elapsed_seconds")
    return abs(frozen - elapsed) <= 1.0 and frozen <= started and started + 1.0 >= runtime


def _single_gpu_seal_boundary(snapshot, runtime, budget_seconds, tolerance, finite, error):
    """Validate the seal/snapshot/runtime ordering of a single-GPU run.

    Returns the seal elapsed seconds.  ``finite`` converts a value to a float or
    raises; ``error`` builds the validator's exception type.
    """
    freeze = snapshot.get("freeze")
    if not isinstance(freeze, dict):
        raise error("final snapshot lacks its freeze record")
    seal = finite(freeze.get("seal_requested_elapsed_seconds"))
    elapsed = finite(snapshot.get("snapshot_elapsed_seconds"))
    if abs(seal - float(budget_seconds)) > float(tolerance):
        raise error(
            f"single-GPU seal is outside the fixed-budget boundary: seal={seal}, budget={budget_seconds}"
        )
    if elapsed < seal - 1.0 or elapsed - seal > SINGLE_GPU_MAX_JUDGE_DRAIN_SECONDS:
        raise error(f"final snapshot does not follow the seal within the drain ceiling: seal={seal}, snapshot={elapsed}")
    if float(runtime) < elapsed - 1.0 or float(runtime) - elapsed > SINGLE_GPU_RETIREMENT_WINDOW_SECONDS:
        raise error(f"final snapshot boundary does not bracket the retired Work runtime: snapshot={elapsed}, runtime={runtime}")
    return seal


def _validate_final_handoff_lease(task_root, *, final_eval_id, task_id, expected_archive_sha256, report_sha256, submission_id, error):
    """Bind the persisted final report to the coordinator's durable lease."""
    import json as _json
    if not isinstance(submission_id, str) or _FINAL_HANDOFF_SUBMISSION_ID.fullmatch(submission_id) is None:
        raise error("coordinator final report submission identity is invalid")
    lease_path = Path(task_root) / "resource_lease.json"
    if lease_path.is_symlink() or not lease_path.is_file():
        raise error("durable resource lease record is missing")
    try:
        lease = _json.loads(lease_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise error(f"durable resource lease record is unreadable: {exc}")
    identity = lease.get("identity") if isinstance(lease, dict) else None
    final_judge = lease.get("final_judge") if isinstance(lease, dict) else None
    if (
        not isinstance(lease, dict)
        or lease.get("state") != "CLOSED"
        or lease.get("recovery_policy") != "never_resume"
        or not isinstance(identity, dict)
        or identity.get("task_id") != task_id
        or identity.get("run_id") != Path(task_root).resolve().parent.name
        or not isinstance(final_judge, dict)
        or not isinstance(final_judge.get("request"), dict)
        or not isinstance(final_judge.get("artifacts"), dict)
    ):
        raise error("durable resource lease did not close on this final Judge")
    request = final_judge["request"]
    artifacts = final_judge["artifacts"]
    expected = expected_archive_sha256.strip().lower()
    if (
        request.get("final_run_id") != final_eval_id
        or request.get("source_run_id") != identity.get("run_id")
        or request.get("task_id") != task_id
        or request.get("archive_sha256") != "sha256:" + expected
        or not isinstance(artifacts.get("report"), dict)
        or artifacts["report"].get("sha256") != "sha256:" + report_sha256
        or not isinstance(artifacts.get("submission"), dict)
        or artifacts["submission"].get("sha256") != "sha256:" + expected
    ):
        raise error("durable resource lease does not bind this final report")
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


def sha256(path: Path) -> str:
    """Hash a regular file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(value: str) -> None:
    """Reject non-finite JSON constants."""
    raise ValueError(f"non-finite JSON constant: {value}")


def finite_tree(value: Any) -> bool:
    """Check all numeric leaves for finiteness."""
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


def regular_file(path: Path, label: str, maximum: int) -> Path:
    """Require a bounded non-symlink regular file with one link."""
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or not 0 < metadata.st_size <= maximum
    ):
        raise ValueError(f"{label} violates the regular-file contract")
    return path


def real_directory(path: Path, label: str) -> Path:
    """Require a real directory before resolving its absolute identity."""
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    if expanded.is_symlink() or not expanded.is_dir():
        raise ValueError(f"{label} must be a real directory")
    return expanded.resolve(strict=True)


def json_object(path: Path, label: str) -> dict[str, Any]:
    """Read one bounded finite JSON object."""
    regular_file(path, label, MAX_JSON_BYTES)
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict) or not finite_tree(value):
        raise ValueError(f"{label} must be a finite JSON object")
    return value


def validate_digest(value: str, label: str) -> str:
    """Require a lowercase hexadecimal SHA-256."""
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} is not a SHA-256")
    return value


def artifact(path: Path, label: str, maximum: int = MAX_JSON_BYTES) -> dict[str, Any]:
    """Describe one trusted regular file."""
    regular_file(path, label, maximum)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size": path.stat().st_size,
    }


def finite_number(value: Any, label: str) -> float:
    """Normalize a non-boolean finite numeric field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def nonnegative_integer(value: Any, label: str) -> int:
    """Normalize a nonnegative integer field."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def archive_record(path: Path) -> dict[str, Any]:
    """Validate a candidate archive using the dedicated archive checker."""
    from validate_eval_artifacts import validate_archive

    digest = sha256(regular_file(path, "final_archive.tar.gz", MAX_ARCHIVE_BYTES))
    return validate_archive(path, digest)


def final_version_from_archive(path: Path) -> int:
    """Read the canonical selector without extracting participant files."""
    import tarfile

    with tarfile.open(path, mode="r:gz") as archive:
        matches = [
            member
            for member in archive.getmembers()
            if member.name.removeprefix("./") == "FINAL_VERSION"
        ]
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError("archive must contain exactly one FINAL_VERSION")
        stream = archive.extractfile(matches[0])
        if stream is None:
            raise ValueError("cannot read FINAL_VERSION")
        raw = stream.read(16)
    try:
        value = int(raw.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("FINAL_VERSION is not a canonical integer") from exc
    if not 1 <= value <= 999:
        raise ValueError("FINAL_VERSION is outside 1..999")
    return value


def validate_pre_final(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the Work-session freeze barrier before private final scoring."""
    root = real_directory(args.task_root, "task run root")
    if args.task_id != TASK_ID:
        raise ValueError("task identity differs")
    if getattr(args, "evaluation_mode", EVALUATION_MODE) not in EVALUATION_MODES:
        raise ValueError("unknown evaluation mode")
    if not args.run_id or args.run_id.startswith(("manual-", "candidate-")):
        raise ValueError("agent run ID is missing or reserved")
    if args.budget_seconds < 90 or not 0 < args.interval_seconds < args.budget_seconds:
        raise ValueError("budget or interval contract differs")
    archive_path = root / "final_archive.tar.gz"
    final_archive = archive_record(archive_path)
    required_json = {
        "final_result": root / "final_result.json",
        "final_snapshot": root / "final_snapshot.json",
        "run_history": root / "run_history.json",
    }
    records = {}
    documents = {}
    for name, path in required_json.items():
        documents[name] = json_object(path, path.name)
        records[name] = artifact(path, path.name)
    result = documents["final_result"]
    expected_identity = {
        "agent": getattr(args, "agent", "codex"),
        "task": args.task_id,
        "run_id": args.run_id,
        "model": args.model,
    }
    for field, expected in expected_identity.items():
        if result.get(field) != expected:
            raise ValueError(f"final_result.{field} differs")
    if result.get("timed_out") is not True:
        raise ValueError("fixed-budget Work run must end at its timeout")
    runtime = finite_number(result.get("runtime_seconds"), "runtime_seconds")
    if getattr(args, "evaluation_mode", EVALUATION_MODE) != SINGLE_GPU_EVALUATION_MODE and not args.budget_seconds - 5 <= runtime <= args.budget_seconds + 60:
        raise ValueError("Work runtime does not prove the fixed budget")
    if nonnegative_integer(
        result.get("archive_size_bytes"),
        "archive_size_bytes",
    ) != final_archive["size"]:
        raise ValueError("final archive size differs from final_result")
    agent_submissions = nonnegative_integer(
        result.get("agent_submissions"),
        "agent_submissions",
    )
    if nonnegative_integer(result.get("auto_submissions"), "auto_submissions") != 0:
        raise ValueError("automatic final evaluation must remain disabled")
    if nonnegative_integer(result.get("total_rounds"), "total_rounds") != agent_submissions:
        raise ValueError("total rounds differ from manual submission count")

    snapshot = documents["final_snapshot"]
    elapsed = finite_number(
        snapshot.get("snapshot_elapsed_seconds"),
        "snapshot_elapsed_seconds",
    )
    if getattr(args, "evaluation_mode", EVALUATION_MODE) == SINGLE_GPU_EVALUATION_MODE:
        _single_gpu_seal_boundary(snapshot, runtime, args.budget_seconds, 60.0, lambda v: finite_number(v, "single-GPU boundary"), ValueError)
    else:
        if abs(elapsed - args.budget_seconds) > 60 or (
            elapsed + 1.0 < runtime and not _frozen_before_capture(snapshot, elapsed, runtime)
        ):
            raise ValueError("final snapshot is outside the fixed-budget boundary")
    recorded_archive_digest = snapshot.get("archive_sha256")
    if (
        recorded_archive_digest is not None
        and recorded_archive_digest != final_archive["sha256"]
    ):
        raise ValueError("final snapshot archive digest differs")

    history = documents["run_history"]
    if history.get("run_id") != args.run_id:
        raise ValueError("run history has the wrong run ID")
    history_counts = {
        "agent_submissions": agent_submissions,
        "auto_submissions": 0,
    }
    for field, expected in history_counts.items():
        observed = nonnegative_integer(
            history.get(field),
            f"history.{field}",
        )
        if observed != expected:
            raise ValueError(
                f"history.{field} must equal {expected}, got {observed}"
            )
    if "public_submissions" in history and nonnegative_integer(
        history.get("public_submissions"),
        "history.public_submissions",
    ) != 0:
        raise ValueError("public tests must remain local to Work")
    entries = history.get("entries")
    if not isinstance(entries, list) or len(entries) != agent_submissions:
        raise ValueError("run history length differs from submission count")
    expected_rounds = [
        f"agent-{index}"
        for index in range(1, agent_submissions + 1)
    ]
    observed_rounds = [
        entry.get("round") if isinstance(entry, dict) else None
        for entry in entries
    ]
    if observed_rounds != expected_rounds:
        raise ValueError("run history rounds are not canonical")

    from validate_eval_artifacts import validate

    manual_rounds = []
    for round_id, entry in zip(expected_rounds, entries, strict=True):
        assert isinstance(entry, dict)
        expected_entry = {
            "type": "submission",
            "status": "completed",
            "task_id": args.task_id,
        }
        for field, expected in expected_entry.items():
            if entry.get(field) != expected:
                raise ValueError(
                    f"history {round_id}.{field} must equal {expected!r}"
                )
        submission_id = entry.get("submission_id")
        if not isinstance(submission_id, str) or not submission_id:
            raise ValueError(f"history {round_id} lacks a submission ID")
        submission = root / "submissions" / round_id
        submission_archive = submission / "submission.tar.gz"
        archive_digest = sha256(
            regular_file(
                submission_archive,
                f"{round_id} submission archive",
                MAX_ARCHIVE_BYTES,
            )
        )
        evaluation = validate(
            submission,
            submission_archive,
            archive_digest,
        )
        round_report = json_object(
            submission / "report.json",
            f"{round_id} report",
        )
        if round_report.get("submission_id") != submission_id:
            raise ValueError(f"history {round_id} report identity differs")
        manual_rounds.append(
            {
                "round_id": round_id,
                "submission_id": submission_id,
                "report": evaluation["report"],
                "submission_archive": evaluation["persisted_archive"],
                "private_judge_artifact": evaluation[
                    "private_judge_artifact"
                ],
            }
        )
    return {
        "schema_version": 1,
        "evaluation_mode": getattr(args, "evaluation_mode", EVALUATION_MODE),
        "status": "PASS",
        "phase": "pre",
        "run_id": args.run_id,
        "task_id": args.task_id,
        "model": args.model,
        "budget_seconds": args.budget_seconds,
        "interval_seconds": args.interval_seconds,
        "final_version": final_version_from_archive(archive_path),
        "final_archive": final_archive,
        "manual_rounds": manual_rounds,
        **records,
    }


def locate_final_submission(final_task_root: Path) -> Path:
    """Locate one persisted manual submission below the final task run."""
    direct = final_task_root / "submissions/manual-1"
    if direct.is_dir() and not direct.is_symlink():
        return direct
    matches = [
        path
        for path in final_task_root.glob("submissions/manual-*")
        if path.is_dir() and not path.is_symlink()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one final manual submission, found {len(matches)}")
    return matches[0]


def locate_private_artifact(final_task_root: Path, submission: Path) -> Path | None:
    """Find the run-scoped private artifact archive if SForge persisted one."""
    candidates = [
        submission / "private_artifacts.tar.gz",
        submission / "private_judge_artifacts.tar.gz",
        final_task_root / "private_artifacts.tar.gz",
    ]
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    return None


def validate_final_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Bind the evaluated archive and persisted report to the precheck digest."""
    validate_digest(args.expected_archive_sha256, "expected archive digest")
    task_root = real_directory(args.task_root, "task run root")
    final_task_root = real_directory(
        args.final_task_root,
        "final evaluation task root",
    )
    if args.task_id != TASK_ID:
        raise ValueError("task identity differs")
    if getattr(args, "evaluation_mode", EVALUATION_MODE) not in EVALUATION_MODES:
        raise ValueError("unknown evaluation mode")
    if not args.final_eval_id.startswith(
        ("manual-", "candidate-")
    ) and not args.final_eval_id.endswith("-final"):
        raise ValueError("final evaluation ID is not host-side")
    source_archive = task_root / "final_archive.tar.gz"
    if not source_archive.exists():
        source_archive = task_root / "submission.tar.gz"
    submission = locate_final_submission(final_task_root)
    from validate_eval_artifacts import validate

    evaluation = validate(
        submission,
        source_archive,
        args.expected_archive_sha256,
        qualification=args.phase == "qualify",
    )
    report_path = submission / "report.json"
    report = json_object(report_path, "report.json")
    if getattr(args, "evaluation_mode", EVALUATION_MODE) == SINGLE_GPU_EVALUATION_MODE:
        _validate_final_handoff_lease(
            task_root,
            final_eval_id=args.final_eval_id,
            task_id=args.task_id,
            expected_archive_sha256=args.expected_archive_sha256.removeprefix("sha256:"),
            report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            submission_id=report.get("submission_id"),
            error=ValueError,
        )
    elif report.get("submission_id") != args.final_eval_id:
        raise ValueError("independent final report has the wrong submission ID")
    if evaluation["report"]["status"] != "PASS":
        raise ValueError("independent final evaluation is not a valid pass")
    private_record = evaluation.get("private_judge_artifact")
    if not isinstance(private_record, dict):
        raise ValueError("independent final evaluation lacks private Judge evidence")
    return {
        "schema_version": 1,
        "evaluation_mode": getattr(args, "evaluation_mode", EVALUATION_MODE),
        "status": "PASS",
        "phase": args.phase,
        "final_eval_id": args.final_eval_id,
        "task_id": args.task_id,
        "evaluated_archive": evaluation["persisted_archive"],
        "final_report": artifact(report_path, "report.json"),
        "validated_report": {
            "status": evaluation["report"]["status"],
            "valid": report.get("valid"),
            "score": report.get("score", report.get("reward", 0.0)),
        },
        "private_judge_artifact": private_record,
        "evaluation": evaluation,
    }


def parser() -> argparse.ArgumentParser:
    """Build the phase-specific CLI."""
    root = argparse.ArgumentParser(description=__doc__)
    phases = root.add_subparsers(dest="phase", required=True)
    pre = phases.add_parser("pre")
    pre.add_argument("--task-root", required=True, type=Path)
    pre.add_argument("--run-id", required=True)
    pre.add_argument("--task-id", required=True)
    pre.add_argument("--model", required=True)
    pre.add_argument("--agent", choices=("codex", "claude-code"), default="codex")
    pre.add_argument("--budget-seconds", required=True, type=int)
    pre.add_argument("--interval-seconds", required=True, type=int)
    pre.add_argument(
        "--evaluation-mode",
        choices=EVALUATION_MODES,
        default=EVALUATION_MODE,
        help="scheduling mode recorded by the launcher (legacy dual GPU or single-GPU lease)",
    )
    for name in ("post", "eval", "qualify"):
        final = phases.add_parser(name)
        final.add_argument("--task-root", required=True, type=Path)
        final.add_argument("--final-task-root", required=True, type=Path)
        final.add_argument("--final-eval-id", required=True)
        final.add_argument("--task-id", required=True)
        final.add_argument("--expected-archive-sha256", required=True)
        final.add_argument(
            "--evaluation-mode",
            choices=EVALUATION_MODES,
            default=EVALUATION_MODE,
            help="scheduling mode recorded by the launcher (legacy dual GPU or single-GPU lease)",
        )
    return root


INVALID_FINAL_STATUS = "invalid_final"
INVALID_FINAL_EXIT_CODE = 3


def invalid_final_record(args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    """Describe a rejected final as a durable, machine-readable attestation.

    The record deliberately carries no `evaluated_archive`, `final_report` or
    `private_judge_artifact` binding: nothing downstream may treat it as a
    pass.  It only preserves *why* the fail-closed validator refused, so the
    reason survives the launcher's stdout.
    """
    evaluation_mode = getattr(args, "evaluation_mode", EVALUATION_MODE)
    return {
        "schema_version": 1,
        "evaluation_mode": evaluation_mode,
        "status": INVALID_FINAL_STATUS,
        "phase": args.phase,
        "final_eval_id": args.final_eval_id,
        "task_id": args.task_id,
        "expected_archive_sha256": args.expected_archive_sha256,
        "final_task_root": str(args.final_task_root),
        "reason": str(error) or type(error).__name__,
        "error_type": type(error).__name__,
    }


def main(argv: list[str] | None = None) -> int:
    """Validate one phase and emit its JSON attestation.

    A rejected final (post/eval/qualify) still fails closed: the process exits
    non-zero, but it first prints a complete `invalid_final` record on stdout
    so the launcher can persist `formal-postcheck.json` instead of an empty
    temporary file.
    """
    args = parser().parse_args(argv)
    if args.phase == "pre":
        result = validate_pre_final(args)
    else:
        try:
            result = validate_final_eval(args)
        except Exception as error:  # noqa: BLE001 - every failure is recorded, none is accepted
            record = invalid_final_record(args, error)
            print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
            print(f"error: {record['reason']}", file=sys.stderr)
            return INVALID_FINAL_EXIT_CODE
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
