#!/usr/bin/env python3
"""Trusted EAGLE-3 verifier entry point used by SForge.

This file owns host-requested ``candidate_version`` selection. Interactive
rounds must use the archive's ``FINAL_VERSION``; only root-owned manual and
candidate replay rounds may select another retained ``engine_vNNN``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

JUDGE_ROOT = Path(os.environ.get("EAGLE3_JUDGE_ROOT", "/opt/eagle3_judge"))
EVALUATION_CONTEXT = Path("/tmp/.sforge-evaluation-context.json")
DETAIL_NAME = "eagle3_spec_decode_private.json"
MAX_CONTEXT_BYTES = 16 * 1024
GATES = (
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


def strict_json(
    path: Path,
    label: str,
    *,
    maximum: int = MAX_CONTEXT_BYTES,
    expected_uid: int | None = None,
    exact_mode: int | None = None,
) -> dict[str, Any]:
    """Load one bounded regular JSON object without accepting constants."""
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or not 0 < metadata.st_size <= maximum
        or (expected_uid is not None and metadata.st_uid != expected_uid)
        or (exact_mode is not None and stat.S_IMODE(metadata.st_mode) != exact_mode)
    ):
        raise RuntimeError(f"{label} does not satisfy the file contract")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def load_evaluation_request(
    path: Path = EVALUATION_CONTEXT,
    *,
    expected_uid: int = 0,
) -> tuple[str, str, int | None]:
    """Return profile, round ID, and trusted version override."""
    context = strict_json(
        path,
        "SForge evaluation context",
        expected_uid=expected_uid,
        exact_mode=0o600,
    )
    round_id = context.get("round_id")
    if not isinstance(round_id, str) or not round_id:
        raise RuntimeError("SForge evaluation context lacks round_id")
    candidate_version = context.get("candidate_version")
    if candidate_version is not None and (
        isinstance(candidate_version, bool)
        or not isinstance(candidate_version, int)
        or not 1 <= candidate_version <= 999
    ):
        raise RuntimeError("SForge candidate_version is invalid")
    if round_id.startswith(("auto-", "agent-")):
        if candidate_version is not None:
            raise RuntimeError("candidate_version override is host-side only")
        return "final", round_id, None
    if round_id.startswith(("manual-", "candidate-")):
        return "final", round_id, candidate_version
    raise RuntimeError("SForge round_id has an unknown evaluation profile")


def fallback_request() -> tuple[str, str, int | None]:
    """Support explicit no-Docker smokes when no SForge context exists."""
    profile = os.environ.get("EAGLE3_JUDGE_PROFILE", "anchor")
    if profile not in {"anchor", "final"}:
        raise RuntimeError("EAGLE3_JUDGE_PROFILE must be anchor or final")
    return profile, "host-smoke", None


def locate_spec_root(app_dir: Path) -> Path:
    """Find exactly one submitted spec_lab workspace."""
    choices = (
        app_dir / "workdir/spec_lab",
        app_dir / "environment/workdir/spec_lab",
        app_dir,
    )
    matches = []
    for choice in choices:
        if (
            choice.is_dir()
            and not choice.is_symlink()
            and (choice / "solutions").is_dir()
            and (choice / "FINAL_VERSION").is_file()
        ):
            absolute = choice.absolute()
            if absolute not in matches:
                matches.append(absolute)
    if len(matches) != 1:
        raise RuntimeError(f"expected one spec_lab submission, found {len(matches)}")
    return matches[0]


def process_table() -> dict[int, tuple[int, int]]:
    """Snapshot Linux PID to parent-PID/process-group relationships."""
    table: dict[int, tuple[int, int]] = {}
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return table
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw.rsplit(")", 1)[1].split()
            process_id = int(entry.name)
            table[process_id] = (int(fields[1]), int(fields[2]))
        except (IndexError, OSError, ProcessLookupError, ValueError):
            continue
    return table


def process_tree_groups(root_pid: int) -> set[int]:
    """Return all groups below root, including separately created sessions."""
    table = process_table()
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for process_id, (parent_id, _group_id) in table.items():
            if parent_id in descendants and process_id not in descendants:
                descendants.add(process_id)
                changed = True
    groups = {
        table[process_id][1]
        for process_id in descendants
        if process_id in table
    }
    try:
        groups.add(os.getpgid(root_pid))
    except (OSError, ProcessLookupError):
        pass
    return groups


def signal_groups(groups: set[int], requested: signal.Signals) -> None:
    """Signal a stable snapshot of process groups, newest leaders first."""
    for group in sorted(groups, reverse=True):
        try:
            os.killpg(group, requested)
        except (OSError, ProcessLookupError):
            pass


def terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate and then kill descendants even if they created sessions."""
    groups = process_tree_groups(process.pid)
    signal_groups(groups, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    groups.update(process_tree_groups(process.pid))
    signal_groups(groups, signal.SIGKILL)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def run_controller_command(
    command: list[str],
    *,
    timeout: float,
) -> subprocess.CompletedProcess[bytes]:
    """Run the controller in a process tree that is fully killable on timeout."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process_tree(process)
        raise
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout,
        stderr,
    )


def write_private(path: Path, content: str) -> None:
    """Atomically replace one root-only result file with exact mode 0600."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_reward(log_dir: Path, detail: dict[str, Any], accepted_score: float) -> None:
    """Write mutually consistent SForge reward records atomically enough for a private dir."""
    reward: dict[str, Any] = {
        "schema_version": 1,
        "reward": accepted_score,
        "score_tokens_per_s": accepted_score,
    }
    for name in GATES:
        reward[name] = 1.0 if detail.get("gates", {}).get(name) is True else 0.0
        detail[name] = reward[name]
    detail["reward"] = accepted_score
    detail["score_tokens_per_s"] = accepted_score
    write_private(
        log_dir / "reward.json",
        json.dumps(reward, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    write_private(log_dir / "reward.txt", f"{accepted_score}\n")
    write_private(
        log_dir / DETAIL_NAME,
        json.dumps(detail, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def evaluate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Run the root controller and normalize its private report."""
    has_context = EVALUATION_CONTEXT.exists()
    if has_context:
        profile, round_id, candidate_version = load_evaluation_request()
    else:
        profile, round_id, candidate_version = fallback_request()
    all_anchor = not has_context and os.environ.get("EAGLE3_ALL_ANCHOR") == "1"
    spec_root = locate_spec_root(args.app_dir)
    private_report = args.log_dir / DETAIL_NAME
    command = [
        sys.executable,
        str(args.judge_root / "verifier/controller.py"),
        "--candidate-root",
        str(spec_root),
        "--profile",
        profile,
        "--output",
        str(private_report),
        "--judge-root",
        str(args.judge_root),
        "--spec-root",
        str(spec_root),
    ]
    if candidate_version is not None:
        command.extend(("--candidate-version", str(candidate_version)))
    if all_anchor:
        command.append("--all-anchor")
    completed = run_controller_command(
        command,
        timeout=args.controller_timeout,
    )
    detail = strict_json(
        private_report,
        "private controller report",
        maximum=16 * 1024 * 1024,
    )
    detail["round_id"] = round_id
    detail["evaluation_profile"] = profile
    detail["candidate_version_override"] = candidate_version
    qualification_mode = os.environ.get("EAGLE3_QUALIFICATION_MODE") == "1"
    detail["qualification_mode"] = qualification_mode
    qualified = detail.get("qualification_state") == "QUALIFIED"
    passed = detail.get("status") == "PASS" and completed.returncode == 0
    accepted_score = (
        float(detail.get("score_tokens_per_s", 0.0))
        if passed and (qualified or qualification_mode)
        else 0.0
    )
    if not qualified and not qualification_mode:
        detail["status"] = "PENDING_QUALIFICATION"
        return_code = 0
    else:
        return_code = 0 if passed else 1
    write_reward(args.log_dir, detail, accepted_score)
    return return_code, detail


def failure_detail(message: str) -> dict[str, Any]:
    """Construct a fail-closed record without candidate-controlled detail."""
    return {
        "schema_version": 1,
        "status": "ERROR",
        "valid": False,
        "score_tokens_per_s": 0.0,
        "error": message,
        "gates": {name: False for name in GATES},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the stable verifier command line used by the image metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, default=Path(os.environ.get("APP_DIR", "/app")))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("LOG_DIR", "/logs/verifier")),
    )
    parser.add_argument("--judge-root", type=Path, default=JUDGE_ROOT)
    parser.add_argument(
        "--controller-timeout",
        type=float,
        default=float(os.environ.get("EAGLE3_CONTROLLER_TIMEOUT", "5300")),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Prepare the private result directory and execute fail-closed."""
    args = parse_args(argv)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.chmod(0o700)
    for name in ("reward.json", "reward.txt", DETAIL_NAME):
        path = args.log_dir / name
        if path.exists() or path.is_symlink():
            path.unlink()
    try:
        return_code, _detail = evaluate(args)
        return return_code
    except Exception as exc:
        detail = failure_detail(f"{type(exc).__name__}: {exc}")
        write_reward(args.log_dir, detail, 0.0)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
