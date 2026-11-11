#!/usr/bin/env python3
"""Run one arbitrary v2.2 hidden Candidate against frozen Direct evidence."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any


RUNNER_DIR = Path(__file__).resolve().parent
SOURCE = RUNNER_DIR / "run-gpu67-v2.2-candidate-only.py"


def load_source() -> Any:
    spec = importlib.util.spec_from_file_location("kvbench_v22_candidate", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import v2.2 evaluator: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V22 = load_source()

PROJECT = V22.V21.PROJECT
SANDBOX_OWNER = "qwen36-swe192-v2-1-gpu67-validation"


class EvaluationCancelled(Exception):
    """A cooperative INT/TERM request that must still run trusted teardown."""


def _container_ids_for_label(label: str) -> list[str]:
    completed = subprocess.run(
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"label={label}",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"runtime residue query failed ({completed.returncode}): "
            + completed.stdout[-2000:]
        )
    return [line for line in completed.stdout.splitlines() if line]


def runtime_residue() -> dict[str, list[str]]:
    """Return only resources owned by this fixed validation stack."""

    return {
        "compose_container_ids": _container_ids_for_label(
            f"com.docker.compose.project={PROJECT}"
        ),
        "sandbox_container_ids": _container_ids_for_label(
            "org.harbor-kv-scheduler.swe-sandbox.owner=" + SANDBOX_OWNER
        ),
    }


def _residue_is_empty(residue: dict[str, list[str]]) -> bool:
    return not residue["compose_container_ids"] and not residue["sandbox_container_ids"]


def wait_runtime_quiescent(timeout: float = 30.0) -> dict[str, list[str]]:
    deadline = time.monotonic() + timeout
    while True:
        residue = runtime_residue()
        if _residue_is_empty(residue):
            return residue
        if time.monotonic() >= deadline:
            return residue
        time.sleep(0.25)


def reconcile_runtime(candidate_dir: Path) -> bool:
    """Clean an orphaned prior stack while the measurement lock is held."""

    residue = runtime_residue()
    if _residue_is_empty(residue):
        return False
    V22.run_stack("stop", candidate_dir=candidate_dir)
    residue = wait_runtime_quiescent()
    if not _residue_is_empty(residue):
        raise RuntimeError("validation runtime remained dirty after reconciliation")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hidden Candidate only; never rerun Direct."
    )
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--expected-candidate-tree-sha256", required=True)
    nonce = parser.add_mutually_exclusive_group(required=True)
    nonce.add_argument("--hidden-workload-nonce")
    nonce.add_argument("--hidden-workload-nonce-source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_dir = args.candidate_dir.expanduser().resolve(strict=True)
    if not candidate_dir.is_dir() or candidate_dir.is_symlink():
        raise SystemExit("--candidate-dir must be a real directory")
    launch = candidate_dir / "launch.sh"
    if not launch.is_file() or launch.is_symlink() or not os.access(launch, os.X_OK):
        raise SystemExit("candidate must contain an executable regular launch.sh")

    expected_tree = args.expected_candidate_tree_sha256
    if len(expected_tree) != 64 or any(c not in "0123456789abcdef" for c in expected_tree):
        raise SystemExit("--expected-candidate-tree-sha256 must be 64 lowercase hex")
    observed_tree = V22.V21.tree_sha256(candidate_dir)
    if observed_tree != expected_tree:
        raise SystemExit(f"candidate tree mismatch: {observed_tree} != {expected_tree}")

    hidden_spec = V22.BASELINE_SPECS["hidden"]
    raw_nonce = args.hidden_workload_nonce
    if args.hidden_workload_nonce_source is not None:
        nonce_source = args.hidden_workload_nonce_source.expanduser().resolve(strict=True)
        if not nonce_source.is_file() or nonce_source.is_symlink():
            raise SystemExit("--hidden-workload-nonce-source must be a real file")
        # The secret source is a host-only audit log.  JSON escaping and argv
        # serialization differ between Codex releases, so select by the
        # frozen SHA-256 commitment instead of depending on one command-line
        # rendering.  The raw value never enters MLSys Forge feedback.
        nonce_pattern = re.compile(r"(?<![0-9a-f])([0-9a-f]{64})(?![0-9a-f])")
        matching_nonces: set[str] = set()
        with nonce_source.open(encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                for candidate_nonce in nonce_pattern.findall(line):
                    try:
                        V22.validate_workload_nonce(
                            candidate_nonce,
                            expected_sha256=hidden_spec["workload_nonce_sha256"],
                        )
                    except ValueError:
                        continue
                    matching_nonces.add(candidate_nonce)
        if len(matching_nonces) != 1:
            raise SystemExit(
                "nonce source must contain exactly one value matching the frozen commitment"
            )
        raw_nonce = matching_nonces.pop()
    assert raw_nonce is not None
    nonce = V22.validate_workload_nonce(
        raw_nonce,
        expected_sha256=hidden_spec["workload_nonce_sha256"],
    )
    baseline = V22.load_and_validate_baseline("hidden")
    output = args.output.expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite evidence: {output}")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "gpu67-v2.2-hidden-candidate-only-vs-frozen-v2.1-direct",
        "status": "running",
        "phase": "created",
        "started_at_utc": V22.utc_now(),
        "completed_at_utc": None,
        "direct_leg_executed": False,
        "candidate_dir": str(candidate_dir),
        "candidate_tree_sha256": observed_tree,
        "preflight_backend_serving_contract_sha256": None,
        "baseline_evidence": {
            key: baseline[key]
            for key in (
                "path",
                "file_sha256",
                "workload_nonce_sha256",
                "direct_valid_steps_per_min",
                "backend_serving_contract_sha256",
            )
        },
        "leg": None,
        "evaluation": None,
        "errors": [],
        "cleanup": {
            "attempted": False,
            "stack_stop_succeeded": False,
            "quiescent": False,
            "reconciled_before_start": False,
            "residue": None,
        },
        "limitations": [
            "Direct is frozen prior evidence and was not rerun.",
            "The original hidden workload nonce and stable backend serving contract are reused.",
            "Prefix-cache state and wall-clock time differ from the earlier Direct leg.",
        ],
    }
    V22.V21.atomic_write(output, report, initial=True)

    lock_path = V22.MEASUREMENT_LOCK_PATH
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    stack_touched = False
    lock_acquired = False
    terminal_kind = "failed"
    exit_code = 5
    caught_error: BaseException | None = None
    previous_handlers: dict[int, Any] = {}

    def request_cancel(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        raise EvaluationCancelled(f"received {signal_name}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_cancel)

        report["phase"] = "lock_wait"
        V22.V21.atomic_write(output, report)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_acquired = True
        report["phase"] = "preflight"
        report["preflight_backend_serving_contract_sha256"] = (
            baseline["backend_serving_contract_sha256"]
        )
        report["cleanup"]["reconciled_before_start"] = reconcile_runtime(
            candidate_dir
        )
        V22.V21.atomic_write(output, report)

        stack_touched = True
        report["phase"] = "stack_starting"
        V22.V21.atomic_write(output, report)
        V22.run_stack("start", candidate_dir=candidate_dir)
        if V22.V21.tree_sha256(candidate_dir) != observed_tree:
            raise RuntimeError("candidate tree changed before hidden leg")
        report["phase"] = "measuring"
        V22.V21.atomic_write(output, report)
        candidate = V22.run_candidate_leg(
            suite="hidden",
            profile=hidden_spec["profile"],
            workload_nonce=nonce,
            request_timeout_sec=hidden_spec["request_timeout_sec"],
        )
        if V22.V21.tree_sha256(candidate_dir) != observed_tree:
            raise RuntimeError("candidate tree changed during hidden leg")
        evaluation = V22.candidate_evaluation(
            baseline_bundle=baseline,
            candidate=candidate,
            workload_nonce=nonce,
        )
        report["leg"] = candidate
        report["evaluation"] = evaluation
        report["phase"] = "teardown"
        V22.V21.atomic_write(output, report)
        terminal_kind = "complete"
        exit_code = 0 if evaluation["qualification_passed"] else 3
    except BlockingIOError as exc:
        caught_error = exc
        terminal_kind = "resource_busy"
        exit_code = 4
        report["errors"].append("BlockingIOError: measurement lock is already held")
    except (EvaluationCancelled, KeyboardInterrupt) as exc:
        caught_error = exc
        terminal_kind = "cancelled"
        exit_code = 130
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    except BaseException as exc:
        caught_error = exc
        terminal_kind = "failed"
        exit_code = 5
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        # A second TERM must not interrupt the one cleanup opportunity.  The
        # supervising Judge applies a bounded grace and may KILL afterward.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        cleanup_error: BaseException | None = None
        cleanup = report["cleanup"]
        if lock_acquired:
            cleanup["attempted"] = True
            try:
                V22.run_stack("stop", candidate_dir=candidate_dir)
                cleanup["stack_stop_succeeded"] = True
            except BaseException as exc:
                cleanup_error = exc
                report["errors"].append(
                    f"cleanup: {type(exc).__name__}: {exc}"
                )
        try:
            residue = wait_runtime_quiescent()
            cleanup["residue"] = residue
            cleanup["quiescent"] = _residue_is_empty(residue)
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
            cleanup["residue"] = {"query_error": type(exc).__name__}
            cleanup["quiescent"] = False
            report["errors"].append(
                f"cleanup verification: {type(exc).__name__}: {exc}"
            )

        cleanup_ok = cleanup["quiescent"] is True and cleanup_error is None
        if terminal_kind == "complete" and cleanup_ok:
            report["status"] = "complete"
            report["phase"] = "quiescent"
        elif terminal_kind == "resource_busy":
            report["status"] = "resource_busy"
            report["phase"] = "terminal"
        elif terminal_kind == "cancelled":
            report["status"] = "cancelled_clean" if cleanup_ok else "cancelled_dirty"
            report["phase"] = "terminal"
            if not cleanup_ok:
                exit_code = 5
        else:
            report["status"] = "failed_clean" if cleanup_ok else "failed_dirty"
            report["phase"] = "terminal"
            exit_code = 5
        if terminal_kind == "complete" and not cleanup_ok:
            report["status"] = "failed_dirty"
            report["phase"] = "terminal"
            exit_code = 5
        report["completed_at_utc"] = V22.utc_now()
        V22.V21.atomic_write(output, report)

        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        os.close(descriptor)

    if report["status"] == "complete":
        print(
            json.dumps(
                {"output": str(output), "evaluation": report["evaluation"]},
                sort_keys=True,
            )
        )
    elif caught_error is not None:
        print(
            f"trusted hidden runner terminated as {report['status']}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
