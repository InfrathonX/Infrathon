#!/usr/bin/env python3
"""Run one arbitrary Candidate on the frozen ten-minute public workload.

This trusted runner receives a pre-hosted vLLM URL through the Judge and sends
all model traffic through the audited model-api service.  It has no knowledge
of host GPU indices, vLLM container identity, or the backend transport.
"""

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


TASK_ROOT = Path(__file__).resolve().parents[1]
FROZEN_HIDDEN_RUNNER = (
    TASK_ROOT
    / "runtime/frozen/harbor_run_scripts/"
    "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.2-gpu45/"
    "run-gpu67-v2.2-hidden-candidate-only.py"
)

PUBLIC_SUITE = "public"
PUBLIC_PROFILE = "public-swe192-distinct"
PUBLIC_WORKFLOWS = 192
PUBLIC_CONCURRENCY = 192
PUBLIC_MAX_STEPS = 100
PUBLIC_WARMUP_SEC = 300
PUBLIC_MEASUREMENT_SEC = 600
PUBLIC_CONTROLLER_IMAGE = (
    "harbor-kv/bench-controller:qwen36-swe192-v24-url-v1"
)
PUBLIC_NONCE_SHA256 = (
    "4902baae73726c1a2f372e35e9e104f6bb3db7dc7593822359548fba60e1b00c"
)

PROJECT = "kvbench-qwen36-swe192-v2-1-gpu67-validation"
SANDBOX_OWNER = "qwen36-swe192-v2-1-gpu67-validation"
STACK_TIMEOUT_SEC = 600
QUIESCENCE_TIMEOUT_SEC = 30.0


def _load_hidden_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "kvbench_v22_hidden_for_public_runner", FROZEN_HIDDEN_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot import frozen candidate evaluator: {FROZEN_HIDDEN_RUNNER}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HIDDEN = _load_hidden_runner()
V22 = HIDDEN.V22


class EvaluationCancelled(Exception):
    """A cooperative INT/TERM request that still requires trusted teardown."""


class CandidateInvalid(Exception):
    """The submitted Candidate changed after its tree was pinned."""


def _validate_frozen_public_spec() -> dict[str, Any]:
    spec = V22.BASELINE_SPECS.get(PUBLIC_SUITE)
    if not isinstance(spec, dict) or any(
        (
            spec.get("profile") != PUBLIC_PROFILE,
            spec.get("workload_nonce_sha256") != PUBLIC_NONCE_SHA256,
            spec.get("request_timeout_sec") != 3600,
            V22.V21.EXPECTED_WINDOWS.get(PUBLIC_SUITE)
            != {
                "warmup_sec": PUBLIC_WARMUP_SEC,
                "measurement_sec": PUBLIC_MEASUREMENT_SEC,
            },
        )
    ):
        raise RuntimeError("frozen public workload specification changed")
    return spec


def _extract_public_nonce(source: Path, *, expected_sha256: str) -> str:
    source = source.expanduser()
    if source.is_symlink():
        raise ValueError("--public-workload-nonce-source must be a real file")
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError("--public-workload-nonce-source must be a real file")

    pattern = re.compile(r"(?<![0-9a-f])([0-9a-f]{64})(?![0-9a-f])")
    matches: set[str] = set()
    with source.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            for raw_nonce in pattern.findall(line):
                try:
                    V22.validate_workload_nonce(
                        raw_nonce,
                        expected_sha256=expected_sha256,
                    )
                except ValueError:
                    continue
                matches.add(raw_nonce)
    if len(matches) != 1:
        raise ValueError(
            "nonce source must contain exactly one value matching the frozen "
            "public commitment"
        )
    return matches.pop()


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
    """Return only resources belonging to the shared frozen validation stack."""

    return {
        "compose_container_ids": _container_ids_for_label(
            f"com.docker.compose.project={PROJECT}"
        ),
        "sandbox_container_ids": _container_ids_for_label(
            "org.harbor-kv-scheduler.swe-sandbox.owner=" + SANDBOX_OWNER
        ),
    }


def _residue_is_empty(residue: dict[str, list[str]]) -> bool:
    return not residue["compose_container_ids"] and not residue[
        "sandbox_container_ids"
    ]


def wait_runtime_quiescent(
    timeout: float = QUIESCENCE_TIMEOUT_SEC,
) -> dict[str, list[str]]:
    deadline = time.monotonic() + timeout
    while True:
        residue = runtime_residue()
        if _residue_is_empty(residue) or time.monotonic() >= deadline:
            return residue
        time.sleep(0.25)


def reconcile_runtime(candidate_dir: Path) -> bool:
    """Remove an orphaned stack while the shared measurement lock is held."""

    residue = runtime_residue()
    if _residue_is_empty(residue):
        return False
    V22.run_stack(
        "stop", candidate_dir=candidate_dir, timeout=STACK_TIMEOUT_SEC
    )
    residue = wait_runtime_quiescent()
    if not _residue_is_empty(residue):
        raise RuntimeError("validation runtime remained dirty after reconciliation")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Candidate on the fixed 192-workflow, ten-minute public "
            "diagnostic; never rerun Direct."
        )
    )
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--expected-candidate-tree-sha256", required=True)
    parser.add_argument(
        "--public-workload-nonce-source", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _validate_candidate(candidate_dir: Path, expected_tree: str) -> tuple[Path, str]:
    candidate_dir = candidate_dir.expanduser()
    if candidate_dir.is_symlink():
        raise ValueError("--candidate-dir must be a real directory")
    candidate_dir = candidate_dir.resolve(strict=True)
    if not candidate_dir.is_dir():
        raise ValueError("--candidate-dir must be a real directory")
    launch = candidate_dir / "launch.sh"
    if not launch.is_file() or launch.is_symlink() or not os.access(launch, os.X_OK):
        raise ValueError("candidate must contain an executable regular launch.sh")
    if re.fullmatch(r"[0-9a-f]{64}", expected_tree) is None:
        raise ValueError(
            "--expected-candidate-tree-sha256 must be 64 lowercase hex"
        )
    observed_tree = V22.V21.tree_sha256(candidate_dir)
    if observed_tree != expected_tree:
        raise ValueError("candidate tree does not match the expected digest")
    return candidate_dir, observed_tree


def main() -> int:
    args = parse_args()
    try:
        candidate_dir, observed_tree = _validate_candidate(
            args.candidate_dir, args.expected_candidate_tree_sha256
        )
        public_spec = _validate_frozen_public_spec()
        raw_nonce = _extract_public_nonce(
            args.public_workload_nonce_source,
            expected_sha256=public_spec["workload_nonce_sha256"],
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    try:
        baseline = V22.load_and_validate_baseline(PUBLIC_SUITE)
    except RuntimeError:
        print("trusted public baseline is unavailable", file=sys.stderr)
        return 5

    # Validate the raw value once more at the trust boundary.  Only its
    # commitment is persisted; neither argv nor output contains the secret.
    nonce = V22.validate_workload_nonce(
        raw_nonce,
        expected_sha256=public_spec["workload_nonce_sha256"],
    )
    output = args.output.expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite evidence: {output}")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "gpu67-v2.2-public-candidate-only-vs-frozen-v2.1-direct",
        "status": "running",
        "phase": "created",
        "started_at_utc": V22.utc_now(),
        "completed_at_utc": None,
        "direct_leg_executed": False,
        "candidate_dir": str(candidate_dir),
        "candidate_tree_sha256": observed_tree,
        "preflight_backend_serving_contract_sha256": None,
        "workload_contract": {
            "suite": PUBLIC_SUITE,
            "profile": PUBLIC_PROFILE,
            "workflows": PUBLIC_WORKFLOWS,
            "concurrency": PUBLIC_CONCURRENCY,
            "max_steps": PUBLIC_MAX_STEPS,
            "measurement_warmup_sec": PUBLIC_WARMUP_SEC,
            "measurement_window_sec": PUBLIC_MEASUREMENT_SEC,
            "workload_nonce_sha256": public_spec["workload_nonce_sha256"],
        },
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
            "The frozen public nonce, 192-task assignment, controller, corpus, and backend serving contract are reused.",
            "Prefix-cache state and wall-clock time differ from the earlier Direct leg, so qualification is diagnostic only.",
        ],
    }
    V22.V21.atomic_write(output, report, initial=True)

    descriptor: int | None = None
    lock_acquired = False
    terminal_kind = "infrastructure_error"
    exit_code = 5
    caught_error: BaseException | None = None
    previous_handlers: dict[int, Any] = {}

    def request_cancel(signum: int, _frame: Any) -> None:
        raise EvaluationCancelled(f"received {signal.Signals(signum).name}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_cancel)

        report["phase"] = "lock_wait"
        V22.V21.atomic_write(output, report)
        lock_path = V22.MEASUREMENT_LOCK_PATH
        lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_acquired = True

        report["phase"] = "preflight"
        report["baseline_evidence"] = {
            key: baseline[key]
            for key in (
                "path",
                "file_sha256",
                "workload_nonce_sha256",
                "direct_valid_steps_per_min",
                "backend_serving_contract_sha256",
            )
        }
        report["preflight_backend_serving_contract_sha256"] = (
            baseline["backend_serving_contract_sha256"]
        )
        report["cleanup"]["reconciled_before_start"] = reconcile_runtime(
            candidate_dir
        )
        V22.V21.atomic_write(output, report)

        report["phase"] = "stack_starting"
        V22.V21.atomic_write(output, report)
        # Spell out the same single URL-backed controller used by hidden
        # evaluation; the validation stack rejects every other image tag.
        os.environ["KV_GPU67_CONTROLLER_IMAGE"] = PUBLIC_CONTROLLER_IMAGE
        V22.run_stack(
            "start", candidate_dir=candidate_dir, timeout=STACK_TIMEOUT_SEC
        )
        if V22.V21.tree_sha256(candidate_dir) != observed_tree:
            raise CandidateInvalid("candidate tree changed before the public leg")

        report["phase"] = "measuring"
        V22.V21.atomic_write(output, report)
        candidate = V22.run_candidate_leg(
            suite=PUBLIC_SUITE,
            profile=PUBLIC_PROFILE,
            workload_nonce=nonce,
            request_timeout_sec=public_spec["request_timeout_sec"],
        )
        if V22.V21.tree_sha256(candidate_dir) != observed_tree:
            raise CandidateInvalid("candidate tree changed during the public leg")

        evaluation = V22.candidate_evaluation(
            baseline_bundle=baseline,
            candidate=candidate,
            workload_nonce=nonce,
        )
        report["leg"] = candidate
        report["evaluation"] = evaluation
        report["phase"] = "teardown"
        V22.V21.atomic_write(output, report)
        if evaluation.get("valid") is True:
            terminal_kind = "complete"
            exit_code = 0
        else:
            terminal_kind = "candidate_invalid"
            exit_code = 3
    except BlockingIOError as exc:
        caught_error = exc
        terminal_kind = "resource_busy"
        exit_code = 4
        report["errors"].append("BlockingIOError: measurement lock is already held")
    except CandidateInvalid as exc:
        caught_error = exc
        terminal_kind = "candidate_invalid"
        exit_code = 3
        report["errors"].append(f"CandidateInvalid: {exc}")
    except (EvaluationCancelled, KeyboardInterrupt) as exc:
        caught_error = exc
        terminal_kind = "cancelled"
        exit_code = 130
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    except BaseException as exc:
        caught_error = exc
        terminal_kind = "infrastructure_error"
        exit_code = 5
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        # The supervisor may send another signal, but trusted cleanup gets one
        # bounded opportunity before an external hard kill.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)

        cleanup_error: BaseException | None = None
        cleanup = report["cleanup"]
        if lock_acquired:
            cleanup["attempted"] = True
            try:
                V22.run_stack(
                    "stop",
                    candidate_dir=candidate_dir,
                    timeout=STACK_TIMEOUT_SEC,
                )
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
        if terminal_kind in {"complete", "candidate_invalid"} and cleanup_ok:
            report["status"] = terminal_kind
            report["phase"] = "quiescent"
        elif terminal_kind == "resource_busy":
            report["status"] = "resource_busy"
            report["phase"] = "terminal"
        elif terminal_kind == "cancelled" and cleanup_ok:
            report["status"] = "cancelled_clean"
            report["phase"] = "terminal"
        else:
            report["status"] = (
                "infrastructure_error_clean"
                if cleanup_ok
                else "infrastructure_error_dirty"
            )
            report["phase"] = "terminal"
            exit_code = 5

        report["completed_at_utc"] = V22.utc_now()
        V22.V21.atomic_write(output, report)

        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if descriptor is not None:
            os.close(descriptor)

    if report["status"] in {"complete", "candidate_invalid"}:
        # The full leg (including task identifiers) stays in the 0600 report.
        # Stdout contains only aggregate diagnostics.
        evaluation = report.get("evaluation") or {}
        public_summary = {
            key: evaluation.get(key)
            for key in (
                "valid",
                "qualification_passed",
                "strictly_better_than_v2_1_direct",
                "direct_valid_steps_per_min",
                "candidate_valid_steps_per_min",
                "speedup",
                "change_percent",
                "all_invalid_reasons",
            )
        }
        print(
            json.dumps(
                {
                    "output": str(output),
                    "status": report["status"],
                    "evaluation": public_summary,
                },
                sort_keys=True,
            )
        )
    elif caught_error is not None:
        print(
            f"trusted public runner terminated as {report['status']}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
