#!/usr/bin/env python3
"""Run only the v2.2 Oracle Candidate against frozen v2.1 Direct evidence.

This is an author-side screening runner, not a newly paired A/B measurement.
It deliberately never invokes a Baseline leg.  It reuses the frozen v2.1
benchmark stack and validates the new Candidate against the exact public and
hidden Direct records for the same stable backend serving contract.  A Direct
record must carry a well-formed controller-config commitment, and the Candidate
must match the selected record exactly; no historical per-suite controller
digest is compiled into this runner.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


RUNNER_DIR = Path(__file__).resolve().parent
BENCH_ROOT = RUNNER_DIR.parents[1]
V21_RUNNER_DIR = (
    BENCH_ROOT
    / "harbor_run_scripts"
    / "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45"
)
V21_RUNNER_PATH = V21_RUNNER_DIR / "run-gpu67-v2.1-pair.py"
V21_STACK = V21_RUNNER_DIR / "gpu67-v2.1-validation-stack.sh"
DEFAULT_CANDIDATE_DIR = (
    BENCH_ROOT
    / "harbor_tasks"
    / "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.2-gpu45"
    / "solution"
    / "reference_gateway"
)
DEFAULT_OUTPUT_ROOT = (
    Path.home()
    / ".cache"
    / "harbor-kv-qwen36-swe192-v2-2-gpu67-results"
)
BASELINE_PATH_ENVS = {
    "public": "KVBENCH_PUBLIC_BASELINE_PATH",
    "hidden": "KVBENCH_HIDDEN_BASELINE_PATH",
}

EXPECTED_CANDIDATE_TREE_SHA256 = (
    "b8f92b46bcd25529c19d20354be94d620b632e2bce209909f8ae666a71de0e75"
)
EXPECTED_RELEASE_ID = "qwen36-27b-swe192-v2-1-gpu45"

BASELINE_SPECS: dict[str, dict[str, Any]] = {
    "public": {
        "profile": "public-swe192-distinct",
        "workload_nonce_sha256": (
            "4902baae73726c1a2f372e35e9e104f6bb3db7dc7593822359548fba60e1b00c"
        ),
        "request_timeout_sec": 3600,
    },
    "hidden": {
        "profile": "hidden-pressure-192",
        "workload_nonce_sha256": (
            "c525ae31a2bb1d3ed0232c345f9fc978fd52574d415aaa2777bead17f2fac20a"
        ),
        "request_timeout_sec": 6000,
    },
}


def load_v21_runner() -> Any:
    spec = importlib.util.spec_from_file_location("kvbench_v21_pair", V21_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen v2.1 evaluator: {V21_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V21 = load_v21_runner()
MEASUREMENT_LOCK_PATH = V21.MEASUREMENT_LOCK_PATH
BACKEND_SERVING_CONTRACT_SHA256 = V21.BACKEND_SERVING_CONTRACT_SHA256
ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256 = (
    V21.ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_workload_nonce(raw: str, *, expected_sha256: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", raw) is None:
        raise ValueError("workload nonce must be exactly 64 lowercase hexadecimal characters")
    observed = hashlib.sha256(raw.encode()).hexdigest()
    if observed != expected_sha256:
        raise ValueError(
            f"workload nonce digest mismatch: {observed} != {expected_sha256}"
        )
    return raw


def ordered_task_ids(leg: dict[str, Any]) -> list[str]:
    workflows = leg.get("workflows")
    if not isinstance(workflows, list):
        return []
    return [
        row.get("task_id")
        for row in workflows
        if isinstance(row, dict) and isinstance(row.get("task_id"), str)
    ]


def configured_baseline_path(suite: str) -> Path:
    environment_name = BASELINE_PATH_ENVS[suite]
    raw = os.environ.get(environment_name, "").strip()
    if not raw:
        raise RuntimeError(f"{environment_name} is required")
    path = Path(raw)
    if not path.is_absolute() or path != Path(os.path.normpath(raw)):
        raise RuntimeError(f"{environment_name} must be a normalized absolute path")
    return path


def load_and_validate_baseline(suite: str) -> dict[str, Any]:
    """Validate selected Direct evidence without pinning a historical controller."""

    spec = BASELINE_SPECS[suite]
    path = configured_baseline_path(suite)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"frozen {suite} baseline is unavailable or unsafe: {path}")
    observed_file_sha256 = file_sha256(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("status") != "complete"
        or report.get("suite") != suite
        or report.get("profile") != spec["profile"]
        or report.get("workload_nonce_sha256") != spec["workload_nonce_sha256"]
    ):
        raise RuntimeError(f"frozen {suite} outer evidence identity is invalid")
    legs = report.get("legs")
    baseline = legs.get("baseline") if isinstance(legs, dict) else None
    if not isinstance(baseline, dict):
        raise RuntimeError(f"frozen {suite} evidence has no Direct leg")
    baseline = dict(baseline)
    try:
        baseline["model_provenance"] = V21.project_stable_model_provenance(
            baseline.get("model_provenance")
        )
    except ValueError as exc:
        raise RuntimeError(
            f"frozen {suite} Direct backend contract is invalid"
        ) from exc
    reasons = V21.leg_validation_reasons(
        baseline,
        suite=suite,
        profile=spec["profile"],
        mode="baseline",
    )
    if reasons:
        raise RuntimeError(f"frozen {suite} Direct leg is invalid: {reasons}")
    baseline_provenance = baseline["model_provenance"]
    controller_config_sha256 = baseline.get("controller_config_sha256")
    if (
        not isinstance(controller_config_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", controller_config_sha256) is None
    ):
        raise RuntimeError(
            f"frozen {suite} Direct controller-config commitment is invalid"
        )
    direct_rate_raw = baseline.get("valid_steps_per_min")
    direct_rate = (
        float(direct_rate_raw)
        if isinstance(direct_rate_raw, (int, float))
        and not isinstance(direct_rate_raw, bool)
        and math.isfinite(float(direct_rate_raw))
        and float(direct_rate_raw) > 0.0
        else None
    )
    if any(
        (
            baseline.get("workload_nonce_sha256") != spec["workload_nonce_sha256"],
            baseline.get("release_id") != EXPECTED_RELEASE_ID,
            baseline_provenance.get("backend_serving_contract_sha256")
            not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256,
            direct_rate is None,
            len(ordered_task_ids(baseline)) != 192,
            len(set(ordered_task_ids(baseline))) != 192,
        )
    ):
        raise RuntimeError(f"frozen {suite} Direct identity/rate contract changed")
    return {
        "suite": suite,
        "profile": spec["profile"],
        "path": str(path),
        "file_sha256": observed_file_sha256,
        "workload_nonce_sha256": spec["workload_nonce_sha256"],
        "direct_valid_steps_per_min": direct_rate,
        # Report the revision the anchor was measured under, not the revision
        # this checkout happens to ship.
        "backend_serving_contract_sha256": baseline_provenance[
            "backend_serving_contract_sha256"
        ],
        "report": report,
        "leg": baseline,
    }


def candidate_evaluation(
    *,
    baseline_bundle: dict[str, Any],
    candidate: dict[str, Any],
    workload_nonce: str,
) -> dict[str, Any]:
    suite = baseline_bundle["suite"]
    profile = baseline_bundle["profile"]
    baseline = baseline_bundle["leg"]
    candidate = dict(candidate)
    try:
        candidate["model_provenance"] = V21.project_stable_model_provenance(
            candidate.get("model_provenance")
        )
    except ValueError:
        # The normal validator below emits a closed reason code.
        pass
    candidate_reasons = V21.leg_validation_reasons(
        candidate,
        suite=suite,
        profile=profile,
        mode="candidate",
    )
    pair_reasons = V21.pair_validation_reasons(
        baseline,
        candidate,
        workload_nonce=workload_nonce,
    )

    identity_reasons: list[str] = []
    baseline_provenance = baseline.get("model_provenance")
    candidate_provenance = candidate.get("model_provenance")
    baseline_provenance_sha256 = V21.canonical_sha256(baseline_provenance)
    candidate_provenance_sha256 = V21.canonical_sha256(candidate_provenance)
    baseline_contract = V21.backend_serving_contract_sha256(
        baseline_provenance
    )
    candidate_contract = V21.backend_serving_contract_sha256(
        candidate_provenance
    )
    if (
        baseline_contract not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
        or candidate_contract not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
    ):
        identity_reasons.append("backend_serving_contract_mismatch")
    if candidate.get("controller_config_sha256") != baseline.get(
        "controller_config_sha256"
    ):
        identity_reasons.append("controller_config_mismatch")
    if ordered_task_ids(candidate) != ordered_task_ids(baseline):
        identity_reasons.append("ordered_task_assignment_mismatch")

    direct_rate = float(baseline_bundle["direct_valid_steps_per_min"])
    candidate_rate_raw = candidate.get("valid_steps_per_min")
    candidate_rate = (
        float(candidate_rate_raw)
        if isinstance(candidate_rate_raw, (int, float))
        and not isinstance(candidate_rate_raw, bool)
        and math.isfinite(float(candidate_rate_raw))
        else math.nan
    )
    strictly_improved = math.isfinite(candidate_rate) and candidate_rate > direct_rate
    all_reasons = sorted(set(candidate_reasons + pair_reasons + identity_reasons))
    return {
        "suite": suite,
        "profile": profile,
        "direct_valid_steps_per_min": direct_rate,
        "candidate_valid_steps_per_min": candidate_rate,
        "speedup": candidate_rate / direct_rate if math.isfinite(candidate_rate) else None,
        "change_percent": (
            (candidate_rate / direct_rate - 1.0) * 100.0
            if math.isfinite(candidate_rate)
            else None
        ),
        "strictly_better_than_v2_1_direct": strictly_improved,
        "candidate_invalid_reasons": candidate_reasons,
        "pair_identity_invalid_reasons": pair_reasons,
        "additional_identity_invalid_reasons": identity_reasons,
        "all_invalid_reasons": all_reasons,
        "baseline_model_provenance_sha256": baseline_provenance_sha256,
        "candidate_model_provenance_sha256": candidate_provenance_sha256,
        "backend_serving_contract_sha256": candidate_contract,
        "valid": not all_reasons,
        "qualification_passed": not all_reasons and strictly_improved,
    }


def run_stack(action: str, *, candidate_dir: Path, timeout: int = 600) -> str:
    environment = {
        **os.environ,
        "KV_AUTHOR_REFERENCE_GATEWAY_DIR": str(candidate_dir),
        "KV_GPU67_V21_OWNER_TOKEN": "gpu67-v2-2-candidate-only",
    }
    completed = subprocess.run(
        [str(V21_STACK), action],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"v2.1 frozen stack {action} failed ({completed.returncode})\n"
            + completed.stdout[-12000:]
        )
    return completed.stdout


def run_candidate_leg(
    *,
    suite: str,
    profile: str,
    workload_nonce: str,
    request_timeout_sec: int,
) -> dict[str, Any]:
    controller = V21.service_container("bench-controller")
    return V21.controller_run(
        controller,
        mode="candidate",
        nonce=workload_nonce,
        timeout_sec=request_timeout_sec,
        suite=suite,
        profile=profile,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v2.2 Oracle Candidate only; never rerun Direct."
    )
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--public-workload-nonce", required=True)
    parser.add_argument("--hidden-workload-nonce", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / stamp / "v2.2-oracle-candidate-only.json"


def main() -> int:
    args = parse_args()
    candidate_dir = args.candidate_dir.expanduser().resolve(strict=True)
    if not candidate_dir.is_dir() or candidate_dir.is_symlink():
        raise SystemExit("--candidate-dir must be a real directory")
    candidate_tree_sha256 = V21.tree_sha256(candidate_dir)
    if candidate_tree_sha256 != EXPECTED_CANDIDATE_TREE_SHA256:
        raise SystemExit(
            "candidate is not the pinned v2.2 old Oracle tree: "
            f"{candidate_tree_sha256}"
        )

    baselines = {
        suite: load_and_validate_baseline(suite) for suite in ("public", "hidden")
    }
    nonces = {
        "public": validate_workload_nonce(
            args.public_workload_nonce,
            expected_sha256=BASELINE_SPECS["public"]["workload_nonce_sha256"],
        ),
        "hidden": validate_workload_nonce(
            args.hidden_workload_nonce,
            expected_sha256=BASELINE_SPECS["hidden"]["workload_nonce_sha256"],
        ),
    }
    output = (args.output or default_output_path()).expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite evidence: {output}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "gpu67-v2.2-oracle-candidate-only-vs-frozen-v2.1-direct",
        "qualification": (
            "author-screening-candidate-only-same-nonce-same-backend-contract"
        ),
        "status": "running",
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "direct_leg_executed": False,
        "candidate_tree_sha256": candidate_tree_sha256,
        "candidate_dir": str(candidate_dir),
        "expected_backend_serving_contract_sha256": (
            BACKEND_SERVING_CONTRACT_SHA256
        ),
        "preflight_backend_serving_contract_sha256": None,
        "baseline_evidence": {
            suite: {
                key: bundle[key]
                for key in (
                    "path",
                    "file_sha256",
                    "workload_nonce_sha256",
                    "direct_valid_steps_per_min",
                    "backend_serving_contract_sha256",
                )
            }
            for suite, bundle in baselines.items()
        },
        "legs": {},
        "evaluations": {},
        "overall": None,
        "errors": [],
        "limitations": [
            "Direct is frozen prior evidence and was not rerun.",
            "The workload nonce, controller, corpus, task assignment, and stable backend serving contract are required to match.",
            "The live prefix-cache state and time-of-day are not reset to the earlier Direct starting state, so this remains screening rather than a fresh paired A/B result.",
        ],
    }
    V21.atomic_write(output, report, initial=True)

    lock_path = MEASUREMENT_LOCK_PATH
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_descriptor = os.open(
        lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600
    )
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline_contracts = {
            bundle["backend_serving_contract_sha256"]
            for bundle in baselines.values()
        }
        if baseline_contracts != {BACKEND_SERVING_CONTRACT_SHA256}:
            raise RuntimeError(
                "configured Direct evidence uses an unexpected backend contract"
            )
        report["preflight_backend_serving_contract_sha256"] = (
            BACKEND_SERVING_CONTRACT_SHA256
        )
        V21.atomic_write(output, report)
        for suite in ("public", "hidden"):
            spec = BASELINE_SPECS[suite]
            stack_touched = False
            try:
                stack_touched = True
                run_stack("start", candidate_dir=candidate_dir)
                if V21.tree_sha256(candidate_dir) != candidate_tree_sha256:
                    raise RuntimeError(f"candidate tree changed before {suite} leg")
                candidate = run_candidate_leg(
                    suite=suite,
                    profile=spec["profile"],
                    workload_nonce=nonces[suite],
                    request_timeout_sec=spec["request_timeout_sec"],
                )
                if V21.tree_sha256(candidate_dir) != candidate_tree_sha256:
                    raise RuntimeError(f"candidate tree changed during {suite} leg")
                evaluation = candidate_evaluation(
                    baseline_bundle=baselines[suite],
                    candidate=candidate,
                    workload_nonce=nonces[suite],
                )
                report["legs"][suite] = candidate
                report["evaluations"][suite] = evaluation
                V21.atomic_write(output, report)
            finally:
                if stack_touched:
                    try:
                        run_stack("stop", candidate_dir=candidate_dir)
                    except Exception as cleanup_error:
                        report["errors"].append(
                            f"{suite} cleanup: {type(cleanup_error).__name__}: {cleanup_error}"
                        )
                        V21.atomic_write(output, report)

        suite_passes = {
            suite: bool(report["evaluations"].get(suite, {}).get("qualification_passed"))
            for suite in ("public", "hidden")
        }
        clean_shutdown = not report["errors"]
        gate_passed = all(suite_passes.values()) and clean_shutdown
        report["overall"] = {
            "public_strictly_better_and_valid": suite_passes["public"],
            "hidden_strictly_better_and_valid": suite_passes["hidden"],
            "both_suites_strictly_better_and_valid": all(suite_passes.values()),
            "stack_cleanup_clean": clean_shutdown,
            "terra_max_two_hour_launch_gate_passed": gate_passed,
        }
        report["status"] = "complete"
        report["completed_at_utc"] = utc_now()
        V21.atomic_write(output, report)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "evaluations": report["evaluations"],
                    "overall": report["overall"],
                },
                sort_keys=True,
            )
        )
        return 0 if gate_passed else 3
    except Exception as exc:
        report["status"] = "failed"
        report["completed_at_utc"] = utc_now()
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        V21.atomic_write(output, report)
        raise
    finally:
        os.close(lock_descriptor)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BlockingIOError:
        print(
            f"measurement lock is already held: {MEASUREMENT_LOCK_PATH}",
            file=sys.stderr,
        )
        raise SystemExit(4)
