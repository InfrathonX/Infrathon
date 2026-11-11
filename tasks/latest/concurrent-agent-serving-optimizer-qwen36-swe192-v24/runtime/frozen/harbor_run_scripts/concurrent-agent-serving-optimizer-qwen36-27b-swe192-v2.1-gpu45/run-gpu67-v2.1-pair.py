#!/usr/bin/env python3
"""Run a Direct/Oracle v2.1 distinct-SWE-192 pair on the configured backend."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any


RUNNER_DIR = Path(__file__).resolve().parent
STACK = RUNNER_DIR / "gpu67-v2.1-validation-stack.sh"
PROJECT = "kvbench-qwen36-swe192-v2-1-gpu67-validation"
MEASUREMENT_LOCK_PATH = Path(
    os.environ.get(
        "KVBENCH_MEASUREMENT_LOCK_PATH",
        "/tmp/kvbench-qwen36-swe192-benchmark-measurement.lock",
    )
)
# Revision v2.5 states the backend requirement as a capability predicate
# instead of an "NVIDIA H100 NVL" name whitelist.  Both revisions describe the
# same serving system (identical model, runtime image, TP, KV and scheduler
# fields), so evidence recorded under either one pairs; the launcher preflight
# re-derives that equivalence from the contract files before a run starts.
BACKEND_SERVING_CONTRACT_SHA256 = (
    "1cf6d2e6ff7af8448d6b7735590621d083d6a5071252bc60b40c9756ffe328c4"
)
LEGACY_BACKEND_SERVING_CONTRACT_SHA256 = (
    "44e4a3408b298a5fdc53d7bb84f0739a22f047da1974212754a98b5755777ba8"
)
PREVIOUS_BACKEND_SERVING_CONTRACT_SHA256 = (
    "18d27d97c736f90e1d74822e80ce1f212685e26627b31a08e47569a1457f52c6"
)
ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256 = frozenset(
    {
        BACKEND_SERVING_CONTRACT_SHA256,
        LEGACY_BACKEND_SERVING_CONTRACT_SHA256,
        PREVIOUS_BACKEND_SERVING_CONTRACT_SHA256,
    }
)
LEGACY_MODEL_PROVENANCE = {
    "profile_id": "qwen3.6-27b-6a9e13bd-h100-nvl-tp2-kv30g-v2",
    "profile_sha256": (
        "7289ed315324b9b5165d4d680efb93004c59e0d34f0f98ef9d85406ed5a9153a"
    ),
    "model_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    "runtime_image_id": (
        "sha256:1d8369b9e3c91aa8cba5c7aeeef26e3c4afd455e5d2ce9aebe6f30f6dd9344bd"
    ),
    "tensor_parallel_size": 2,
    "kv_cache_dtype": "fp8",
}
STABLE_MODEL_PROVENANCE_FIELDS = (
    "profile_id",
    "profile_sha256",
    "model_revision",
    "runtime_image_id",
    "tensor_parallel_size",
    "kv_cache_dtype",
)
EXPECTED_RELEASE_ID = "qwen36-27b-swe192-v2-1-gpu45"
EXPECTED_ORACLE_TREE_SHA256 = (
    "b8f92b46bcd25529c19d20354be94d620b632e2bce209909f8ae666a71de0e75"
)
EXPECTED_WINDOWS = {
    "public": {"warmup_sec": 300, "measurement_sec": 600},
    "hidden": {"warmup_sec": 300, "measurement_sec": 1800},
}
REMOTE_REQUEST = r"""
import json
import pathlib
import sys
import urllib.request

path, timeout = sys.argv[1:]
raw_payload = sys.stdin.buffer.read()
token = pathlib.Path("/run/kvbench-admin/token").read_text().strip()
request = urllib.request.Request(
    "http://127.0.0.1:7000" + path,
    data=raw_payload,
    method="POST",
    headers={
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    },
)
with urllib.request.urlopen(request, timeout=int(timeout)) as response:
    sys.stdout.buffer.write(response.read())
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(
    command: list[str],
    *,
    timeout: int = 600,
    input_text: str | None = None,
    sensitive_values: tuple[str, ...] = (),
) -> str:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        rendered_command = " ".join(command)
        rendered_output = completed.stdout[-12000:]
        for value in sensitive_values:
            if value:
                rendered_command = rendered_command.replace(value, "<redacted>")
                rendered_output = rendered_output.replace(value, "<redacted>")
        raise RuntimeError(
            f"command failed ({completed.returncode}): {rendered_command}\n"
            + rendered_output
        )
    return completed.stdout


def service_container(service: str, *, running: bool = True) -> str:
    command = [
        "docker",
        "ps",
        *( [] if running else ["--all"] ),
        "--filter",
        f"label=com.docker.compose.project={PROJECT}",
        "--filter",
        f"label=com.docker.compose.service={service}",
        "--format",
        "{{.ID}}",
    ]
    containers = [line for line in run(command).splitlines() if line]
    if len(containers) != 1:
        raise RuntimeError(
            f"expected exactly one {service} container, observed {containers}"
        )
    return containers[0]


def controller_run(
    controller: str,
    *,
    mode: str,
    nonce: str,
    timeout_sec: int,
    suite: str,
    profile: str,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "suite": suite,
            "profile": profile,
            "mode": mode,
            "workload_nonce": nonce,
        },
        separators=(",", ":"),
    )
    raw = run(
        [
            "docker",
            "exec",
            "-i",
            controller,
            "python3",
            "-c",
            REMOTE_REQUEST,
            "/admin/run",
            str(timeout_sec),
        ],
        timeout=timeout_sec + 60,
        input_text=payload,
        sensitive_values=(nonce,),
    )
    response = json.loads(raw)
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"{mode} response has no result")
    return result


def atomic_write(path: Path, payload: dict[str, Any], *, initial: bool = False) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if initial and (path.exists() or path.is_symlink()):
        raise RuntimeError(f"refusing to overwrite evidence: {path}")
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str | None:
    if not isinstance(value, dict) or not value:
        return None
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def project_stable_model_provenance(value: Any) -> dict[str, Any]:
    """Drop legacy boot identity and add the topology-free contract digest."""

    if not isinstance(value, dict):
        raise ValueError("model provenance is not an object")
    projected = {field: value.get(field) for field in STABLE_MODEL_PROVENANCE_FIELDS}
    if (
        not isinstance(projected["profile_id"], str)
        or not projected["profile_id"]
        or not isinstance(projected["profile_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", projected["profile_sha256"]) is None
        or not isinstance(projected["model_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", projected["model_revision"]) is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(projected["runtime_image_id"])
        )
        is None
        or projected["tensor_parallel_size"] != 2
        or projected["kv_cache_dtype"] != "fp8"
    ):
        raise ValueError("model provenance metadata is invalid")
    digest = value.get("backend_serving_contract_sha256")
    if digest is not None and digest not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256:
        raise ValueError("backend serving contract digest is unexpected")
    if digest is None:
        if projected != LEGACY_MODEL_PROVENANCE:
            raise ValueError("legacy model provenance cannot be upgraded safely")
        digest = LEGACY_BACKEND_SERVING_CONTRACT_SHA256
    projected["backend_serving_contract_sha256"] = digest
    return projected


def backend_serving_contract_sha256(value: Any) -> str | None:
    try:
        projected = project_stable_model_provenance(value)
    except ValueError:
        return None
    return projected["backend_serving_contract_sha256"]


def fixed_window_backend_audit_valid(audit: Any) -> bool:
    """Accept only strict traffic plus trusted hard-cutoff right-censoring."""

    count_fields = (
        "client_requests",
        "audited_backend_requests",
        "strict_client_requests",
        "cutoff_client_requests",
        "cutoff_backend_admitted",
        "cutoff_backend_unadmitted",
        "extra_backend_requests",
        "missing_backend_requests",
    )
    if (
        not isinstance(audit, dict)
        or audit.get("passed") is not True
        or audit.get("window_censored") is not True
        or audit.get("mismatches") != []
        or any(
            not isinstance(audit.get(key), int)
            or isinstance(audit.get(key), bool)
            or audit[key] < 0
            for key in count_fields
        )
        or audit.get("extra_backend_requests") != 0
    ):
        return False

    # A request atomically frozen at the trusted hard cutoff may still be
    # queued inside the candidate gateway and therefore have no backend
    # record. This is valid right-censoring, not a missing strict request.
    return bool(
        audit["missing_backend_requests"] == audit["cutoff_backend_unadmitted"]
        and audit["strict_client_requests"] + audit["cutoff_client_requests"]
        == audit["client_requests"]
        and audit["cutoff_backend_admitted"] + audit["cutoff_backend_unadmitted"]
        == audit["cutoff_client_requests"]
        and audit["strict_client_requests"] + audit["cutoff_backend_admitted"]
        == audit["audited_backend_requests"]
    )


def leg_validation_reasons(
    result: dict[str, Any],
    *,
    suite: str,
    profile: str,
    mode: str,
) -> list[str]:
    """Validate the complete v2.1 fixed-window evidence, not only its rate."""

    reasons: list[str] = []
    expected = EXPECTED_WINDOWS[suite]
    expected_warmup = expected["warmup_sec"]
    expected_measurement = expected["measurement_sec"]

    if result.get("release_id") != EXPECTED_RELEASE_ID:
        reasons.append("release_id_mismatch")
    if result.get("profile") != profile or result.get("mode") != mode:
        reasons.append("leg_identity_mismatch")
    if result.get("infrastructure_failure") is not None:
        reasons.append("infrastructure_failure")
    if result.get("candidate_failure") is not None:
        reasons.append("candidate_failure")

    config = result.get("profile_config")
    if not isinstance(config, dict) or any(
        (
            config.get("workflows") != 192,
            config.get("concurrency") != 192,
            config.get("max_steps") != 100,
            config.get("measurement_warmup_sec") != expected_warmup,
            config.get("measurement_window_sec") != expected_measurement,
        )
    ):
        reasons.append("profile_shape_mismatch")

    workflows = result.get("workflows")
    if not isinstance(workflows, list) or len(workflows) != 192:
        reasons.append("workflow_count_mismatch")
    else:
        task_ids = [row.get("task_id") for row in workflows if isinstance(row, dict)]
        run_ids = [row.get("run_id") for row in workflows if isinstance(row, dict)]
        if len(task_ids) != 192 or len(set(task_ids)) != 192 or any(
            not isinstance(value, str) or not value for value in task_ids
        ):
            reasons.append("task_ids_not_192_distinct")
        if len(run_ids) != 192 or len(set(run_ids)) != 192 or any(
            not isinstance(value, str) or not value for value in run_ids
        ):
            reasons.append("run_ids_not_192_distinct")
        workflow_accounting_valid = all(
            isinstance(row, dict)
            and isinstance(row.get("total_executed_steps"), int)
            and not isinstance(row.get("total_executed_steps"), bool)
            and 0 <= row["total_executed_steps"] <= 100
            and isinstance(row.get("valid_steps"), int)
            and not isinstance(row.get("valid_steps"), bool)
            and 0 <= row["valid_steps"] <= row["total_executed_steps"]
            for row in workflows
        )
        if not workflow_accounting_valid:
            reasons.append("per_workflow_step_ceiling_invalid")
        elif (
            sum(row["total_executed_steps"] for row in workflows)
            != result.get("total_executed_steps")
            or sum(row["valid_steps"] for row in workflows)
            != result.get("valid_steps")
        ):
            reasons.append("workflow_step_totals_mismatch")

    if not fixed_window_backend_audit_valid(result.get("backend_audit")):
        reasons.append("backend_audit_invalid")

    warmup_audit = result.get("warmup_audit")
    if (
        not isinstance(warmup_audit, dict)
        or warmup_audit.get("passed") is not True
        or warmup_audit.get("mismatches") != []
        or warmup_audit.get("extra_backend_requests") != 0
        or warmup_audit.get("missing_backend_requests") != 0
    ):
        reasons.append("warmup_audit_invalid")

    window = result.get("measurement_window")
    if not isinstance(window, dict):
        reasons.append("measurement_window_missing")
    else:
        if any(
            (
                window.get("configured") is not True,
                window.get("deadline_reached") is not True,
                window.get("control_passed") is not True,
                window.get("audit_passed") is not True,
                window.get("backend_quiescent_after_cutoff") is not True,
                window.get("passed") is not True,
                window.get("errors") != [],
                window.get("termination") != "trusted_fixed_window",
                window.get("duration_sec") != expected_measurement,
                window.get("warmup_duration_sec") != expected_warmup,
                window.get("semantic_evaluated_workflows") != 0,
            )
        ):
            reasons.append("measurement_window_contract_invalid")
        truncated = window.get("truncated_workflows")
        if (
            not isinstance(truncated, int)
            or isinstance(truncated, bool)
            or not 0 <= truncated <= 192
        ):
            reasons.append("truncated_workflow_count_invalid")

        partition = (
            window.get("pre_measurement_steps"),
            result.get("valid_steps"),
            window.get("post_cutoff_steps"),
            window.get("total_executed_steps"),
        )
        if (
            any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in partition
            )
            or sum(partition[:3]) != partition[3]
            or result.get("total_executed_steps") != partition[3]
            or (expected_warmup == 0 and partition[0] != 0)
            or (expected_warmup > 0 and partition[0] <= 0)
        ):
            reasons.append("step_partition_invalid")

        workload_started = window.get("workload_started_unix")
        measurement_started = window.get("started_unix")
        cutoff = window.get("cutoff_unix")
        if (
            not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in (workload_started, measurement_started, cutoff)
            )
            or not math.isclose(
                float(measurement_started) - float(workload_started),
                float(expected_warmup),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(cutoff) - float(measurement_started),
                float(expected_measurement),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            reasons.append("measurement_boundaries_invalid")

    valid_steps = result.get("valid_steps")
    rate = result.get("valid_steps_per_min")
    if (
        not isinstance(valid_steps, int)
        or isinstance(valid_steps, bool)
        or valid_steps <= 0
        or not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not math.isfinite(float(rate))
        or not math.isclose(
            float(rate),
            valid_steps * 60.0 / expected_measurement,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        reasons.append("throughput_accounting_invalid")

    request_success = result.get("request_success_rate")
    valid_ratio = result.get("valid_step_ratio")
    measured_requests = result.get("measured_request_count")
    if (
        not isinstance(request_success, (int, float))
        or isinstance(request_success, bool)
        or not math.isfinite(float(request_success))
        or not 0.0 <= float(request_success) <= 1.0
        or float(request_success) < 0.98
    ):
        reasons.append("request_success_below_floor")
    if (
        not isinstance(valid_ratio, (int, float))
        or isinstance(valid_ratio, bool)
        or not math.isfinite(float(valid_ratio))
        or float(valid_ratio) < 0.0
        or not isinstance(measured_requests, int)
        or isinstance(measured_requests, bool)
        or measured_requests <= 0
        or not math.isclose(
            float(valid_ratio),
            valid_steps / measured_requests,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or float(valid_ratio) < 0.90
    ):
        reasons.append("valid_step_ratio_below_floor")

    pressure = result.get("pressure_validation")
    if mode == "baseline":
        if (
            not isinstance(pressure, dict)
            or pressure.get("required") is not True
            or pressure.get("valid") is not True
            or pressure.get("reasons") != []
        ):
            reasons.append("baseline_high_kv_pressure_invalid")
    elif (
        not isinstance(pressure, dict)
        or pressure.get("required") is not False
        or pressure.get("valid") is not True
    ):
        reasons.append("candidate_pressure_diagnostic_invalid")

    if mode == "candidate":
        lifecycle = result.get("program_lifecycle")
        if (
            not isinstance(lifecycle, dict)
            or lifecycle.get("enabled") is not True
            or lifecycle.get("measured_attempted") != 192
            or lifecycle.get("measured_acknowledged") != 192
            or lifecycle.get("measured_failed") != 0
            or lifecycle.get("measured_unsupported") != 0
        ):
            reasons.append("candidate_program_lifecycle_invalid")

    try:
        project_stable_model_provenance(result.get("model_provenance"))
    except ValueError:
        reasons.append("backend_serving_contract_invalid")

    return reasons


def pair_validation_reasons(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    workload_nonce: str,
) -> list[str]:
    """Validate that Direct and Oracle are the same immutable experiment."""

    reasons: list[str] = []
    expected_nonce_sha256 = hashlib.sha256(workload_nonce.encode()).hexdigest()
    nonce_pair = (
        baseline.get("workload_nonce_sha256"),
        candidate.get("workload_nonce_sha256"),
    )
    if nonce_pair != (expected_nonce_sha256, expected_nonce_sha256):
        reasons.append("workload_nonce_pair_mismatch")

    for field in (
        "controller_config_sha256",
        "corpus_fingerprint",
        "corpus_namespace",
        "release_id",
    ):
        left, right = baseline.get(field), candidate.get(field)
        if not isinstance(left, str) or not left or left != right:
            reasons.append(f"{field}_pair_mismatch")
    controller_digest = baseline.get("controller_config_sha256")
    if (
        not isinstance(controller_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", controller_digest) is None
    ):
        reasons.append("controller_config_sha256_invalid")

    if baseline.get("profile_config") != candidate.get("profile_config"):
        reasons.append("profile_config_pair_mismatch")
    if baseline.get("model_adapter") != candidate.get("model_adapter"):
        reasons.append("model_adapter_pair_mismatch")

    baseline_contract = backend_serving_contract_sha256(
        baseline.get("model_provenance")
    )
    candidate_contract = backend_serving_contract_sha256(
        candidate.get("model_provenance")
    )
    if (
        baseline_contract not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
        or candidate_contract not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
    ):
        reasons.append("backend_serving_contract_mismatch")

    def task_ids(result: dict[str, Any]) -> set[str]:
        workflows = result.get("workflows")
        if not isinstance(workflows, list):
            return set()
        return {
            row["task_id"]
            for row in workflows
            if isinstance(row, dict)
            and isinstance(row.get("task_id"), str)
            and row["task_id"]
        }

    baseline_tasks = task_ids(baseline)
    candidate_tasks = task_ids(candidate)
    if (
        len(baseline_tasks) != 192
        or len(candidate_tasks) != 192
        or baseline_tasks != candidate_tasks
    ):
        reasons.append("task_set_pair_mismatch")
    return reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--suite", required=True, choices=("public", "hidden"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workload-nonce", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--request-timeout-sec", type=int, default=3600)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="run and persist only the Direct leg for calibration",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate = args.candidate_dir.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if not candidate.is_dir() or candidate.is_symlink():
        raise SystemExit("--candidate-dir must be a real directory")
    if not (candidate / "launch.sh").is_file():
        raise SystemExit("candidate launch.sh is missing")
    if re.fullmatch(r"[0-9a-f]{64}", args.workload_nonce) is None:
        raise SystemExit("--workload-nonce must be 64 lowercase hexadecimal characters")
    if not 1800 <= args.request_timeout_sec <= 21600:
        raise SystemExit("--request-timeout-sec must be in 1800..21600")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite evidence: {output}")

    expected_tree = tree_sha256(candidate)
    if expected_tree != EXPECTED_ORACLE_TREE_SHA256:
        raise SystemExit(
            "candidate tree is not the pinned v2.1 official ThunderAgent Oracle"
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "gpu67-v2.1-distinct-swe192-direct-oracle-pair",
        "status": "running",
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "candidate_tree_sha256": expected_tree,
        "workload_nonce_sha256": hashlib.sha256(
            args.workload_nonce.encode()
        ).hexdigest(),
        "suite": args.suite,
        "profile": args.profile,
        "legs": {},
        "evaluation": None,
        "errors": [],
    }
    atomic_write(output, report, initial=True)

    lock_path = MEASUREMENT_LOCK_PATH
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    stack_started = False
    stack_runner = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = {
            **os.environ,
            "KV_AUTHOR_REFERENCE_GATEWAY_DIR": str(candidate),
        }

        def stack(action: str, timeout: int = 600) -> str:
            completed = subprocess.run(
                [str(STACK), action],
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"stack {action} failed ({completed.returncode})\n"
                    + completed.stdout[-12000:]
                )
            return completed.stdout

        stack_runner = stack
        # A failed Compose start can still leave a partially-created project.
        # Arm exact project cleanup before requesting startup.
        stack_started = True
        stack("start")
        controller = service_container("bench-controller")
        model_api = service_container("model-api")
        controller_id = run(
            ["docker", "inspect", "--format", "{{.Id}}", controller]
        ).strip()
        model_api_id = run(
            ["docker", "inspect", "--format", "{{.Id}}", model_api]
        ).strip()

        stack("stop-oracle")
        baseline_result = controller_run(
            controller,
            mode="baseline",
            nonce=args.workload_nonce,
            timeout_sec=args.request_timeout_sec,
            suite=args.suite,
            profile=args.profile,
        )
        baseline_result = dict(baseline_result)
        baseline_result["model_provenance"] = project_stable_model_provenance(
            baseline_result.get("model_provenance")
        )
        report["legs"]["baseline"] = baseline_result
        atomic_write(output, report)

        if args.baseline_only:
            baseline_reasons = leg_validation_reasons(
                baseline_result,
                suite=args.suite,
                profile=args.profile,
                mode="baseline",
            )
            if baseline_reasons:
                raise RuntimeError(
                    f"Direct calibration leg is invalid: {baseline_reasons}"
                )
            report["status"] = "complete"
            report["completed_at_utc"] = utc_now()
            report["evaluation"] = None
            atomic_write(output, report)
            print(
                json.dumps(
                    {
                        "backend_serving_contract_sha256": (
                            BACKEND_SERVING_CONTRACT_SHA256
                        ),
                        "baseline_valid_steps_per_min": baseline_result[
                            "valid_steps_per_min"
                        ],
                        "mode": "baseline-only",
                        "valid": True,
                    },
                    sort_keys=True,
                )
            )
            return 0

        stack("restart-oracle")
        if tree_sha256(candidate) != expected_tree:
            raise RuntimeError("candidate tree changed before Candidate leg")
        candidate_result = controller_run(
            controller,
            mode="candidate",
            nonce=args.workload_nonce,
            timeout_sec=args.request_timeout_sec,
            suite=args.suite,
            profile=args.profile,
        )
        candidate_result = dict(candidate_result)
        candidate_result["model_provenance"] = project_stable_model_provenance(
            candidate_result.get("model_provenance")
        )
        report["legs"]["candidate"] = candidate_result
        if tree_sha256(candidate) != expected_tree:
            raise RuntimeError("candidate tree changed during Candidate leg")
        if run(["docker", "inspect", "--format", "{{.Id}}", controller]).strip() != controller_id:
            raise RuntimeError("controller identity changed across the pair")
        if run(["docker", "inspect", "--format", "{{.Id}}", model_api]).strip() != model_api_id:
            raise RuntimeError("model-api identity changed across the pair")

        baseline = report["legs"]["baseline"]
        candidate_result = report["legs"]["candidate"]
        baseline_rate = float(baseline["valid_steps_per_min"])
        candidate_rate = float(candidate_result["valid_steps_per_min"])
        speedup = candidate_rate / baseline_rate
        baseline_reasons = leg_validation_reasons(
            baseline,
            suite=args.suite,
            profile=args.profile,
            mode="baseline",
        )
        candidate_reasons = leg_validation_reasons(
            candidate_result,
            suite=args.suite,
            profile=args.profile,
            mode="candidate",
        )
        pair_reasons = pair_validation_reasons(
            baseline,
            candidate_result,
            workload_nonce=args.workload_nonce,
        )
        baseline_provenance = canonical_sha256(baseline.get("model_provenance"))
        candidate_provenance = canonical_sha256(
            candidate_result.get("model_provenance")
        )
        baseline_contract = backend_serving_contract_sha256(
            baseline.get("model_provenance")
        )
        candidate_contract = backend_serving_contract_sha256(
            candidate_result.get("model_provenance")
        )
        provenance_match = (
            baseline_contract in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
            and candidate_contract in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
        )
        report["evaluation"] = {
            "baseline_valid_steps_per_min": baseline_rate,
            "candidate_valid_steps_per_min": candidate_rate,
            "speedup": speedup,
            "change_percent": (speedup - 1.0) * 100.0,
            "baseline_request_success_rate": baseline.get("request_success_rate"),
            "candidate_request_success_rate": candidate_result.get(
                "request_success_rate"
            ),
            "baseline_valid": not baseline_reasons,
            "candidate_valid": not candidate_reasons,
            "baseline_invalid_reasons": baseline_reasons,
            "candidate_invalid_reasons": candidate_reasons,
            "pair_invalid_reasons": pair_reasons,
            "baseline_model_provenance_sha256": baseline_provenance,
            "candidate_model_provenance_sha256": candidate_provenance,
            "backend_serving_contract_sha256": baseline_contract,
            "model_provenance_match": provenance_match,
            "performance_improved": speedup > 1.0,
            "valid": (
                not baseline_reasons
                and not candidate_reasons
                and not pair_reasons
                and provenance_match
            ),
        }
        report["status"] = "complete"
        report["completed_at_utc"] = utc_now()
        atomic_write(output, report)
        print(json.dumps(report["evaluation"], sort_keys=True))
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["completed_at_utc"] = utc_now()
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        atomic_write(output, report)
        raise
    finally:
        if stack_started and stack_runner is not None:
            try:
                stack_runner("stop", timeout=600)
            except Exception as cleanup_error:
                print(f"warning: benchmark stack cleanup failed: {cleanup_error}")
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
