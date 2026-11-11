#!/usr/bin/env python3
"""Fail closed unless the installed SForge has the complete serial handoff.

This task-local probe follows the production config contract. It deliberately
does not prescribe public ``resource_lease`` parameters: the runtime creates
the trusted lease internally and binds it through the host-admin channel.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import socket
import sys
from pathlib import Path


EXIT_UNAVAILABLE = 78
TASK_ROOT = Path(__file__).resolve().parents[1]
TASK_SPEC = TASK_ROOT / "tasks" / "standalone_k_grouped_fp8_wgrad_h100_scratch_v2.json"
LAUNCHER = "launch/standalone-kgrouped-wgrad/run.sh"
PUBLIC_CASE_COUNT = 123
INTERACTIVE_CASE_COUNT = 1082
FINAL_CASE_COUNT = 1082
FINAL_SCORED = 1082
FINAL_HARD_GATES = 0
FINAL_EDGE_CASES = 166
MANIFESTS = {
    "resource_manifest_sha256": TASK_ROOT / "portable/resource-manifest.json",
    "build": TASK_ROOT / "portable/build-resource-manifest.json",
    "public": TASK_ROOT / "portable/public-suite-manifest.json",
    "interactive": TASK_ROOT / "portable/interactive-suite-manifest.json",
    "final": TASK_ROOT / "portable/final-suite-manifest.json",
    "oracle": TASK_ROOT / "portable/oracle-protocol.json",
    "selection": TASK_ROOT / "portable/selection-protocol.json",
    "software_abi": TASK_ROOT / "portable/software-abi.json",
    "calibration_protocol": TASK_ROOT / "portable/calibration-protocol.json",
}


def _model_fields(model: object) -> set[str]:
    fields = getattr(model, "model_fields", None)
    if fields is None:
        fields = getattr(model, "__fields__", {})
    return set(fields)


def _verify_portable_files() -> list[str]:
    from sforge.harness.portable_spec import (
        parse_portable_spec,
        verify_sha256_file,
    )

    value = json.loads(TASK_SPEC.read_text(encoding="utf-8"))
    portable = parse_portable_spec(value.get("portable"), task_id=value["task_id"])
    suite = portable.evaluation_suites[portable.defaults.evaluation_suite_id]
    profile = portable.execution_profiles[portable.defaults.execution_profile_id]
    declarations = {
        "resource_manifest_sha256": portable.task_core.resource_manifest_sha256,
        "build": value["delivery"]["build_resource_manifest_sha256"],
        "public": suite.roles["public"].manifest_sha256,
        "interactive": suite.roles["interactive"].manifest_sha256,
        "final": suite.roles["final"].manifest_sha256,
        "oracle": suite.oracle.spec_sha256,
        "selection": suite.selection.spec_sha256,
        "software_abi": profile.software_abi.spec_sha256,
        "calibration_protocol": profile.calibration_protocol.spec_sha256,
    }
    missing: list[str] = []
    for name, path in MANIFESTS.items():
        try:
            verify_sha256_file(path, declarations[name])
        except Exception as exc:
            missing.append(f"portable {name} verification failed: {exc}")
    for role, expected in (
        ("public", PUBLIC_CASE_COUNT),
        ("interactive", INTERACTIVE_CASE_COUNT),
        ("final", FINAL_CASE_COUNT),
    ):
        manifest = json.loads(MANIFESTS[role].read_text(encoding="utf-8"))
        actual = len(manifest.get("cases", []))
        declared = suite.roles[role].case_count
        if actual != declared or actual != expected:
            missing.append(
                f"portable {role} case count is file={actual}, "
                f"declared={declared}, expected={expected}"
            )
    final_manifest = json.loads(MANIFESTS["final"].read_text(encoding="utf-8"))
    final_cases = final_manifest.get("cases", [])
    scored = [case for case in final_cases if case.get("role") == "performance"]
    gates = [case for case in final_cases if case.get("role") == "correctness"]
    if (
        len(scored) != FINAL_SCORED
        or len(gates) != FINAL_HARD_GATES
        or any(case.get("weight", {}).get("numerator") != 0 for case in gates)
    ):
        missing.append(
            f"portable final suite must contain {FINAL_SCORED} scored performance "
            f"cases and {FINAL_HARD_GATES} zero-weight correctness hard gates"
        )
    edge_suite = final_manifest.get("edge_suite") or {}
    if edge_suite.get("case_count") != FINAL_EDGE_CASES:
        missing.append(
            f"portable final suite must declare {FINAL_EDGE_CASES} private edge cases"
        )
    return missing


def main() -> int:
    missing = _verify_portable_files()
    try:
        from sforge import cli
        from sforge.harness import (
            config as config_module,
            evolve_scripts,
            judge_server,
            run_agent,
            run_evaluation,
        )
        from sforge.harness.backend import ContainerBackend
        from sforge.harness.backend.docker_backend import DockerBackend
        from sforge.harness.config import (
            SForgeConfig,
            validate_single_gpu_handoff_config,
        )
    except Exception as exc:
        print(
            "ERROR: cannot import production SForge orchestration: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_UNAVAILABLE

    device_uuid = os.environ.get("SFORGE_ACCELERATOR_DEVICE_UUID")
    allocation_key = os.environ.get("SFORGE_ACCELERATOR_ALLOCATION_KEY")
    trusted_lease_id = os.environ.get("SFORGE_TRUSTED_LEASE_ID")
    judge_port_raw = os.environ.get("JUDGE_PORT", "18096")
    if not judge_port_raw.isdigit():
        missing.append("JUDGE_PORT must be an integer")
    else:
        judge_port = int(judge_port_raw, 10)
        if judge_port < 1024 or judge_port > 65535 or judge_port == 3128:
            missing.append(
                "JUDGE_PORT must be 1024..65535 and must not be 3128"
            )
        else:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.bind(("0.0.0.0", judge_port))
            except OSError as exc:
                missing.append(
                    f"scheduler-selected JUDGE_PORT {judge_port} is not "
                    f"available: {exc}"
                )
            finally:
                probe.close()
    if not isinstance(device_uuid, str) or re.fullmatch(
        r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
        device_uuid,
    ) is None:
        missing.append(
            "K-grouped-wgrad single-GPU mode requires a qualified canonical "
            "whole-GPU UUID; MIG is not qualified"
        )
    try:
        config = SForgeConfig(
            backend="docker",
            single_gpu_handoff=True,
            accelerator_device_uuid=device_uuid,
            accelerator_allocation_key=allocation_key,
            accelerator_observer="nvml",
            trusted_lease_id=trusted_lease_id,
        )
        validate_single_gpu_handoff_config(config)
    except Exception as exc:
        missing.append(f"production single-GPU config rejected allocation: {exc}")

    config_fields = set(inspect.signature(SForgeConfig).parameters)
    for field in (
        "single_gpu_handoff",
        "accelerator_device_uuid",
        "accelerator_allocation_key",
        "accelerator_observer",
        "trusted_lease_id",
    ):
        if field not in config_fields:
            missing.append(f"SForgeConfig is missing {field}")

    # This task launches through the stock Docker backend (no task-local
    # container adapter), so the backend itself must forward labels and the
    # exact accelerator DeviceRequest.
    for backend_class in (ContainerBackend, DockerBackend):
        create_parameters = set(
            inspect.signature(backend_class.create_container).parameters
        )
        for parameter in ("annotations", "accelerator_allocation"):
            if parameter not in create_parameters:
                missing.append(
                    f"{backend_class.__name__}.create_container is missing {parameter}"
                )

    run_source = inspect.getsource(run_agent)
    for marker in (
        "config.single_gpu_handoff",
        "handoff_annotations",
        'create_kwargs["annotations"]',
        'create_kwargs["accelerator_allocation"]',
        "/api/v1/handoff/attach-work",
        "/api/v1/handoff/seal",
        "/api/v1/handoff/final-work-archive",
    ):
        if marker not in run_source:
            missing.append(f"run_agent lacks production handoff step {marker}")

    config_source = inspect.getsource(config_module)
    if (
        "SFORGE_AGENT_EXTRA_ENV" not in config_source
        or "env.update(config.agent_extra_env)" not in run_source
    ):
        missing.append(
            "production runtime does not propagate task-scoped agent "
            "environment"
        )
    submit_source = evolve_scripts.generate_submit_script()
    hook_position = submit_source.find("SFORGE_PRE_SUBMIT_HOOK")
    archive_position = submit_source.find("tar czf")
    submit_position = submit_source.find("/api/v1/submit")
    if (
        hook_position < 0
        or archive_position < 0
        or submit_position < 0
        or not hook_position < archive_position < submit_position
    ):
        missing.append(
            "generated sforge-submit lacks a fail-closed pre-submit hook "
            "before archive and HTTP admission"
        )

    registration = _model_fields(judge_server.RegisterRequest)
    required_registration = {
        "single_gpu_handoff",
        "trusted_lease_id",
        "allocation_key",
        "accelerator_device_ids",
        "evaluation_contract_sha256",
    }
    if not required_registration <= registration:
        missing.append(
            "Judge registration lacks host-admin binding fields: "
            + ", ".join(sorted(required_registration - registration))
        )

    evaluation_parameters = set(
        inspect.signature(run_evaluation.judge_submission).parameters
    )
    for parameter in ("accelerator_allocation", "container_lifecycle"):
        if parameter not in evaluation_parameters:
            missing.append(f"judge_submission is missing {parameter}")

    required_final_request = {
        "admin_secret",
        "source_run_id",
        "final_run_id",
        "task_id",
        "trusted_lease_id",
        "allocation_key",
        "evaluation_contract_sha256",
        "archive_sha256",
        "archive_size",
        "timeout",
    }
    final_request_model = getattr(judge_server, "FinalJudgeRequest", None)
    final_request_fields = (
        _model_fields(final_request_model)
        if final_request_model is not None
        else set()
    )
    if not required_final_request <= final_request_fields:
        missing.append(
            "FinalJudgeRequest lacks coordinator binding fields: "
            + ", ".join(
                sorted(required_final_request - final_request_fields)
            )
        )

    judge_source = inspect.getsource(judge_server)
    for marker in (
        '"/api/v1/handoff/final-judge"',
        "state.run_final_judge(",
        "FINAL_GPU_IDLE_ATTESTED",
        "LeaseState.CLOSED",
    ):
        if marker not in judge_source:
            missing.append(
                "live coordinator final Judge lacks semantic step " + marker
            )

    cli_source = inspect.getsource(cli)
    for marker in (
        "--standalone-final-handoff",
        "--handoff-source-run-id",
        "--judge-url",
        "_request_coordinator_final_judge(",
        "/api/v1/handoff/final-judge",
    ):
        if marker not in cli_source:
            missing.append(
                "sforge eval lacks coordinator final-handoff client marker "
                + marker
            )
    runner_source = (TASK_ROOT / "scripts/run_fixed_budget.sh").read_text(
        encoding="utf-8"
    )
    if "--standalone-final-handoff" not in runner_source:
        missing.append(
            "task fixed-budget body does not explicitly select standalone-final"
        )
    for marker in (
        "raw_score_comparison_identity_json",
        "raw_score_comparison_identity_sha256",
        "same_evaluation_contract_and_measurement_profile_only",
        "requires_qualified_pair_identity",
        "deployment_attestation_json",
        "measurement_profile_sha256",
    ):
        if marker not in runner_source:
            missing.append(
                "task runner lacks measurement-profile / comparison identity marker "
                + marker
            )
    raw_task = json.loads(TASK_SPEC.read_text(encoding="utf-8"))
    task_cwd = raw_task.get("cwd")
    expected_hook = (
        task_cwd.rstrip("/") + "/.sforge/prepare_submit.sh"
        if isinstance(task_cwd, str) and task_cwd.startswith("/")
        else ""
    )
    if (
        not expected_hook
        or f"SFORGE_PRE_SUBMIT_HOOK={expected_hook}" not in runner_source
    ):
        missing.append(
            "task pre-submit hook is not the protected copy inside task cwd"
        )
    initial_lock_position = runner_source.find(
        'flock -n 8 || die "single GPU allocation'
    )
    initial_attestation_position = runner_source.find(
        "NvmlAcceleratorObserver().attest_idle("
    )
    first_container_probe_position = runner_source.find(
        "assert_container_accelerator"
    )
    if not (
        0 <= initial_lock_position
        < initial_attestation_position
        < first_container_probe_position
    ) or 'phase="before_work_create"' not in runner_source:
        missing.append(
            "task runner lacks production NVML initial idle attestation "
            "between the UUID flock and first Docker probe"
        )
    launch_source = (TASK_ROOT.parents[2] / LAUNCHER).read_text(
        encoding="utf-8"
    )
    if "--judge-port" not in launch_source:
        missing.append("unified task launcher lacks scheduler-owned Judge port")
    if "ss -ltn" not in runner_source:
        missing.append("task runner lacks a second Judge-port collision check")

    if missing:
        print(
            "ERROR: production single-GPU Work/Judge handoff is incomplete; "
            "refusing to run the compatibility body or touch Docker/GPU state.",
            file=sys.stderr,
        )
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return EXIT_UNAVAILABLE

    print("Single-GPU production API preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
