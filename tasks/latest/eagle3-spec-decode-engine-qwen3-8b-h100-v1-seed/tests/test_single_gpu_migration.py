"""The single-allocation Work/Judge contract for this task.

Mirrors the Flash-KMeans pilot: the wrapper and preflight fail closed before
any Docker or GPU state is touched, the launcher never falls back across
modes, the runner takes one UUID lock and attests idle before its first
container probe, the task Docker adapter forwards the coordinator allocation,
and the cooperative pre-submit hook is installed root-owned inside the task
cwd.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from conftest import REPOSITORY_ROOT, ROOT

LAUNCH = REPOSITORY_ROOT / "launch/eagle3-spec-decode/run.sh"
HOOK = "/app/workdir/spec_lab/.sforge/prepare_submit.sh"
UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


def _sforge_binary() -> str:
    """The SForge CLI of the checkout venv, else the one next to this interpreter."""
    repository_sforge = REPOSITORY_ROOT / ".venv/bin/sforge"
    if repository_sforge.is_file():
        return str(repository_sforge)
    return str(Path(sys.executable).parent / "sforge")


def _clean_env(**overrides: str) -> dict[str, str]:
    """A copy of the environment without any single-allocation or GPU variables."""
    env = dict(os.environ)
    for name in (
        "WORK_GPU", "JUDGE_GPU", "SFORGE_SINGLE_GPU_HANDOFF",
        "SFORGE_ACCELERATOR_DEVICE_UUID", "SFORGE_ACCELERATOR_ALLOCATION_KEY",
        "SFORGE_ACCELERATOR_OBSERVER", "SFORGE_TRUSTED_LEASE_ID",
        "SFORGE_CALIBRATION_EPOCH", "JUDGE_PORT",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPOSITORY_ROOT), env.get("PYTHONPATH", ""))))
    env.update(overrides)
    return env


def _run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run one command from the repository root."""
    return subprocess.run(command, cwd=REPOSITORY_ROOT, env=env, text=True, capture_output=True, check=False)


def test_single_wrapper_fails_before_runtime_for_noncanonical_uuid() -> None:
    """An index or alias is rejected before any Docker or GPU operation."""
    result = _run(["bash", str(ROOT / "scripts/run_single_gpu_fixed_budget.sh")], _clean_env(SFORGE_ACCELERATOR_DEVICE_UUID="0"))
    assert result.returncode != 0
    assert "never an index, alias, or MIG identity" in result.stderr
    assert "docker" not in result.stdout.casefold()


def test_single_wrapper_rejects_work_judge_gpu_variables() -> None:
    """Legacy index variables cannot leak into single mode."""
    env = _clean_env(SFORGE_ACCELERATOR_DEVICE_UUID=UUID)
    env["WORK_GPU"] = "1"
    result = _run(["bash", str(ROOT / "scripts/run_single_gpu_fixed_budget.sh")], env)
    assert result.returncode != 0
    assert "rejects WORK_GPU/JUDGE_GPU" in result.stderr


def test_single_wrapper_rejects_noncanonical_allocation_key() -> None:
    """A scheduler allocation key must equal the canonical device-set digest."""
    result = _run(
        ["bash", str(ROOT / "scripts/run_single_gpu_fixed_budget.sh")],
        _clean_env(SFORGE_BIN=_sforge_binary(), PYTHON_BIN=sys.executable, SFORGE_ACCELERATOR_DEVICE_UUID=UUID,
                   SFORGE_ACCELERATOR_ALLOCATION_KEY="allocation-sha256:" + "0" * 64),
    )
    assert result.returncode != 0
    assert "does not match the canonical" in result.stderr


def test_runtime_preflight_rejects_mismatched_allocation_digest() -> None:
    """The production probe refuses a mismatched allocation digest with exit 78."""
    result = _run(
        [sys.executable, "-B", str(ROOT / "scripts/single_gpu_runtime_preflight.py")],
        _clean_env(SFORGE_ACCELERATOR_DEVICE_UUID=UUID, SFORGE_ACCELERATOR_ALLOCATION_KEY="allocation-sha256:" + "0" * 64,
                   SFORGE_TRUSTED_LEASE_ID="lease-eagle3-meta-test"),
    )
    assert result.returncode == 78, result.stderr
    assert "does not match the canonical" in result.stderr
    assert "refusing" in result.stderr


def test_runtime_preflight_passes_with_a_canonical_allocation() -> None:
    """Everything except the deliberately invalid port is satisfied."""
    from sforge.harness.backend import AcceleratorAllocation
    from sforge.harness.resource_handoff import canonical_allocation_key

    key = canonical_allocation_key(AcceleratorAllocation("nvidia", (UUID,)))
    result = _run(
        [sys.executable, "-B", str(ROOT / "scripts/single_gpu_runtime_preflight.py")],
        _clean_env(SFORGE_ACCELERATOR_DEVICE_UUID=UUID, SFORGE_ACCELERATOR_ALLOCATION_KEY=key,
                   SFORGE_TRUSTED_LEASE_ID="lease-eagle3-meta-test", JUDGE_PORT="0"),
    )
    assert result.returncode == 78
    assert result.stderr.count("\n  - ") == 1, result.stderr
    assert "JUDGE_PORT must be 1024..65535" in result.stderr


def test_launch_rejects_cross_mode_environment_before_runtime() -> None:
    """The launcher never mixes or falls back between modes."""
    legacy = _run(["bash", str(LAUNCH), "--legacy-dual-gpu"], _clean_env(SFORGE_ACCELERATOR_DEVICE_UUID=UUID))
    assert legacy.returncode != 0
    assert "reject inherited" in legacy.stderr
    single_env = _clean_env()
    single_env["WORK_GPU"] = "1"
    single = _run(["bash", str(LAUNCH)], single_env)
    assert single.returncode != 0
    assert "single mode rejects WORK_GPU/JUDGE_GPU" in single.stderr
    bad_port = _run(["bash", str(LAUNCH), "--judge-port", "3128"], _clean_env())
    assert bad_port.returncode != 0
    assert "must be 1024..65535 and not 3128" in bad_port.stderr
    mixed = _run(["bash", str(LAUNCH), "--legacy-dual-gpu", "--device-uuid", UUID], _clean_env())
    assert mixed.returncode != 0
    assert "cannot be combined with --legacy-dual-gpu" in mixed.stderr


def test_single_contract_has_one_lock_and_explicit_final_handoff() -> None:
    """The runner's single branch locks, attests, probes, hands off and keeps the coordinator alive."""
    common = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/run_single_gpu_fixed_budget.sh").read_text(encoding="utf-8")
    lib = (ROOT / "scripts/lib.sh").read_text(encoding="utf-8")
    assert "canonical_allocation_key" in wrapper and "SFORGE_TRUSTED_LEASE_ID" in wrapper
    assert "lease-eagle3-" in wrapper and "SFORGE_SINGLE_GPU_HANDOFF=1" in wrapper and "SFORGE_ACCELERATOR_OBSERVER=nvml" in wrapper
    assert "set -Eeuo pipefail" in wrapper and "main() {" in wrapper
    for name in ("SFORGE_SINGLE_GPU_HANDOFF", "SFORGE_ACCELERATOR_DEVICE_UUID", "SFORGE_ACCELERATOR_ALLOCATION_KEY",
                 "SFORGE_ACCELERATOR_OBSERVER", "SFORGE_TRUSTED_LEASE_ID", "SFORGE_CALIBRATION_EPOCH"):
        assert name in lib, name
    for helper in ("acquire_single_gpu_lock() {", "assert_single_gpu_lock_held() {", "assert_container_accelerator() {",
                   "assert_portable_gpu_idle() {", "gpu_index_of_uuid() {", "acquire_dual_gpu_locks() {"):
        assert helper in lib, helper
    assert "profile_min_memory_bytes" in common and "profile_min_compute_capability" in common
    lock_position = common.index('acquire_single_gpu_lock "${work_uuid}"')
    attestation_position = common.index("NvmlAcceleratorObserver().attest_idle(")
    probe_position = common.index("assert_container_accelerator")
    assert lock_position < attestation_position < probe_position
    assert 'phase="before_work_create"' in common and "AcceleratorAllocation" in common
    assert "--standalone-final-handoff" in common
    assert '--handoff-source-run-id "${RUN_ID}"' in common
    assert '--judge-url "http://127.0.0.1:${JUDGE_PORT}"' in common
    # The coordinator binds the bridge gateway for the Work sidecar and, in single
    # mode, loopback for the standalone-final client; never a public interface.
    assert "judge_serve_extra_hosts+=(--host 127.0.0.1)" in common
    assert 'serve --host "${bridge_gateway}" "${judge_serve_extra_hosts[@]}" --port "${JUDGE_PORT}"' in common
    assert "--host 0.0.0.0" not in common
    assert "single_allocation_serial_v1" in common
    assert 'scheduling_mode="single_gpu_lease_manual_judge_then_standalone_final_handoff"' in common
    assert 'scheduling_mode="dual_h100_manual_judge_then_independent_holdout_final"' in common
    # Single mode keeps the coordinator, network and proxy alive across the final.
    final_eval = common.index("--standalone-final-handoff")
    assert common.index('if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then\n  # The Work container has been frozen') < final_eval
    assert common.index("  cleanup\n  trap - EXIT INT TERM HUP\n  (( final_eval_rc == 0 ))") > final_eval
    assert common.count('"${SCRIPT_DIR}/eval_archive.sh" "${final_archive}" "${final_eval_id}"') == 1
    assert common.count('--evaluation-mode "${scheduling_mode}"') == 2
    # The role-bound Judge environment literal and its guard are unchanged.
    assert 'assert_interactive_judge_env_spec "${SFORGE_JUDGE_EXTRA_ENV}" "${judge_uuid}"' in common
    assert 'export EAGLE3_JUDGE_GPU_UUID="${judge_uuid}"' in common
    # Measurement identities.
    for marker in ("measurement_profile_sha256", "deployment_attestation_json", "unqualified_missing_calibration_epoch",
                   "raw_score_comparison_identity_json", "raw_score_comparison_identity_sha256",
                   "same_evaluation_contract_and_measurement_profile_only", "requires_qualified_pair_identity"):
        assert marker in common, marker
    pair_start = common.index('raw_score_comparison_identity_json="$(')
    pair_end = common.index('raw_score_comparison_identity_sha256="sha256:$(', pair_start)
    pair_source = common[pair_start:pair_end]
    assert '--arg evaluation_contract_sha256 "${evaluation_contract_sha256}"' in pair_source
    assert '--arg measurement_profile_sha256 "${measurement_profile_sha256}"' in pair_source
    assert "qualification_status" not in pair_source and "gpu_name" not in pair_source
    profile_start = common.index('measurement_profile_json="$(')
    profile_end = common.index('measurement_profile_sha256="sha256:$(', profile_start)
    profile_source = common[profile_start:profile_end]
    assert "evaluation_contract_sha256" not in profile_source
    assert "gpu_uuid" not in profile_source and "sm_clock" not in profile_source
    assert "recorded_work_image_id" in profile_source and "recorded_judge_image_id" in profile_source
    attestation_start = common.index('deployment_attestation_json="$(')
    attestation_source = common[attestation_start:profile_start]
    assert "accelerator_uuid: $gpu_uuid" in attestation_source and "observed_sm_clock_mhz" in attestation_source
    assert common.count("raw_score_comparison: ($raw_score_comparison_identity") == 1
    assert "portable_contract: {" in common
    assert "comparability_key_sha256: $measurement_profile_sha256" in common
    assert f"SFORGE_PRE_SUBMIT_HOOK={HOOK}" in common


def test_adapter_forwards_the_coordinator_allocation_and_binding_labels() -> None:
    """The task Docker adapter accepts and validates the handoff inputs."""
    source = (ROOT / "scripts/sforge_h100_entry.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    parameters: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "role_bound_create_container":
            parameters = {argument.arg for argument in (*node.args.args, *node.args.kwonlyargs)}
    assert {"annotations", "mounts", "accelerator_allocation"} <= parameters
    assert "list(accelerator_allocation.device_ids) != [requested_uuid]" in source
    assert "_validated_handoff_binding_labels" in source and "HANDOFF_BINDING_ANNOTATIONS" in source
    assert "DockerBackend.create_container = role_bound_create_container" in source
    # The labels the legacy cleanup relies on are still attached.
    for label in ('"mlsysforge.task_id": TASK_ID', '"mlsysforge.run_id": _run_scope()', '"mlsysforge.gpu_role": role', '"mlsysforge.gpu_uuid": requested_uuid'):
        assert label in source, label


def test_launch_defaults_single_and_keeps_explicit_legacy_path() -> None:
    """The unified launcher defaults to the single-allocation entry."""
    launch = LAUNCH.read_text(encoding="utf-8")
    assert 'ENTRY="${SINGLE_ENTRY}"' in launch and "run_single_gpu_fixed_budget.sh" in launch
    for flag in ("--device-uuid", "--allocation-key", "--lease-id", "--calibration-epoch", "--judge-port", "--legacy-dual-gpu"):
        assert flag in launch, flag
    assert 'ENTRY="${LEGACY_ENTRY}"' in launch
    assert "never" in launch and "falls back to the legacy path" in launch
    assert "reject inherited single-allocation environment" in launch
    assert "SFORGE_SINGLE_GPU_HANDOFF=0" in launch and "SFORGE_ACCELERATOR_OBSERVER" in launch
    assert "eagle3-spec-decode-engine-qwen3-8b-h100-v1-seed" in launch


def test_prepare_submit_is_cooperative_and_fails_closed() -> None:
    """The hook is the pilot's byte-identical cooperative drain."""
    script = ROOT / "images/work/public_tests/prepare_submit.sh"
    source = script.read_text(encoding="utf-8")
    assert source == (REPOSITORY_ROOT / "tasks/latest/flash-kmeans-paper-h100-v7-scratch/images/work/public_tests/prepare_submit.sh").read_text(encoding="utf-8")
    for marker in ("/dev/nvidia", "kill -TERM", "kill -KILL", "exit 75", "not host idle attestation"):
        assert marker in source
    with tempfile.TemporaryDirectory() as directory:
        env = dict(os.environ)
        env["SFORGE_GPU_PROC_ROOT"] = directory
        result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "drain complete" in result.stdout


def test_agent_must_cleanup_immediately_before_submit() -> None:
    """Prompt, README, agent_query and Dockerfile carry the handoff contract."""
    command = "sforge-submit --details"
    agent_task = (ROOT / "images/work/AGENT_TASK.md").read_text(encoding="utf-8")
    readme = (ROOT / "images/work/workdir/spec_lab/README_AGENT.md").read_text(encoding="utf-8")
    assert command in agent_task and command in readme
    assert "automatically invokes" in agent_task and ".sforge/prepare_submit.sh" in agent_task and ".sforge/prepare_submit.sh" in readme
    assert "disables in-run automatic hidden evaluation" in agent_task
    assert "## Submitting on the single-accelerator profile" in agent_task
    task = json.loads((ROOT / "tasks/eagle3_spec_decode_engine_qwen3_8b_h100_v1.json").read_text(encoding="utf-8"))
    query = task["work"]["agent_query"]
    for required in ("prepare_submit.sh", "disables in-run automatic hidden evaluation", "wait for all public-test and compiler workers", command):
        assert required in query, required
    assert task["cwd"] == "/app/workdir/spec_lab" and HOOK.startswith(task["cwd"] + "/")
    dockerfile = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    assert f"COPY public_tests/prepare_submit.sh {HOOK}" in dockerfile
    assert "chown -R root:root /app/workdir/spec_lab/.sforge" in dockerfile
    assert dockerfile.index("chown -R agent:agent /app/workdir") < dockerfile.index("chown -R root:root /app/workdir/spec_lab/.sforge")
    assert "H100" not in agent_task.split("\n")[0]
    # Workspace admission enumerates only solutions/, so the hook directory is admitted.
    admission = (ROOT / "images/judge/verifier/source_admission.py").read_text(encoding="utf-8")
    assert "unexpected root entry" not in admission


def test_formal_artifact_validator_accepts_both_scheduling_modes() -> None:
    """Every validator phase records and validates the scheduling mode."""
    source = (ROOT / "scripts/validate_formal_artifacts.py").read_text(encoding="utf-8")
    assert 'SINGLE_GPU_EVALUATION_MODE = (\n    "single_gpu_lease_manual_judge_then_standalone_final_handoff"\n)' in source
    assert source.count("--evaluation-mode") == 2
    assert source.count('"evaluation_mode": getattr(args, "evaluation_mode", EVALUATION_MODE),') == 2
    bad = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/validate_formal_artifacts.py"), "pre", "--task-root", "/nonexistent", "--run-id", "r",
         "--task-id", "t", "--model", "m", "--budget-seconds", "7200", "--interval-seconds", "1800", "--evaluation-mode", "something_else"],
        capture_output=True, text=True, check=False,
    )
    assert bad.returncode != 0 and "invalid choice" in bad.stderr


def test_layout_validator_enforces_the_single_allocation_contract() -> None:
    """validate_layout.py refuses a runner without the standalone final handoff."""
    from conftest import load_file

    validator = load_file("eagle3_single_gpu_layout_validator", ROOT / "scripts/validate_layout.py")
    errors: list[str] = []
    validator.validate_single_gpu_handoff(ROOT, errors)
    assert errors == []
    for name in ("scripts/run_single_gpu_fixed_budget.sh", "scripts/single_gpu_runtime_preflight.py", "scripts/refresh_portable_manifests.py"):
        assert name in validator.SCRIPTS
    assert "images/work/public_tests/prepare_submit.sh" in validator.PORTABLE_FILES
    assert "docs/SINGLE_GPU_MIGRATION.md" in validator.DOCS


@pytest.mark.parametrize("name", ["run_single_gpu_fixed_budget.sh", "run_fixed_budget.sh", "lib.sh", "eval_archive.sh"])
def test_task_scripts_pass_shell_syntax(name: str) -> None:
    """Shell entries parse."""
    result = subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
