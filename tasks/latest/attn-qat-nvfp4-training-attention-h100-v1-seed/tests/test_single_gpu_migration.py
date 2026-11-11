"""The single-allocation Work/Judge contract for this task.

Mirrors the Flash-KMeans pilot: the wrapper and preflight fail closed before
any Docker or GPU state is touched, the launcher never falls back across
modes, the runner takes one UUID lock and attests idle before its first
container probe, and the cooperative pre-submit hook is installed root-owned
inside the task cwd.
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

from conftest import REPOSITORY_ROOT, TASK_ROOT

ROOT = TASK_ROOT
LAUNCH = REPOSITORY_ROOT / "launch/attn-qat-train/run.sh"
HOOK = "/app/workdir/attn_qat_lab/.sforge/prepare_submit.sh"
UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


def _sforge_binary() -> str:
    repository_sforge = REPOSITORY_ROOT / ".venv/bin/sforge"
    if repository_sforge.is_file():
        return str(repository_sforge)
    return str(Path(sys.executable).parent / "sforge")


def _clean_env(**overrides: str) -> dict[str, str]:
    env = dict(os.environ)
    for name in (
        "WORK_GPU", "JUDGE_GPU", "SFORGE_SINGLE_GPU_HANDOFF",
        "SFORGE_ACCELERATOR_DEVICE_UUID", "SFORGE_ACCELERATOR_ALLOCATION_KEY",
        "SFORGE_ACCELERATOR_OBSERVER", "SFORGE_TRUSTED_LEASE_ID",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(REPOSITORY_ROOT), env.get("PYTHONPATH", "")))
    )
    env.update(overrides)
    return env


def _run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=REPOSITORY_ROOT, env=env, text=True, capture_output=True, check=False
    )


def test_single_wrapper_fails_before_runtime_for_noncanonical_uuid() -> None:
    result = _run(
        ["bash", str(ROOT / "scripts/run_single_gpu_fixed_budget.sh")],
        _clean_env(SFORGE_ACCELERATOR_DEVICE_UUID="0"),
    )
    assert result.returncode != 0
    assert "never an index, alias, or MIG identity" in result.stderr
    assert "docker" not in result.stdout.casefold()


def test_single_wrapper_rejects_work_judge_gpu_variables() -> None:
    env = _clean_env(SFORGE_ACCELERATOR_DEVICE_UUID=UUID)
    env["WORK_GPU"] = "1"
    result = _run(["bash", str(ROOT / "scripts/run_single_gpu_fixed_budget.sh")], env)
    assert result.returncode != 0
    assert "rejects WORK_GPU/JUDGE_GPU" in result.stderr


def test_single_wrapper_rejects_noncanonical_allocation_key() -> None:
    result = _run(
        ["bash", str(ROOT / "scripts/run_single_gpu_fixed_budget.sh")],
        _clean_env(
            SFORGE_BIN=_sforge_binary(),
            PYTHON_BIN=sys.executable,
            SFORGE_ACCELERATOR_DEVICE_UUID=UUID,
            SFORGE_ACCELERATOR_ALLOCATION_KEY="allocation-sha256:" + "0" * 64,
        ),
    )
    assert result.returncode != 0
    assert "does not match the canonical" in result.stderr


def test_runtime_preflight_rejects_mismatched_allocation_digest() -> None:
    result = _run(
        [sys.executable, "-B", str(ROOT / "scripts/single_gpu_runtime_preflight.py")],
        _clean_env(
            SFORGE_ACCELERATOR_DEVICE_UUID=UUID,
            SFORGE_ACCELERATOR_ALLOCATION_KEY="allocation-sha256:" + "0" * 64,
            SFORGE_TRUSTED_LEASE_ID="lease-attn-qat-meta-test",
        ),
    )
    assert result.returncode == 78, result.stderr
    assert "does not match the canonical" in result.stderr
    assert "refusing" in result.stderr


def test_runtime_preflight_passes_with_a_canonical_allocation() -> None:
    from sforge.harness.backend import AcceleratorAllocation
    from sforge.harness.resource_handoff import canonical_allocation_key

    key = canonical_allocation_key(AcceleratorAllocation("nvidia", (UUID,)))
    result = _run(
        [sys.executable, "-B", str(ROOT / "scripts/single_gpu_runtime_preflight.py")],
        _clean_env(
            SFORGE_ACCELERATOR_DEVICE_UUID=UUID,
            SFORGE_ACCELERATOR_ALLOCATION_KEY=key,
            SFORGE_TRUSTED_LEASE_ID="lease-attn-qat-meta-test",
            JUDGE_PORT="0",  # rejected as < 1024 so the probe never binds a real port
        ),
    )
    # Everything except the deliberately invalid port must be satisfied.
    assert result.returncode == 78
    assert result.stderr.count("\n  - ") == 1, result.stderr
    assert "JUDGE_PORT must be 1024..65535" in result.stderr


def test_launch_rejects_cross_mode_environment_before_runtime() -> None:
    legacy = _run(["bash", str(LAUNCH), "--legacy-dual-gpu"],
                  _clean_env(SFORGE_ACCELERATOR_DEVICE_UUID=UUID))
    assert legacy.returncode != 0
    assert "reject inherited" in legacy.stderr

    single_env = _clean_env()
    single_env["WORK_GPU"] = "1"
    single = _run(["bash", str(LAUNCH), "--shared-gpu"], single_env)
    assert single.returncode != 0
    assert "shared mode rejects WORK_GPU/JUDGE_GPU" in single.stderr

    bad_port = _run(["bash", str(LAUNCH), "--judge-port", "3128"], _clean_env())
    assert bad_port.returncode != 0
    assert "must be 1024..65535 and not 3128" in bad_port.stderr

    mixed = _run(["bash", str(LAUNCH), "--legacy-dual-gpu", "--device-uuid", UUID], _clean_env())
    assert mixed.returncode != 0
    assert "cannot be combined with --legacy-dual-gpu" in mixed.stderr


def test_single_contract_has_one_lock_and_explicit_final_handoff() -> None:
    common = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/run_single_gpu_fixed_budget.sh").read_text(encoding="utf-8")
    lib = (ROOT / "scripts/lib.sh").read_text(encoding="utf-8")
    assert "canonical_allocation_key" in wrapper
    assert "SFORGE_TRUSTED_LEASE_ID" in wrapper
    assert "lease-attn-qat-" in wrapper
    assert "SFORGE_SINGLE_GPU_HANDOFF=1" in wrapper
    assert "SFORGE_ACCELERATOR_OBSERVER=nvml" in wrapper
    for name in (
        "SFORGE_SINGLE_GPU_HANDOFF", "SFORGE_ACCELERATOR_DEVICE_UUID",
        "SFORGE_ACCELERATOR_ALLOCATION_KEY", "SFORGE_ACCELERATOR_OBSERVER",
        "SFORGE_TRUSTED_LEASE_ID", "SFORGE_CALIBRATION_EPOCH",
    ):
        assert name in lib, name
    assert "assert_container_accelerator()" in lib
    assert "assert_portable_gpu_idle()" in lib
    assert "profile_min_memory_bytes" in common
    assert "profile_min_compute_capability" in common
    assert 'exec 8>"$(gpu_lock_path "${work_gpu_uuid}")"' in common
    lock_position = common.index('flock -n 8 || die "single GPU allocation')
    initial_attestation_position = common.index("NvmlAcceleratorObserver().attest_idle(")
    first_container_probe_position = common.index("assert_container_accelerator")
    assert lock_position < initial_attestation_position < first_container_probe_position
    assert 'phase="before_work_create"' in common
    assert "AcceleratorAllocation" in common
    assert "--standalone-final-handoff" in common
    assert '--handoff-source-run-id "${RUN_ID}"' in common
    assert '--judge-url "http://127.0.0.1:${JUDGE_PORT}"' in common
    assert "single_allocation_serial_v1" in common
    assert 'scheduling_mode="single_gpu_lease_shared_final_v2_handoff"' in common
    assert 'scheduling_mode="dual_h100_shared_final_v2"' in common
    # Single mode keeps the coordinator alive across the standalone final.
    single_stop = common.index('if [[ "${SINGLE_GPU_HANDOFF}" != "1" ]]; then\n  stop_judge_server')
    final_eval = common.index("--standalone-final-handoff")
    assert single_stop < final_eval
    assert common.index('if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then\n  stop_judge_server', final_eval) > final_eval
    # Measurement identities.
    for marker in (
        "measurement_profile_sha256", "deployment_attestation_json",
        "unqualified_missing_calibration_epoch", "raw_score_comparison_identity_json",
        "raw_score_comparison_identity_sha256",
        "same_evaluation_contract_and_measurement_profile_only",
        "requires_qualified_pair_identity",
    ):
        assert marker in common, marker
    pair_start = common.index('raw_score_comparison_identity_json="$(')
    pair_end = common.index('raw_score_comparison_identity_sha256="sha256:$(', pair_start)
    pair_source = common[pair_start:pair_end]
    assert '--arg evaluation_contract_sha256 "${evaluation_contract_sha256}"' in pair_source
    assert '--arg measurement_profile_sha256 "${measurement_profile_sha256}"' in pair_source
    assert "qualification_status" not in pair_source
    assert "gpu_name" not in pair_source
    profile_start = common.index('measurement_profile_json="$(')
    profile_end = common.index('measurement_profile_sha256="sha256:$(', profile_start)
    profile_source = common[profile_start:profile_end]
    assert "evaluation_contract_sha256" not in profile_source
    assert "gpu_uuid" not in profile_source
    assert "sm_clock" not in profile_source
    attestation_start = common.index('deployment_attestation_json="$(')
    attestation_source = common[attestation_start:profile_start]
    assert "accelerator_uuid: $gpu_uuid" in attestation_source
    assert "observed_sm_clock_mhz" in attestation_source
    assert common.count("raw_score_comparison: {") == 1
    assert common.count("raw_score_comparison: ($raw_score_comparison_identity") == 1
    assert 'experiment_manifest="${RUN_LOG_ROOT}/experiment-manifest.json"' in common
    assert 'sha256sum "${experiment_manifest}"' in common
    assert "portable_contract: {" in common
    assert "comparability_key_sha256: $measurement_profile_sha256" in common
    assert 'exec 9>"$(gpu_lock_path "${judge_gpu_uuid}")"' in common
    assert 'if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]' in common
    assert f"SFORGE_PRE_SUBMIT_HOOK={HOOK}" in common
    assert "attn-qat-v1-${execution_profile_id}-" in common
    assert "attn-qat-v1-h100-" in common
    assert '--evaluation-mode "${scheduling_mode}"' in common


def test_launch_defaults_single_and_keeps_explicit_legacy_path() -> None:
    launch = LAUNCH.read_text(encoding="utf-8")
    assert 'ENTRY="${SINGLE_ENTRY}"' in launch
    assert "run_single_gpu_fixed_budget.sh" in launch
    for flag in ("--device-uuid", "--allocation-key", "--lease-id",
                 "--calibration-epoch", "--judge-port", "--legacy-dual-gpu"):
        assert flag in launch, flag
    assert 'ENTRY="${LEGACY_ENTRY}"' in launch
    assert "never" in launch
    assert "never falls back to separate GPUs" in launch
    assert "reject inherited single-allocation environment" in launch
    assert "SFORGE_SINGLE_GPU_HANDOFF=0" in launch
    assert "SFORGE_ACCELERATOR_OBSERVER" in launch
    assert "attn-qat-nvfp4-training-attention-h100-v1-seed" in launch


def test_prepare_submit_is_cooperative_and_fails_closed() -> None:
    script = ROOT / "images/work/benchmark/prepare_submit.sh"
    source = script.read_text(encoding="utf-8")
    assert source == (
        REPOSITORY_ROOT
        / "tasks/latest/flash-kmeans-paper-h100-v7-scratch/images/work/public_tests/prepare_submit.sh"
    ).read_text(encoding="utf-8")
    assert "/dev/nvidia" in source
    assert "kill -TERM" in source
    assert "kill -KILL" in source
    assert "exit 75" in source
    assert "not host idle attestation" in source
    with tempfile.TemporaryDirectory() as directory:
        env = dict(os.environ)
        env["SFORGE_GPU_PROC_ROOT"] = directory
        result = subprocess.run(
            ["bash", str(script)], env=env, text=True, capture_output=True, check=False
        )
    assert result.returncode == 0, result.stderr
    assert "drain complete" in result.stdout


def test_agent_must_cleanup_immediately_before_submit() -> None:
    command = "sforge-submit --details"
    agent_task = (ROOT / "images/work/AGENT_TASK.md").read_text(encoding="utf-8")
    readme = (
        ROOT / "images/work/workdir/attn_qat_lab/README_AGENT.md"
    ).read_text(encoding="utf-8")
    assert command in agent_task
    assert command in readme
    task = json.loads(
        (ROOT / "tasks/attn_qat_nvfp4_training_attention_h100_seed_v1.json").read_text(encoding="utf-8")
    )
    query = task["work"]["agent_query"]
    for required in ("wait for all public-test and compiler workers", command):
        assert required in query, required
    # The GPU layout and the drain hook behind sforge-submit are not agent-facing.
    for text in (agent_task, readme, query):
        for token in ("prepare_submit", "single-accelerator", "share the", "paused", "coordinator"):
            assert token not in text, token
    assert task["cwd"] == "/app/workdir/attn_qat_lab"
    assert HOOK.startswith(task["cwd"] + "/")
    dockerfile = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    assert f"COPY benchmark/prepare_submit.sh {HOOK}" in dockerfile
    assert "chown -R root:root /app/workdir/attn_qat_lab/.sforge" in dockerfile
    # The hook is installed after the workspace is handed to the agent user.
    assert dockerfile.index("chown -R agent:agent /app/workdir") < dockerfile.index(
        "chown -R root:root /app/workdir/attn_qat_lab/.sforge"
    )
    assert "GPU" not in agent_task.split("\n")[0]
    assert "H100" not in agent_task.split("\n")[0]


def test_formal_artifact_validator_accepts_both_scheduling_modes() -> None:
    source = (ROOT / "scripts/validate_formal_artifacts.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    modes = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "EVALUATION_MODES" for t in node.targets
        ):
            modes = node.value
    assert modes is not None
    assert 'SINGLE_GPU_EVALUATION_MODE = (\n    "single_gpu_lease_shared_final_v2_handoff"\n)' in source
    assert source.count('"evaluation_mode": evaluation_mode,') == 2
    assert source.count("--evaluation-mode") == 2
    bad = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/validate_formal_artifacts.py"), "pre",
         "--task-root", "/nonexistent", "--run-id", "r", "--task-id", "t", "--model", "m",
         "--budget-seconds", "7200", "--interval-seconds", "1800",
         "--evaluation-mode", "something_else"],
        capture_output=True, text=True, check=False,
    )
    assert bad.returncode != 0
    assert "invalid choice" in bad.stderr


@pytest.mark.parametrize("name", ["run_single_gpu_fixed_budget.sh", "run_fixed_budget.sh", "lib.sh"])
def test_task_scripts_pass_shell_syntax(name: str) -> None:
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / name)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr

def test_workspace_admission_tolerates_the_root_owned_hook_directory(tmp_path: Path) -> None:
    """The Work image now carries <lab>/.sforge/prepare_submit.sh; the agent's public
    runner admits the live lab in workspace mode, so that entry must not be reported
    as an unexpected root entry (differential: identical roots with and without it)."""
    import importlib.util

    admission_path = ROOT / "images/work/benchmark/source_admission.py"
    spec = importlib.util.spec_from_file_location("single_gpu_workspace_admission", admission_path)
    assert spec is not None and spec.loader is not None
    admission = importlib.util.module_from_spec(spec)
    sys.modules["single_gpu_workspace_admission"] = admission
    spec.loader.exec_module(admission)
    task = json.loads(next((ROOT / "tasks").glob("*_v[0-9].json")).read_text(encoding="utf-8"))
    pattern = task["version_enumeration"]["directory_pattern"]
    prefix = pattern.split("(", 1)[0]
    candidate_file = "submission.py" if (ROOT / "images/work/workdir").rglob("submission.py") else "submission.cu"
    seed = next((ROOT / "images/work/workdir").rglob(f"{prefix}001/submission.*"))

    def build(root: Path, with_hook: bool) -> None:
        candidate = root / "candidates" / f"{prefix}001"
        candidate.mkdir(parents=True)
        (candidate / seed.name).write_bytes(seed.read_bytes())
        (root / "FINAL_VERSION").write_text("1\n", encoding="utf-8")
        (root / "notes.md").write_text("notes\n", encoding="utf-8")
        (root / "AGENT_TASK.md").write_text("x\n", encoding="utf-8")
        (root / "README_AGENT.md").write_text("x\n", encoding="utf-8")
        (root / "build").mkdir()
        (root / "tools").mkdir()
        if with_hook:
            hook = root / ".sforge" / "prepare_submit.sh"
            hook.parent.mkdir()
            hook.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    plain = tmp_path / "plain"
    hooked = tmp_path / "hooked"
    build(plain, False)
    build(hooked, True)
    without = admission.validate_submission(plain, 1, workspace_mode=True)
    with_hook = admission.validate_submission(hooked, 1, workspace_mode=True)
    assert with_hook["errors"] == without["errors"]
    assert not any(".sforge" in error for error in with_hook["errors"])
    strict = admission.validate_submission(hooked, 1)
    assert any("unexpected root entry: .sforge" in error for error in strict["errors"])


def test_invalid_final_leaves_a_durable_postcheck_and_completion_record() -> None:
    """A rejected final fails closed but persists its reason.

    The validator prints a complete ``invalid_final`` record (no
    evaluated_archive / final_report / private_judge_artifact binding, so it
    can never read as a pass) and exits non-zero; the launcher captures that
    record into formal-postcheck.json and writes a ``failed`` completion.json
    (``failure_stage: formal_postcheck``) before dying.
    """
    validator = ROOT / "scripts/validate_formal_artifacts.py"
    rejected = subprocess.run(
        [sys.executable, "-B", str(validator), "post", "--task-root", "/nonexistent",
         "--final-task-root", "/nonexistent", "--final-eval-id", "manual-1",
         "--task-id", "t", "--expected-archive-sha256", "a" * 64,
         "--expected-manifest-sha256", "b" * 64, "--expected-case-count", "3"],
        capture_output=True, text=True, check=False,
    )
    assert rejected.returncode == 3
    record = json.loads(rejected.stdout)
    assert record["status"] == "invalid_final" and record["phase"] == "post"
    assert record["reason"] and record["error_type"]
    assert 'formal artifact validation failed: ' + record["reason"] in rejected.stderr
    for forbidden in ("evaluated_archive", "final_report", "private_judge_artifact"):
        assert forbidden not in record
    run = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    assert "record_invalid_final() {" in run
    assert 'status: "failed",\n      failure_stage: "formal_postcheck",' in run
    assert "sed -n 's/^formal artifact validation failed: //p'" in run
    assert '.status == "invalid_final"' in run and '.status != "invalid_final"' in run
    postcheck_call = run.index("record_invalid_final \"${postcheck_rc}\"")
    assert postcheck_call < run.index('mv -f -- "${postcheck_stdout}" "${formal_postcheck}"')
    # The helper that discarded the temporary file is no longer on the postcheck path.
    assert "write_command_json_atomically \\\n  \"${formal_postcheck}\"" not in run
