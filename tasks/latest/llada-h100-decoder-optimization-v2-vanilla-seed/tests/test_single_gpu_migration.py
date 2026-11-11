"""The single-allocation Work/Judge contract for this task.

Mirrors the Flash-KMeans pilot: the wrapper and preflight fail closed before
any Docker or GPU state is touched, the launcher never falls back across
modes, the runner takes one UUID lock and attests idle before its first
container probe, the task Docker adapter forwards the coordinator allocation
and scopes the coordinator's final Judge to its own root-owned checkpoint
directory, the formal validator binds the final report to the durable lease,
and the cooperative pre-submit hook is installed root-owned inside the task
cwd.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from conftest import REPOSITORY_ROOT, ROOT, load_file

LAUNCH = REPOSITORY_ROOT / "launch/llada-h100/run.sh"
HOOK = "/app/workdir/decoder_lab/.sforge/prepare_submit.sh"
UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"
TASK_ID = "llada_h100_decoder_optimization_v2"
CONTRACT = "sha256:" + "c" * 64


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
        "SFORGE_CALIBRATION_EPOCH", "JUDGE_PORT",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPOSITORY_ROOT), env.get("PYTHONPATH", ""))))
    env.update(overrides)
    return env


def _run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=REPOSITORY_ROOT, env=env, text=True, capture_output=True, check=False)


def test_single_wrapper_fails_before_runtime_for_noncanonical_uuid() -> None:
    result = _run(["bash", str(ROOT / "scripts/run_single_gpu_fixed_budget.sh")], _clean_env(SFORGE_ACCELERATOR_DEVICE_UUID="0"))
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
        _clean_env(SFORGE_BIN=_sforge_binary(), SFORGE_ACCELERATOR_DEVICE_UUID=UUID,
                   SFORGE_ACCELERATOR_ALLOCATION_KEY="allocation-sha256:" + "0" * 64),
    )
    assert result.returncode != 0
    assert "does not match the canonical" in result.stderr


def test_runtime_preflight_rejects_mismatched_allocation_digest() -> None:
    result = _run(
        [sys.executable, "-B", str(ROOT / "scripts/single_gpu_runtime_preflight.py")],
        _clean_env(SFORGE_ACCELERATOR_DEVICE_UUID=UUID, SFORGE_ACCELERATOR_ALLOCATION_KEY="allocation-sha256:" + "0" * 64,
                   SFORGE_TRUSTED_LEASE_ID="lease-llada-meta-test"),
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
        _clean_env(SFORGE_ACCELERATOR_DEVICE_UUID=UUID, SFORGE_ACCELERATOR_ALLOCATION_KEY=key,
                   SFORGE_TRUSTED_LEASE_ID="lease-llada-meta-test", JUDGE_PORT="0"),
    )
    assert result.returncode == 78
    assert result.stderr.count("\n  - ") == 1, result.stderr
    assert "JUDGE_PORT must be 1024..65535" in result.stderr


def test_launch_rejects_cross_mode_environment_before_runtime() -> None:
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
    common = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/run_single_gpu_fixed_budget.sh").read_text(encoding="utf-8")
    lib = (ROOT / "scripts/lib.sh").read_text(encoding="utf-8")
    assert "canonical_allocation_key" in wrapper and "SFORGE_TRUSTED_LEASE_ID" in wrapper
    assert "lease-llada-" in wrapper and "SFORGE_SINGLE_GPU_HANDOFF=1" in wrapper and "SFORGE_ACCELERATOR_OBSERVER=nvml" in wrapper
    assert "set -Eeuo pipefail" in wrapper and "main() {" in wrapper
    for name in ("SFORGE_SINGLE_GPU_HANDOFF", "SFORGE_ACCELERATOR_DEVICE_UUID", "SFORGE_ACCELERATOR_ALLOCATION_KEY",
                 "SFORGE_ACCELERATOR_OBSERVER", "SFORGE_TRUSTED_LEASE_ID", "SFORGE_CALIBRATION_EPOCH"):
        assert f"    {name} \\\n" in lib, name
    for helper in ("acquire_single_gpu_lock() {", "assert_single_gpu_lock_held() {", "assert_container_accelerator() {",
                   "assert_portable_gpu_idle() {", "gpu_index_of_uuid() {", "acquire_dual_gpu_locks() {", "assert_gpu_available() {"):
        assert helper in lib, helper
    assert "profile_min_memory_bytes" in common and "profile_min_compute_capability" in common
    lock_position = common.index('acquire_single_gpu_lock "${work_gpu_uuid}"')
    attestation_position = common.index("NvmlAcceleratorObserver().attest_idle(")
    probe_position = common.index("assert_container_accelerator")
    assert lock_position < attestation_position < probe_position
    assert 'phase="before_work_create"' in common and "AcceleratorAllocation" in common
    assert "--standalone-final-handoff" in common
    assert '--handoff-source-run-id "${RUN_ID}"' in common
    assert '--judge-url "http://127.0.0.1:${JUDGE_PORT}"' in common
    assert "judge_serve_extra_hosts+=(--host 127.0.0.1)" in common
    assert '  --host "${docker_bridge_gateway}" "${judge_serve_extra_hosts[@]}" --port "${JUDGE_PORT}"' in common
    assert "--host 0.0.0.0" not in common
    assert "single_allocation_serial_v1" in common
    assert 'scheduling_mode="single_gpu_lease_manual_judge_then_standalone_final_handoff"' in common
    assert 'scheduling_mode="dual_gpu_interactive_then_frozen_final"' in common
    assert common.count('--evaluation-mode "${scheduling_mode}"') == 2
    # The final Judge is the task-contract ceiling for the coordinator; the
    # qualified 3,900 s outer deadline stays inside the Judge (persistent),
    # and the 6,600 s adapter kill sits between the two.
    single_final = common.index("--standalone-final-handoff")
    single_block = common[common.rindex('if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then', 0, single_final):single_final]
    assert '--timeout "${TASK_EVAL_TIMEOUT}"' in single_block and 'wait_for_http "http://127.0.0.1:${JUDGE_PORT}/openapi.json" 30' in single_block
    assert "(( POLICY_FINAL_OUTER_TIMEOUT < LLADA_JUDGE_KILL_TIMEOUT" in common
    assert "&& LLADA_JUDGE_KILL_TIMEOUT < TASK_EVAL_TIMEOUT ))" in common
    # The coordinator (judge_pid) survives the Work barrier in single mode and
    # is stopped only after the standalone final.
    barrier = common.index("# Freeze/final barrier")
    assert common.index('if [[ "${SINGLE_GPU_HANDOFF}" != "1" ]]; then\n  stop_judge_server\nfi', barrier) < single_final
    assert common.index("  stop_judge_server\n  (( final_eval_rc == 0 ))", single_final) > single_final
    assert 'kill -0 "${judge_pid}" 2>/dev/null' in common
    # Legacy path retained verbatim: two locks, distinct pair, inline final eval.
    assert "acquire_dual_gpu_locks" in common and "assert_distinct_gpu_pair" in common
    assert common.count('"${H100_ENTRY}" \\\n  eval') == 2
    assert 'export LLADA_RUNTIME_RUN_ID="${final_eval_id}"' in common
    # Final-run identity and checkpoint directory are prepared before the
    # coordinator starts; the five host identity fields enter its environment.
    assert 'export LLADA_FINAL_RUNTIME_RUN_ID="${final_eval_id}"' in common
    assert 'export LLADA_FINAL_JUDGE_CHECKPOINT_DIR="${LLADA_JUDGE_CHECKPOINT_DIR}"' in common
    assert common.index('export LLADA_FINAL_JUDGE_CHECKPOINT_DIR="${LLADA_JUDGE_CHECKPOINT_DIR}"') < common.index('"${H100_ENTRY}" serve')
    single_env = common[common.index('# Every full scored round binds the host runtime and image identity.'):common.index('if [[ "${PILOT_MODE}" == "1" ]]; then\n  judge_extra_env=')]
    for field in ("WORK_IMAGE_ID=${recorded_work_image_id}", "JUDGE_IMAGE_ID=${recorded_judge_image_id}", "MODEL_VALIDATION_SHA256=${model_validation_sha256}",
                  "RUNTIME_ENTRY_SHA256=${runtime_entry_sha256}", "SOURCE_DIGEST=${recorded_image_source_digest}"):
        assert f"SFORGE_PRIVATE_CONTEXT_{field}" in single_env, field
    assert "SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256=" not in single_env
    assert common.index('runtime_entry_sha256="$(sha256sum "${H100_ENTRY}"') < common.index("judge_extra_env=")
    assert 'export LLADA_EXECUTION_PROFILE_ID="${execution_profile_id}"' in common
    assert "LLADA_TARGET=${execution_profile_id}" in common and "LLADA_TARGET=dual-cuda-sm80-34g" not in common
    assert '[[ "${execution_profile_id}" == "single-cuda-sm80-34g" && "${profile_count}" == "1" ]]' in common
    assert 'execution_profile_id="dual-cuda-sm80-34g"' in common
    assert 'gpu_lock_scheme="single_uuid_flock_work_judge_final"' in common and 'gpu_lock_scheme="two_uuid_flocks_deterministic_order"' in common
    assert 'interactive_evaluation_lifecycle="serial_single_gpu_lease_handoff"' in common
    assert 'final_work_lock: "held_until_cleanup"' in common
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
    # Exact image IDs resolved before use (the recorded_* IDs from the run
    # manifest are only frozen later and are re-checked against these).
    assert "profile_work_image_id" in profile_source and "profile_judge_image_id" in profile_source
    attestation_start = common.index('deployment_attestation_json="$(')
    attestation_source = common[attestation_start:profile_start]
    assert "accelerator_uuid: $gpu_uuid" in attestation_source and "observed_sm_clock_mhz" in attestation_source
    assert common.count("raw_score_comparison: ($raw_score_comparison_identity") == 1
    assert "portable_contract: {" in common
    assert "comparability_key_sha256: $measurement_profile_sha256" in common
    assert f"SFORGE_PRE_SUBMIT_HOOK={HOOK}" in common


class _FakeNetwork:
    attrs = {"Internal": True}

    def reload(self) -> None:
        return None


class _FakeContainer:
    id = "fake-container-id"
    name = "fake-container"


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        client = self

        class _Containers:
            def create(self, image: str, **kwargs):
                client.calls.append((image, kwargs))
                return _FakeContainer()

        class _Images:
            mapping = {
                f"llada_decoder.judge.{TASK_ID}:local": "sha256:" + "b" * 64,
                "sha256:" + "b" * 64: "sha256:" + "b" * 64,
                f"llada_decoder.work.{TASK_ID}:local": "sha256:" + "a" * 64,
                "sha256:" + "a" * 64: "sha256:" + "a" * 64,
            }

            def get(self, image: str):
                return type("Image", (), {"id": self.mapping[image]})()

        class _Networks:
            def get(self, name: str) -> _FakeNetwork:
                return _FakeNetwork()

        self.containers = _Containers()
        self.images = _Images()
        self.networks = _Networks()


class _FakeBackend:
    def __init__(self) -> None:
        self.client = _FakeClient()


@pytest.fixture
def adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = load_file("llada_single_gpu_adapter_test", ROOT / "scripts/sforge_h100_entry.py")
    repo = tmp_path / "models--GSAI-ML--LLaDA-8B-Instruct"
    (repo / "snapshots" / module.MODEL_REVISION).mkdir(parents=True)
    monkeypatch.setenv(module.MODEL_REPO_ENV, str(repo))
    monkeypatch.setenv(module.NETWORK_ENV, "formal-internal")
    monkeypatch.setenv(module.WORK_UUID_ENV, UUID)
    monkeypatch.setenv(module.JUDGE_UUID_ENV, UUID)
    monkeypatch.setenv(module.RUN_SCOPE_ENV, "unit-run")
    monkeypatch.setenv(module.FINAL_RUN_SCOPE_ENV, "unit-run-final-120m")
    monkeypatch.setenv(module.WORK_IMAGE_ID_ENV, "sha256:" + "a" * 64)
    monkeypatch.setenv(module.JUDGE_IMAGE_ID_ENV, "sha256:" + "b" * 64)
    monkeypatch.setenv(module.EXECUTION_PROFILE_ENV, module.SINGLE_EXECUTION_PROFILE_ID)
    monkeypatch.setattr(module, "_judge_checkpoint_volume", lambda: "/logs/runs/unit-run/judge-checkpoints:/logs/verifier/checkpoints:rw")
    monkeypatch.setattr(module, "_final_judge_checkpoint_volume", lambda: "/logs/runs/unit-run-final-120m/judge-checkpoints:/logs/verifier/checkpoints:rw")
    return module


def _judge_environment(module) -> dict[str, str]:
    return {
        "NVIDIA_VISIBLE_DEVICES": UUID, "CUDA_VISIBLE_DEVICES": "0", "LLADA_TARGET": module.SINGLE_EXECUTION_PROFILE_ID,
        "LLADA_GPU_ROLE": "judge", "LLADA_INTERACTIVE_JUDGE_TIMEOUT": "1200", "LLADA_FINAL_JUDGE_TIMEOUT": "3600",
    }


def _annotations(attempt: str) -> dict[str, str]:
    from sforge.harness.backend import AcceleratorAllocation
    from sforge.harness.resource_handoff import (
        JUDGE_ATTEMPT_ANNOTATION, JUDGE_ROLE_ANNOTATION, canonical_allocation_key, handoff_annotations,
    )

    return {
        **handoff_annotations(run_id="unit-run", trusted_lease_id="lease-llada-unit", allocation_key=canonical_allocation_key(AcceleratorAllocation("nvidia", (UUID,))),
                              evaluation_contract_sha256=CONTRACT),
        JUDGE_ROLE_ANNOTATION: "judge",
        JUDGE_ATTEMPT_ANNOTATION: attempt,
    }


def test_adapter_scopes_the_coordinator_final_judge_to_the_final_run(adapter) -> None:
    from sforge.harness.backend import AcceleratorAllocation
    from sforge.harness.resource_handoff import JUDGE_ATTEMPT_ANNOTATION

    backend = _FakeBackend()
    digest = "0123456789abcdef" * 2
    adapter.role_bound_create_container(
        backend, f"llada_decoder.judge.{TASK_ID}:local", f"llada_decoder.judge.{TASK_ID}.final-{digest[:12]}",
        environment=_judge_environment(adapter), annotations=_annotations("final-" + digest[:24]),
        accelerator_allocation=AcceleratorAllocation("nvidia", (UUID,)), user="1000:1000",
    )
    adapter.role_bound_create_container(
        backend, f"llada_decoder.judge.{TASK_ID}:local", f"llada_decoder.judge.{TASK_ID}.{digest[:12]}",
        environment=_judge_environment(adapter), annotations=_annotations(digest[:12]),
        accelerator_allocation=AcceleratorAllocation("nvidia", (UUID,)), user="1000:1000",
    )
    final_kwargs, interactive_kwargs = (call[1] for call in backend.client.calls)
    assert final_kwargs["volumes"][-1].startswith("/logs/runs/unit-run-final-120m/judge-checkpoints:")
    assert interactive_kwargs["volumes"][-1].startswith("/logs/runs/unit-run/judge-checkpoints:")
    assert final_kwargs["labels"]["mlsysforge.run_id"] == "unit-run-final-120m"
    assert interactive_kwargs["labels"]["mlsysforge.run_id"] == "unit-run"
    for kwargs in (final_kwargs, interactive_kwargs):
        assert kwargs["labels"]["mlsysforge.task_id"] == TASK_ID and kwargs["labels"]["mlsysforge.gpu_role"] == "judge"
        assert kwargs["labels"]["mlsysforge.gpu_uuid"] == UUID and kwargs["labels"][JUDGE_ATTEMPT_ANNOTATION]
        assert kwargs["environment"]["LLADA_TARGET"] == adapter.SINGLE_EXECUTION_PROFILE_ID
        assert kwargs["device_requests"][0]["DeviceIDs"] == [UUID] and kwargs["network_mode"] == "none"
        assert kwargs["labels"]["io.sforge.handoff.role"] == "judge"


@pytest.mark.parametrize(
    ("name_suffix", "attempt"),
    [("final-0123456789ab", "0123456789ab"), ("0123456789ab", "final-0123456789ab0123456789ab"), ("final-0123456789ab", "final-ffffffffffff0123456789ab")],
)
def test_adapter_refuses_a_final_identity_that_name_and_annotation_do_not_share(adapter, name_suffix, attempt) -> None:
    from sforge.harness.backend import AcceleratorAllocation

    backend = _FakeBackend()
    with pytest.raises(RuntimeError, match="final handoff Judge identity mismatch"):
        adapter.role_bound_create_container(
            backend, f"llada_decoder.judge.{TASK_ID}:local", f"llada_decoder.judge.{TASK_ID}.{name_suffix}",
            environment=_judge_environment(adapter), annotations=_annotations(attempt),
            accelerator_allocation=AcceleratorAllocation("nvidia", (UUID,)),
        )
    assert backend.client.calls == []


def test_adapter_requires_the_allocation_to_name_the_pinned_device(adapter) -> None:
    from sforge.harness.backend import AcceleratorAllocation

    backend = _FakeBackend()
    with pytest.raises(RuntimeError, match="does not name the single pinned device"):
        adapter.role_bound_create_container(
            backend, f"llada_decoder.judge.{TASK_ID}:local", f"llada_decoder.judge.{TASK_ID}.0123456789ab",
            environment=_judge_environment(adapter), annotations=_annotations("0123456789ab"),
            accelerator_allocation=AcceleratorAllocation("nvidia", ("GPU-ffffffff-ffff-ffff-ffff-ffffffffffff",)),
        )
    with pytest.raises(RuntimeError, match="only NVIDIA"):
        adapter.role_bound_create_container(
            backend, f"llada_decoder.judge.{TASK_ID}:local", f"llada_decoder.judge.{TASK_ID}.0123456789ab",
            environment=_judge_environment(adapter), annotations=_annotations("0123456789ab"),
            accelerator_allocation=AcceleratorAllocation("amd", (UUID,)),
        )
    assert backend.client.calls == []


def test_adapter_selects_the_execution_profile_from_the_launcher(adapter, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeBackend()
    environment = {**_judge_environment(adapter), "LLADA_TARGET": adapter.EXECUTION_PROFILE_ID}
    with pytest.raises(RuntimeError, match="LLADA_TARGET must be single-cuda-sm80-34g"):
        adapter.role_bound_create_container(backend, f"llada_decoder.judge.{TASK_ID}:local", f"llada_decoder.judge.{TASK_ID}.0123456789ab", environment=environment)
    monkeypatch.delenv(adapter.EXECUTION_PROFILE_ENV)
    with pytest.raises(RuntimeError, match="LLADA_TARGET must be dual-cuda-sm80-34g"):
        adapter.role_bound_create_container(backend, f"llada_decoder.judge.{TASK_ID}:local", f"llada_decoder.judge.{TASK_ID}.0123456789ab", environment=_judge_environment(adapter))
    monkeypatch.setenv(adapter.EXECUTION_PROFILE_ENV, "quad-cuda-sm80-34g")
    with pytest.raises(RuntimeError, match="must be one of"):
        adapter.role_bound_create_container(backend, f"llada_decoder.judge.{TASK_ID}:local", f"llada_decoder.judge.{TASK_ID}.0123456789ab", environment=_judge_environment(adapter))
    assert backend.client.calls == []


def test_final_checkpoint_volume_is_scoped_to_the_final_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_file("llada_single_gpu_checkpoint_test", ROOT / "scripts/sforge_h100_entry.py")
    logs = tmp_path / "logs"
    (logs / "runs/unit-run/judge-checkpoints").mkdir(parents=True)
    monkeypatch.setenv("SFORGE_LOG_DIR", str(logs))
    monkeypatch.setenv(module.RUN_SCOPE_ENV, "unit-run")
    monkeypatch.setenv(module.FINAL_RUN_SCOPE_ENV, "unit-run-final-120m")
    monkeypatch.delenv(module.FINAL_JUDGE_CHECKPOINT_DIR_ENV, raising=False)
    with pytest.raises(RuntimeError, match=module.FINAL_JUDGE_CHECKPOINT_DIR_ENV):
        module._final_judge_checkpoint_volume()
    # The interactive directory is not the final one even though it is root-shaped.
    monkeypatch.setenv(module.FINAL_JUDGE_CHECKPOINT_DIR_ENV, str(logs / "runs/unit-run/judge-checkpoints"))
    with pytest.raises(RuntimeError, match="outside its exact run scope"):
        module._final_judge_checkpoint_volume()
    monkeypatch.delenv(module.FINAL_RUN_SCOPE_ENV)
    with pytest.raises(RuntimeError, match=module.FINAL_RUN_SCOPE_ENV):
        module._final_judge_checkpoint_volume()


def test_formal_artifact_validator_accepts_both_scheduling_modes(tmp_path: Path) -> None:
    source = (ROOT / "scripts/validate_formal_artifacts.py").read_text(encoding="utf-8")
    assert 'SINGLE_GPU_EVALUATION_MODE = (\n    "single_gpu_lease_manual_judge_then_standalone_final_handoff"\n)' in source
    assert source.count("--evaluation-mode") == 2
    assert source.count('"evaluation_mode": mode,') == 2
    bad = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/validate_formal_artifacts.py"), "pre", "--task-root", "/nonexistent", "--run-id", "r",
         "--task-id", "t", "--model", "m", "--budget-seconds", "7200", "--interval-seconds", "1800", "--evaluation-mode", "something_else"],
        capture_output=True, text=True, check=False,
    )
    assert bad.returncode != 0 and "invalid choice" in bad.stderr
    validator = load_file("llada_single_gpu_formal_validator", ROOT / "scripts/validate_formal_artifacts.py")
    assert set(validator.EXPECTED_LIFECYCLES) == set(validator.EVALUATION_MODES) == set(validator.EXPECTED_LOCK_SCHEMES)
    assert validator.EXPECTED_LIFECYCLES[validator.EVALUATION_MODE]["interactive_evaluation"] == "concurrent_distinct_gpus"
    assert validator.EXPECTED_LIFECYCLES[validator.SINGLE_GPU_EVALUATION_MODE]["interactive_evaluation"] == "serial_single_gpu_lease_handoff"
    for lifecycle in validator.EXPECTED_LIFECYCLES.values():
        assert lifecycle["final_work_lock"] == "held_until_cleanup" and lifecycle["checkpoint_ownership"] == "root:root-0700"

    # The final report of the single-GPU mode is bound to the durable lease.
    task_root = tmp_path / "runs/unit-run/task"
    task_root.mkdir(parents=True)
    report = tmp_path / "report.json"
    report.write_text('{"submission_id": "final-0123456789ab"}', encoding="utf-8")
    archive_hex = "e" * 64
    lease = {
        "state": "CLOSED", "recovery_policy": "never_resume",
        "identity": {"task_id": "task", "run_id": "unit-run", "trusted_lease_id": "lease-llada-unit"},
        "final_judge": {
            "request": {"final_run_id": "unit-run-final-120m", "source_run_id": "unit-run", "task_id": "task", "archive_sha256": "sha256:" + archive_hex},
            "artifacts": {"report": {"sha256": "sha256:" + hashlib.sha256(report.read_bytes()).hexdigest()}, "submission": {"sha256": "sha256:" + archive_hex}},
        },
    }
    (task_root / "resource_lease.json").write_text(json.dumps(lease), encoding="utf-8")
    kwargs = dict(final_eval_id="unit-run-final-120m", task_id="task", expected_archive_sha256=archive_hex, report_path=report, submission_id="final-0123456789ab")
    validator.validate_final_handoff_lease(task_root, **kwargs)
    with pytest.raises(ValueError, match="submission identity is invalid"):
        validator.validate_final_handoff_lease(task_root, **{**kwargs, "submission_id": "unit-run-final-120m"})
    with pytest.raises(ValueError, match="does not bind this final report"):
        validator.validate_final_handoff_lease(task_root, **{**kwargs, "final_eval_id": "other-final"})
    with pytest.raises(ValueError, match="does not bind this final report"):
        validator.validate_final_handoff_lease(task_root, **{**kwargs, "expected_archive_sha256": "f" * 64})
    broken = {**lease, "state": "FINAL_JUDGE_RUNNING"}
    (task_root / "resource_lease.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="did not close"):
        validator.validate_final_handoff_lease(task_root, **kwargs)

    experiment = {
        "execution_profile_id": "single-cuda-sm80-34g",
        "judge_gpu": {"uuid": UUID, "name": "accelerator-model"},
        "images": {"work": {"id": "sha256:" + "a" * 64}, "judge": {"id": "sha256:" + "b" * 64}},
        "resource_handoff": {"mode": "single_allocation_serial_v1", "allocation_key": "allocation-sha256:" + "1" * 64, "trusted_lease_id": "lease-llada-unit",
                             "evaluation_contract_sha256": CONTRACT, "execution_profile_id": "single-cuda-sm80-34g", "final_judge": {"path": "coordinator_standalone_final_handoff"}},
        "deployment_attestation": {"accelerator_uuid": UUID, "observed_sm_clock_mhz": "1980"},
        "measurement_profile": {"accelerator": {"model": "accelerator-model"}, "images": {"work": "sha256:" + "a" * 64, "judge": "sha256:" + "b" * 64}, "comparability_key_sha256": "sha256:" + "2" * 64},
        "raw_score_comparison": {"evaluation_contract_sha256": CONTRACT, "measurement_profile_sha256": "sha256:" + "2" * 64, "policy": "same_evaluation_contract_and_measurement_profile_only"},
    }
    validator.validate_single_gpu_handoff_provenance(experiment)
    leaking = json.loads(json.dumps(experiment))
    leaking["measurement_profile"]["accelerator_uuid"] = UUID
    with pytest.raises(ValueError, match="measurement identities are invalid"):
        validator.validate_single_gpu_handoff_provenance(leaking)


def test_layout_validator_enforces_the_single_allocation_contract() -> None:
    validator = load_file("llada_single_gpu_layout_validator", ROOT / "scripts/validate_layout.py")
    errors: list[str] = []
    validator.validate_single_gpu_handoff(ROOT, errors)
    assert errors == []
    for name in ("scripts/run_single_gpu_fixed_budget.sh", "scripts/single_gpu_runtime_preflight.py", "scripts/refresh_portable_manifests.py"):
        assert name in validator.SCRIPTS
    assert "images/work/public_tests/prepare_submit.sh" in validator.PORTABLE_FILES
    assert "docs/SINGLE_GPU_MIGRATION.md" in validator.DOCS
    assert validator.HOOK == HOOK


def test_prepare_submit_is_cooperative_and_fails_closed() -> None:
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
    command = "sforge-submit --details"
    agent_task = (ROOT / "images/work/AGENT_TASK.md").read_text(encoding="utf-8")
    readme = (ROOT / "images/work/workdir/decoder_lab/README_AGENT.md").read_text(encoding="utf-8")
    assert command in agent_task and command in readme
    assert "automatically invokes" in agent_task and ".sforge/prepare_submit.sh" in agent_task and ".sforge/prepare_submit.sh" in readme
    assert "performs no automatic aggregate evaluation" in agent_task
    assert "## Submitting on the single-accelerator profile" in agent_task
    task = json.loads((ROOT / "tasks/llada_h100_decoder_optimization_v2.json").read_text(encoding="utf-8"))
    query = task["work"]["agent_query"]
    for required in ("prepare_submit.sh", "performs no automatic aggregate evaluation", "wait for all public-test and compiler workers", command):
        assert required in query, required
    for forbidden in ("hidden", "private", "timeout boundary", "timeout-boundary"):
        assert forbidden not in query.casefold()
    assert task["cwd"] == "/app/workdir/decoder_lab" and HOOK.startswith(task["cwd"] + "/")
    dockerfile = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    assert f"COPY public_tests/prepare_submit.sh {HOOK}" in dockerfile
    assert "chown -R root:root /app/workdir/decoder_lab/.sforge" in dockerfile
    assert dockerfile.index("chown -R agent:agent") < dockerfile.index("chown -R root:root /app/workdir/decoder_lab/.sforge")
    # Judge source admission enumerates only the captured paths, so the
    # root-owned hook directory needs no allowlist.
    admission = load_file("llada_single_gpu_admission", ROOT / "images/judge/verifier/source_admission.py")
    assert "unexpected root entry" not in (ROOT / "images/judge/verifier/source_admission.py").read_text(encoding="utf-8")
    assert set(admission.ALLOWED_SOLUTION_FILES) == {"__init__.py", "candidate.py"}


def test_harness_provides_the_final_archive_context_and_loopback_bind() -> None:
    """The runner relies on two harness features beyond the pilot primitives."""
    from sforge import cli
    from sforge.harness import judge_server

    assert judge_server.FINAL_ARCHIVE_PRIVATE_CONTEXT_KEY == "SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256"
    context = judge_server._final_judge_private_context({"SFORGE_PRIVATE_CONTEXT_ROLE": "judge"}, archive_sha256="sha256:" + "e" * 64)
    assert context["SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256"] == "e" * 64
    assert cli._serve_hosts(["172.17.0.1", "127.0.0.1"]) == ["172.17.0.1", "127.0.0.1"]
    verifier = (ROOT / "images/judge/verifier/verify_decoder.py").read_text(encoding="utf-8")
    assert '"archive_sha256": "ARCHIVE_SHA256"' in verifier


@pytest.mark.parametrize("name", ["run_single_gpu_fixed_budget.sh", "run_fixed_budget.sh", "lib.sh", "eval_archive.sh"])
def test_task_scripts_pass_shell_syntax(name: str) -> None:
    result = subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_single_mode_measurement_profile_resolves_image_ids_before_use() -> None:
    """Regression: the single-GPU block built its measurement profile from
    ``recorded_*_image_id`` variables that are only assigned when the run-local
    image manifest is frozen further down, so every single-mode launch died
    with "recorded_work_image_id: unbound variable" after preflight."""
    source = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    profile = source.index('measurement_profile_json="$(')
    assert 'profile_work_image_id="$(docker image inspect' in source[:profile]
    assert 'profile_judge_image_id="$(docker image inspect' in source[:profile]
    profile_block = source[profile:source.index('measurement_profile_sha256="sha256:$(', profile)]
    assert '--arg work_image_id "${profile_work_image_id}"' in profile_block
    assert '--arg judge_image_id "${profile_judge_image_id}"' in profile_block
    assert "recorded_work_image_id" not in profile_block
    # Every later use of the recorded IDs follows their assignment from the
    # frozen run-local manifest.
    first_recorded_use = source.index("${recorded_work_image_id}")
    assert source.index('recorded_work_image_id="$(jq') < first_recorded_use


def test_formal_artifact_validator_accepts_the_never_resume_pause_boundary() -> None:
    """The single-GPU harness pauses (never resume) instead of stopping Work.

    Seen live: run llada-single-verify-20260911 died at the pre-final check
    with "did not prove all Work processes stopped" although the snapshot
    proved every process frozen and the container was strictly retired
    afterwards. The dual-GPU mode still requires the stopped proof.
    """
    validator = load_file("llada_single_gpu_freeze_validator", ROOT / "scripts/validate_formal_artifacts.py")
    single = validator.SINGLE_GPU_EVALUATION_MODE
    dual = validator.EVALUATION_MODE
    paused = {
        "mechanism": "docker_pause_never_resume_final",
        "all_work_processes_frozen": True,
        "all_work_processes_stopped": False,
        "container_status": "paused",
    }
    validator.validate_freeze_boundary(paused, single)
    with pytest.raises(ValueError, match="processes stopped"):
        validator.validate_freeze_boundary(paused, dual)
    with pytest.raises(ValueError, match="processes frozen"):
        validator.validate_freeze_boundary({**paused, "all_work_processes_frozen": False}, single)
    with pytest.raises(ValueError, match="never-resume pause"):
        validator.validate_freeze_boundary({**paused, "mechanism": "docker_stop"}, single)
    with pytest.raises(ValueError, match="container paused"):
        validator.validate_freeze_boundary({**paused, "container_status": "exited"}, single)
    stopped = {"mechanism": "docker_stop", "all_work_processes_frozen": True, "all_work_processes_stopped": True, "container_status": "exited"}
    validator.validate_freeze_boundary(stopped, dual)
    with pytest.raises(ValueError, match="processes stopped"):
        validator.validate_freeze_boundary(None, dual)


def test_runner_refuses_a_run_id_the_version_store_already_records() -> None:
    """Reusing a RUN_ID against the same version store only fails at the final
    version persist (after the agent budget); the runner must refuse up front."""
    source = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    guard = source.index('die "version store already records run id ${RUN_ID}')
    assert 'SFORGE_VERSION_STORE_ROOT}/tasks/$(basename "${MIGRATION_ROOT}")/runs/${RUN_ID}' in source[:guard]
    assert guard < source.index("write_command_json_atomically"), "guard must run before any run artifact is produced"


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
         "--task-id", "t", "--expected-archive-sha256", "a" * 64, "--verify-suite", "full"],
        capture_output=True, text=True, check=False,
    )
    assert rejected.returncode == 3
    record = json.loads(rejected.stdout)
    assert record["status"] == "invalid_final" and record["phase"] == "post"
    assert record["reason"] and record["error_type"]
    assert 'error: ' + record["reason"] in rejected.stderr
    for forbidden in ("evaluated_archive", "final_report", "private_judge_artifact"):
        assert forbidden not in record
    run = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    assert "record_invalid_final() {" in run
    assert 'status: "failed",\n      failure_stage: "formal_postcheck",' in run
    assert "sed -n 's/^error: //p'" in run
    assert '.status == "invalid_final"' in run and '.status != "invalid_final"' in run
    postcheck_call = run.index("record_invalid_final \"${postcheck_rc}\"")
    assert postcheck_call < run.index('mv -f -- "${postcheck_stdout}" "${formal_postcheck}"')
    # The helper that discarded the temporary file is no longer on the postcheck path.
    assert "write_command_json_atomically \\\n  \"${formal_postcheck}\"" not in run
