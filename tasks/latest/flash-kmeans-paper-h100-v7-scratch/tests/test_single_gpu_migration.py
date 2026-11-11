from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = next(
    parent for parent in ROOT.parents if (parent / "sforge").is_dir()
)


def _sforge_binary() -> str:
    """The SForge CLI the task launcher must find (repository venv first)."""

    repository_sforge = REPOSITORY_ROOT / ".venv/bin/sforge"
    if repository_sforge.is_file():
        return str(repository_sforge)
    return str(Path(sys.executable).parent / "sforge")


class SingleGpuMigrationTests(unittest.TestCase):
    def test_single_wrapper_fails_before_runtime_for_noncanonical_uuid(self) -> None:
        wrapper = ROOT / "scripts/run_single_gpu_fixed_budget.sh"
        env = dict(os.environ)
        env.pop("WORK_GPU", None)
        env.pop("JUDGE_GPU", None)
        env["SFORGE_ACCELERATOR_DEVICE_UUID"] = "0"
        result = subprocess.run(
            ["bash", str(wrapper)],
            cwd=REPOSITORY_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("never an index, alias, or MIG identity", result.stderr)
        self.assertNotIn("docker", result.stdout.casefold())

    def test_runtime_preflight_rejects_mismatched_allocation_digest(self) -> None:
        preflight = ROOT / "scripts/single_gpu_runtime_preflight.py"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (str(REPOSITORY_ROOT), env.get("PYTHONPATH", "")),
            )
        )
        env.update(
            {
                "SFORGE_ACCELERATOR_DEVICE_UUID": (
                    "GPU-01234567-89ab-cdef-0123-456789abcdef"
                ),
                "SFORGE_ACCELERATOR_ALLOCATION_KEY": (
                    "allocation-sha256:" + "0" * 64
                ),
                "SFORGE_TRUSTED_LEASE_ID": "lease-kmeans-meta-test",
            }
        )
        repository_python = REPOSITORY_ROOT / ".venv/bin/python"
        python = (
            str(repository_python)
            if repository_python.is_file()
            else sys.executable
        )
        result = subprocess.run(
            [python, str(preflight)],
            cwd=REPOSITORY_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertIn("does not match the canonical", result.stderr)
        self.assertIn("refusing", result.stderr)

    def test_single_wrapper_rejects_noncanonical_allocation_key(self) -> None:
        wrapper = ROOT / "scripts/run_single_gpu_fixed_budget.sh"
        env = dict(os.environ)
        for name in ("WORK_GPU", "JUDGE_GPU"):
            env.pop(name, None)
        env.update(
            {
                "SFORGE_BIN": _sforge_binary(),
                "SFORGE_ACCELERATOR_DEVICE_UUID": (
                    "GPU-01234567-89ab-cdef-0123-456789abcdef"
                ),
                "SFORGE_ACCELERATOR_ALLOCATION_KEY": (
                    "allocation-sha256:" + "0" * 64
                ),
            }
        )
        result = subprocess.run(
            ["bash", str(wrapper)],
            cwd=REPOSITORY_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match the canonical", result.stderr)

    def test_launch_rejects_cross_mode_environment_before_runtime(self) -> None:
        launch = REPOSITORY_ROOT / "launch/flash-kmeans/run.sh"
        legacy_env = dict(os.environ)
        legacy_env["SFORGE_ACCELERATOR_DEVICE_UUID"] = (
            "GPU-01234567-89ab-cdef-0123-456789abcdef"
        )
        legacy = subprocess.run(
            ["bash", str(launch), "--legacy-dual-gpu"],
            cwd=REPOSITORY_ROOT,
            env=legacy_env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(legacy.returncode, 0)
        self.assertIn("reject inherited", legacy.stderr)

        single_env = dict(os.environ)
        single_env["WORK_GPU"] = "1"
        single = subprocess.run(
            ["bash", str(launch)],
            cwd=REPOSITORY_ROOT,
            env=single_env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(single.returncode, 0)
        self.assertIn("single mode rejects WORK_GPU/JUDGE_GPU", single.stderr)

        bad_port_env = dict(os.environ)
        for name in ("WORK_GPU", "JUDGE_GPU"):
            bad_port_env.pop(name, None)
        bad_port = subprocess.run(
            ["bash", str(launch), "--judge-port", "3128"],
            cwd=REPOSITORY_ROOT,
            env=bad_port_env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(bad_port.returncode, 0)
        self.assertIn("must be 1024..65535 and not 3128", bad_port.stderr)

    def test_isolated_adapter_forwards_labels_and_exact_allocation(self) -> None:
        path = ROOT / "scripts/sforge_isolated_entry.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "isolated_create_container"
        )
        parameters = {
            argument.arg
            for argument in (*function.args.args, *function.args.kwonlyargs)
        }
        self.assertIn("annotations", parameters)
        self.assertIn("accelerator_allocation", parameters)
        self.assertNotIn("del annotations", source)
        self.assertIn('kwargs["labels"] = labels', source)
        self.assertIn('kwargs["device_requests"]', source)
        self.assertIn("_validated_handoff_binding_labels", source)

    def test_single_contract_has_one_lock_and_explicit_final_handoff(self) -> None:
        common = (ROOT / "scripts/run_fixed_budget.sh").read_text(
            encoding="utf-8"
        )
        wrapper = (ROOT / "scripts/run_single_gpu_fixed_budget.sh").read_text(
            encoding="utf-8"
        )
        lib = (ROOT / "scripts/lib.sh").read_text(encoding="utf-8")
        self.assertIn("canonical_allocation_key", wrapper)
        self.assertIn("SFORGE_TRUSTED_LEASE_ID", wrapper)
        self.assertIn("lease-kmeans-", wrapper)
        self.assertIn("SFORGE_TRUSTED_LEASE_ID", lib)
        self.assertIn("profile_min_memory_bytes", common)
        self.assertIn("profile_min_compute_capability", common)
        self.assertIn('exec 8>"$(gpu_lock_path "${work_gpu_uuid}")"', common)
        lock_position = common.index(
            'flock -n 8 || die "single GPU allocation'
        )
        initial_attestation_position = common.index(
            "NvmlAcceleratorObserver().attest_idle("
        )
        first_container_probe_position = common.index(
            "assert_container_accelerator"
        )
        self.assertLess(lock_position, initial_attestation_position)
        self.assertLess(
            initial_attestation_position,
            first_container_probe_position,
        )
        self.assertIn('phase="before_work_create"', common)
        self.assertIn("AcceleratorAllocation", common)
        self.assertIn("--standalone-final-handoff", common)
        self.assertIn("--handoff-source-run-id", common)
        self.assertIn('--judge-url "http://127.0.0.1:${JUDGE_PORT}"', common)
        self.assertIn("single_allocation_serial_v1", common)
        self.assertIn("measurement_profile_sha256", common)
        self.assertIn("deployment_attestation_json", common)
        self.assertIn("unqualified_missing_calibration_epoch", common)
        self.assertIn("raw_latency_comparison_identity_json", common)
        self.assertIn("raw_latency_comparison_identity_sha256", common)
        self.assertIn(
            "same_evaluation_contract_and_measurement_profile_only", common
        )
        self.assertIn("requires_qualified_pair_identity", common)
        pair_start = common.index('raw_latency_comparison_identity_json="$(')
        pair_end = common.index(
            'raw_latency_comparison_identity_sha256="sha256:$(', pair_start
        )
        pair_source = common[pair_start:pair_end]
        self.assertIn(
            '--arg evaluation_contract_sha256 "${evaluation_contract_sha256}"',
            pair_source,
        )
        self.assertIn(
            '--arg measurement_profile_sha256 "${measurement_profile_sha256}"',
            pair_source,
        )
        self.assertNotIn("qualification_status", pair_source)
        self.assertNotIn("gpu_name", pair_source)
        profile_start = common.index('measurement_profile_json="$(')
        profile_end = common.index(
            'measurement_profile_sha256="sha256:$(', profile_start
        )
        self.assertNotIn(
            "evaluation_contract_sha256",
            common[profile_start:profile_end],
        )
        self.assertEqual(common.count("raw_latency_comparison: {"), 1)
        self.assertEqual(
            common.count("raw_latency_comparison: ($raw_latency_comparison_identity"),
            1,
        )
        self.assertIn('experiment_manifest="${RUN_LOG_ROOT}/experiment-manifest.json"', common)
        self.assertIn('sha256sum "${experiment_manifest}"', common)
        self.assertIn("portable_contract: {", common)
        self.assertIn("evaluation_contract_sha256: $evaluation_contract_sha256", common)
        self.assertIn("execution_profile_id: $execution_profile_id", common)
        self.assertIn("comparability_key_sha256: $measurement_profile_sha256", common)
        self.assertIn(
            "measurement_profile_sha256: $measurement_profile_sha256", common
        )
        self.assertIn(
            "identity_sha256: $raw_latency_comparison_identity_sha256", common
        )
        self.assertIn("qualification_status: $qualification_status", common)
        self.assertIn("path: $experiment_manifest", common)
        self.assertIn("sha256: $experiment_manifest_sha256", common)
        self.assertLess(
            common.index('sha256sum "${experiment_manifest}"'),
            common.index('completion="${RUN_LOG_ROOT}/completion.json"'),
        )
        self.assertNotIn("periodic states are", common)
        self.assertIn('exec 9>"$(gpu_lock_path "${judge_gpu_uuid}")"', common)
        self.assertIn('if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]', common)
        self.assertIn(
            "SFORGE_PRE_SUBMIT_HOOK=/app/workdir/flash_kmeans_lab/.sforge/prepare_submit.sh",
            common,
        )
        self.assertIn(
            "CUDA_VISIBLE_DEVICES=0,FLASH_KMEANS_TARGET=H100", common
        )
        self.assertIn(
            "flash-kmeans-v7-scratch-${execution_profile_id}-", common
        )
        self.assertIn("flash-kmeans-v7-scratch-h100-", common)

    def test_launch_defaults_single_and_keeps_explicit_legacy_path(self) -> None:
        launch = (
            REPOSITORY_ROOT / "launch/flash-kmeans/run.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('ENTRY="${SINGLE_ENTRY}"', launch)
        self.assertIn("--device-uuid", launch)
        self.assertIn("--allocation-key", launch)
        self.assertIn("--lease-id", launch)
        self.assertIn("--calibration-epoch", launch)
        self.assertIn("--judge-port", launch)
        self.assertIn("--legacy-dual-gpu", launch)
        self.assertIn('ENTRY="${LEGACY_ENTRY}"', launch)
        self.assertIn("never", launch)
        self.assertIn("falls back to the legacy path", launch)
        self.assertIn(
            "reject inherited single-allocation environment", launch
        )
        self.assertIn("SFORGE_SINGLE_GPU_HANDOFF=0", launch)
        self.assertIn("SFORGE_ACCELERATOR_OBSERVER", launch)

    def test_prepare_submit_is_cooperative_and_fails_closed(self) -> None:
        script = ROOT / "images/work/public_tests/prepare_submit.sh"
        source = script.read_text(encoding="utf-8")
        self.assertIn("/dev/nvidia", source)
        self.assertIn("kill -TERM", source)
        self.assertIn("kill -KILL", source)
        self.assertIn("exit 75", source)
        self.assertIn("not host idle attestation", source)
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ)
            env["SFORGE_GPU_PROC_ROOT"] = directory
            result = subprocess.run(
                ["bash", str(script)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("drain complete", result.stdout)

    def test_agent_must_cleanup_immediately_before_submit(self) -> None:
        command = "sforge-submit --details"
        agent_task = (ROOT / "images/work/AGENT_TASK.md").read_text(
            encoding="utf-8"
        )
        readme = (
            ROOT
            / "images/work/workdir/flash_kmeans_lab/README_AGENT.md"
        ).read_text(encoding="utf-8")
        self.assertIn(command, agent_task)
        self.assertIn(command, readme)
        self.assertIn("automatically invokes", agent_task)
        common = (ROOT / "scripts/run_fixed_budget.sh").read_text(
            encoding="utf-8"
        )
        task = json.loads(
            (
                ROOT
                / "tasks/flash_kmeans_paper_h100_kernel_lab_v7_scratch.json"
            ).read_text(encoding="utf-8")
        )
        hook = "/app/workdir/flash_kmeans_lab/.sforge/prepare_submit.sh"
        self.assertEqual(task["cwd"], "/app/workdir/flash_kmeans_lab")
        self.assertTrue(hook.startswith(task["cwd"] + "/"))
        self.assertIn(f"SFORGE_PRE_SUBMIT_HOOK={hook}", common)
        dockerfile = (ROOT / "images/work/Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"COPY public_tests/prepare_submit.sh {hook}", dockerfile)
        self.assertIn("chown -R root:root", dockerfile)
        self.assertIn("allocated CUDA accelerator", agent_task)
        self.assertNotIn("Optimize one K-Means iteration on H100", agent_task)


if __name__ == "__main__":
    unittest.main()


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
         "--task-id", "t", "--expected-archive-sha256", "a" * 64],
        capture_output=True, text=True, check=False,
    )
    assert rejected.returncode == 3
    record = json.loads(rejected.stdout)
    assert record["status"] == "invalid_final" and record["phase"] == "post"
    assert record["reason"] and record["error_type"]
    assert 'ERROR: ' + record["reason"] in rejected.stderr
    for forbidden in ("evaluated_archive", "final_report", "private_judge_artifact"):
        assert forbidden not in record
    run = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    assert "record_invalid_final() {" in run
    assert 'status: "failed",\n      failure_stage: "formal_postcheck",' in run
    assert "sed -n 's/^ERROR: //p'" in run
    assert '.status == "invalid_final"' in run and '.status != "invalid_final"' in run
    postcheck_call = run.index("record_invalid_final \"${postcheck_rc}\"")
    assert postcheck_call < run.index('mv -f -- "${postcheck_stdout}" "${formal_postcheck}"')
    # The helper that discarded the temporary file is no longer on the postcheck path.
    assert "write_command_json_atomically \\\n  \"${formal_postcheck}\"" not in run
