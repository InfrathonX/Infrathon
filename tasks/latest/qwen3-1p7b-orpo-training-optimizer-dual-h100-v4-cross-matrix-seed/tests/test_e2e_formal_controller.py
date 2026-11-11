from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from conftest import TASK_ROOT, judge_only_file


CONTROLLER_PATH = (
    TASK_ROOT / "images/judge/verifier/e2e_formal_controller.py"
)
# The mocked worker reports a device that clears the task's capability
# envelope (60 GiB floor, compute capability 8.0).  The model name is
# deliberately not an NVIDIA product: no gate may read it.
MOCK_DEVICE_TOTAL_MEMORY = 64 * 1024**3
MOCK_DEVICE_NAME = "Some Future Accelerator"
MOCK_DEVICE_CAPABILITY = "8.0"
CANDIDATE_UID = 12_345
CANDIDATE_GID = 12_345
SUPPLEMENTARY_GID = 23_456


@pytest.fixture()
def controller():
    name = f"mlsysforge_e2e_formal_controller_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, CONTROLLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def _hidden_manifest() -> dict[str, Any]:
    return json.loads(judge_only_file().read_text(encoding="utf-8"))


def _enabled_manifest() -> dict[str, Any]:
    manifest = copy.deepcopy(_hidden_manifest())
    manifest.update(
        formal_ready=True,
        formal_score_enabled=True,
        workload_plan_status="FROZEN_FORMAL_CROSS_MATRIX",
        formal_thresholds_frozen=True,
    )
    manifest["host_only_qualification"] = {
        "classification": "JUDGE_PRIVATE_CALIBRATION_ATTESTATION",
        "status": "QUALIFIED_OFFICIAL_HF_TRL_REFERENCE",
        "reference_role": "official_hf_trl_reference",
        "reference_provenance": {
            "package": "trl",
            "implementation": "trl.ORPOTrainer",
            "version": "test-only-pinned-version",
            "source_sha256": "a" * 64,
        },
        "raw_artifact_roots": {
            "official_hf_trl_self": "/private/test/official-hf-trl-self",
            "full_liger": "/private/test/full-liger",
        },
        "paired_report_sha256": {
            "official_hf_trl_self": [
                {"file": "official-self.paired.json", "sha256": "b" * 64}
            ],
            "full_liger": [
                {"file": "full-liger.paired.json", "sha256": "c" * 64}
            ],
        },
        "threshold_rationale": {
            "basis": "OFFICIAL_HF_TRL_SELF_NOISE_AND_FULL_LIGER_ATTAINMENT",
            "official_hf_trl_self_report_count": 1,
            "full_liger_report_count": 1,
            "all_correctness_witnesses_passed": True,
        },
    }
    for row in manifest["e2e_rows"]:
        row["dataset_artifact_status"] = "FROZEN"
        if row["dataset_manifest_sha256"] is None:
            row["dataset_manifest_sha256"] = "d" * 64
        if row["tensors_sha256"] is None:
            row["tensors_sha256"] = "e" * 64
        row.update(
            classification="HIDDEN_TLDR_E2E_FORMAL",
            status="FROZEN_FORMAL_CROSS_MATRIX",
            formal=True,
            score_eligible=True,
            loss_tolerance={"frozen": True, "rtol": 0.005, "atol": 0.005},
            correctness_tolerance={
                "frozen": True,
                "aux": {"rtol": 0.06, "atol": 0.015},
                "gradient": {"rtol": 0.02, "atol": 0.01},
                "optimizer": {"rtol": 0.70, "atol": 1e-7},
            },
            memory_gate={
                "frozen": True,
                "max_candidate_over_reference_peak_reserved_ratio": 1.0,
                "absolute_candidate_peak_reserved_bytes_cap": 2_000,
            },
        )
    return manifest


def _pending_manifest() -> dict[str, Any]:
    manifest = copy.deepcopy(_hidden_manifest())
    manifest.update(
        formal_ready=False,
        formal_score_enabled=False,
        workload_plan_status="PENDING_CROSS_MATRIX_ARTIFACTS_AND_CALIBRATION",
        formal_thresholds_frozen=False,
    )
    manifest["host_only_qualification"] = {
        "classification": "JUDGE_PRIVATE_CALIBRATION_ATTESTATION",
        "status": "PENDING_V4_CROSS_MATRIX_QUALIFICATION",
        "reference_role": "official_hf_trl_reference",
        "recalibration_required": True,
    }
    pending = {
        "frozen": False,
        "status": "PENDING_OFFICIAL_HF_TRL_RECALIBRATION",
    }
    for row in manifest["e2e_rows"]:
        row.update(
            classification="HIDDEN_TLDR_E2E_CALIBRATION_ONLY",
            status="PENDING_V4_CROSS_MATRIX_QUALIFICATION",
            formal=False,
            score_eligible=False,
            loss_tolerance=copy.deepcopy(pending),
            correctness_tolerance=copy.deepcopy(pending),
            memory_gate=copy.deepcopy(pending),
        )
    return manifest


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _correctness_telemetry(
    pair_batch: int, warmup_steps: int, timed_steps: int
) -> dict[str, Any]:
    vector_names = {
        "chosen_logps",
        "rejected_logps",
        "chosen_rewards",
        "rejected_rewards",
    }
    names = (
        "chosen_logps",
        "rejected_logps",
        "chosen_logits_mean",
        "rejected_logits_mean",
        "nll_loss",
        "chosen_rewards",
        "rejected_rewards",
        "logsigmoid_ratio_mean",
        "log_odds_mean",
    )
    capture_index = warmup_steps - 2
    expected_step = capture_index + 1
    parameter_elements = 1_720_574_976
    return {
        "schema_version": 1,
        "capture_point": "warmup_after_backward_before_optimizer_step",
        "capture_warmup_step_index": capture_index,
        "capture_expected_optimizer_step": expected_step,
        "memory_qualification": "settled_after_final_plain_warmup",
        "post_capture_allocator_settled": True,
        "optimizer_class": "AdamW",
        "aux": [
            {
                "name": name,
                "shape_per_rank": [pair_batch] if name in vector_names else [],
                "dtype_per_rank": (
                    "torch.bfloat16"
                    if name in {"chosen_logits_mean", "rejected_logits_mean"}
                    else "torch.float32"
                ),
                "global_numel": 2 * pair_batch if name in vector_names else 2,
                "global_sum": float(index + 1),
                "global_l2_sq": float(index + 2),
                "global_min": -float(index + 1),
                "global_max": float(index + 1),
                "finite": True,
            }
            for index, name in enumerate(names)
        ],
        "gradients": {
            "method": "FSDP.clip_grad_norm_(max_norm=inf,norm_type=2)",
            "global_l2_norm": 3.0,
            "global_trainable_parameter_tensors": 10,
            "global_trainable_parameter_elements": parameter_elements,
            "global_gradient_tensors": 10,
            "global_gradient_elements": parameter_elements,
            "finite": True,
        },
        "optimizer_update": {
            "method": "trusted_AdamW_state_and_compact_parameter_signature",
            "expected_step": expected_step,
            "global_active_parameter_tensors": 10,
            "global_state_entries": 10,
            "global_entries_at_expected_step": 10,
            "minimum_step": float(expected_step),
            "maximum_step": float(expected_step),
            "global_exp_avg_l2_norm": 0.1,
            "parameter_before": {
                "global_numel": parameter_elements,
                "global_sum": 2.0,
                "global_l2_sq": 4.0,
            },
            "parameter_after": {
                "global_numel": parameter_elements,
                "global_sum": 1.9,
                "global_l2_sq": 3.9,
            },
            "parameter_global_sum_delta": 1.9 - 2.0,
            "parameter_global_l2_sq_delta": 3.9 - 4.0,
            "parameter_signature_changed": True,
            "finite": True,
        },
        "final_optimizer_update": {
            "capture_point": "after_timed_window_and_peak_reads",
            "method": "trusted_AdamW_final_state_and_compact_parameter_signature",
            "expected_step": warmup_steps + timed_steps,
            "global_active_parameter_tensors": 10,
            "global_state_entries": 10,
            "global_entries_at_expected_step": 10,
            "minimum_step": float(warmup_steps + timed_steps),
            "maximum_step": float(warmup_steps + timed_steps),
            "global_exp_avg_l2_norm": 0.2,
            "global_exp_avg_sq_l2_norm": 0.03,
            "parameter_at_capture_after": {
                "global_numel": parameter_elements,
                "global_sum": 1.9,
                "global_l2_sq": 3.9,
            },
            "parameter_final": {
                "global_numel": parameter_elements,
                "global_sum": 1.8,
                "global_l2_sq": 3.8,
            },
            "cumulative_parameter_global_sum_delta": 1.8 - 1.9,
            "cumulative_parameter_global_l2_sq_delta": 3.8 - 3.9,
            "cumulative_parameter_signature_changed": True,
            "finite": True,
        },
    }


class FakeProcessRunner:
    def __init__(
        self,
        *,
        bad_loss: bool = False,
        reused_identity: bool = False,
        imperative_hook: bool = False,
        candidate_no_patch: bool = False,
        correctness_fault: str | None = None,
        correctness_fault_role: str = "candidate",
        device_total_memory: int = MOCK_DEVICE_TOTAL_MEMORY,
        device_name: str = MOCK_DEVICE_NAME,
        device_capability: str = MOCK_DEVICE_CAPABILITY,
        peak_reserved_over_device_ceiling: bool = False,
    ) -> None:
        self.bad_loss = bad_loss
        self.reused_identity = reused_identity
        self.imperative_hook = imperative_hook
        self.candidate_no_patch = candidate_no_patch
        self.correctness_fault = correctness_fault
        self.correctness_fault_role = correctness_fault_role
        self.device_total_memory = device_total_memory
        self.device_name = device_name
        self.device_capability = device_capability
        self.peak_reserved_over_device_ceiling = peak_reserved_over_device_ceiling
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        command: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
        stdout: Any,
        stderr: Any,
        timeout_seconds: float,
        launch_options: dict[str, Any],
    ) -> int:
        call_index = len(self.calls)
        kind = _option(command, "--kind")
        pair_batch = int(_option(command, "--pair-batch"))
        warmup_steps = int(_option(command, "--warmup-steps"))
        timed_steps = int(_option(command, "--timed-steps"))
        target_throughput = (
            100.0 if kind == "reference" else (120.0 if pair_batch == 4 else 150.0)
        )
        scored = [100 for _ in range(timed_steps)]
        non_padding = [160 for _ in range(timed_steps)]
        durations = [sum(scored) / target_throughput / timed_steps] * timed_steps
        total_seconds = sum(durations)
        loss_offset = 0.25 if self.bad_loss and kind == "candidate" else 0.0
        warmup_losses = [2.0 - 0.01 * index + loss_offset for index in range(warmup_steps)]
        timed_losses = [1.9 - 0.01 * index + loss_offset for index in range(timed_steps)]
        if self.peak_reserved_over_device_ceiling:
            # Both roles sit just under the card's total memory, so the paired
            # ratio still passes and only the device-relative ceiling can fail.
            reserved = self.device_total_memory - (
                1 * 1024**3 if kind == "reference" else 2 * 1024**3
            )
            allocated = reserved - 1_000
        else:
            reserved = 1_000 if kind == "reference" else 800
            allocated = 800 if kind == "reference" else 600
        identity_index = 0 if self.reused_identity else call_index
        first_pid = 10_000 + 2 * identity_index
        batch_fingerprint = hashlib.sha256(f"batch-{pair_batch}".encode()).hexdigest()
        batch = {
            "source": "frozen_tldr",
            "partition": _option(command, "--expected-data-partition"),
            "manifest_sha256": _option(command, "--expected-manifest-sha256"),
            "tensors_sha256": _option(command, "--expected-tensors-sha256"),
            "partition_record_count": 10_000,
            "chosen_first": True,
            "nll_target_contract": "separate_full_sequence_tensor",
            "row_order_on_disk": "all_chosen_then_all_rejected",
            "selection": "disjoint_contiguous_records_by_global_step_then_rank",
            "pair_batch_per_rank": pair_batch,
            "sequence_length": int(_option(command, "--sequence-length")),
            "selected_unique_record_count": (
                (warmup_steps + timed_steps) * 2 * pair_batch
            ),
            "record_indices_by_rank_and_step": [[], []],
            "fingerprint_all_ranks": batch_fingerprint,
            "task_owned_token_counting": {
                "primary": "labels[:, 1:] != -100",
                "diagnostic": "attention_mask == 1",
            },
        }
        no_hook_report = {
            "abi": "Qwen3ModelPatch",
            "exported": False,
            "invoked": False,
            "return": "not_exported",
            "invariants_validated": False,
            "patch_spec": {"exported": False, "applied": False},
        }
        patch_report = {
            "abi": "Qwen3ModelPatch",
            "exported": True,
            "invoked": True,
            "return": "patch_spec_only",
            "invariants_validated": True,
            "patch_spec": {
                "exported": True,
                "applied": True,
                "fields": ["rope", "rms_norm_forward", "swiglu_forward"],
                "norm_module_count": 113,
                "mlp_module_count": 28,
                "attribute_writes_owned_by": "task_runtime",
            },
            "invariants": {
                "root_model_identity": True,
                "parameter_names_identity_shape_dtype_device_requires_grad": True,
                "buffer_names_identity_shape_dtype_device_requires_grad_value": True,
                "tied_input_output_weight_identity": True,
                "training_mode": True,
                "config_use_cache": True,
            },
        }
        if self.imperative_hook and kind == "candidate":
            patch_report["host_internal_optimizer"] = {
                "exported": True,
                "invoked": True,
                "return": "same_model",
            }
        correctness = _correctness_telemetry(
            pair_batch, warmup_steps, timed_steps
        )
        if kind == self.correctness_fault_role:
            if self.correctness_fault == "missing":
                correctness = None
            elif self.correctness_fault == "wrong_capture_step":
                correctness["capture_warmup_step_index"] += 1
            elif self.correctness_fault == "tampered_aux":
                correctness["aux"][0]["name"] = "tampered_chosen_logps"
            elif self.correctness_fault == "wrong_aux_shape":
                correctness["aux"][0]["shape_per_rank"] = []
                correctness["aux"][0]["global_numel"] = 2
            elif self.correctness_fault == "wrong_aux_dtype":
                correctness["aux"][0]["dtype_per_rank"] = "torch.bfloat16"
            elif self.correctness_fault == "zero_gradient":
                correctness["gradients"]["global_l2_norm"] = 0.0
            elif self.correctness_fault == "zero_exp_avg":
                correctness["optimizer_update"]["global_exp_avg_l2_norm"] = 0.0
            elif self.correctness_fault == "skipped_optimizer_step":
                correctness["optimizer_update"]["expected_step"] += 1
            elif self.correctness_fault == "zero_update":
                update = correctness["optimizer_update"]
                update["parameter_after"] = copy.deepcopy(update["parameter_before"])
                update["parameter_global_sum_delta"] = 0.0
                update["parameter_global_l2_sq_delta"] = 0.0
                update["parameter_signature_changed"] = False
            elif self.correctness_fault == "epsilon_update_spoof":
                update = correctness["optimizer_update"]
                update["parameter_after"] = copy.deepcopy(update["parameter_before"])
                update["parameter_global_sum_delta"] = 1e-13
                update["parameter_global_l2_sq_delta"] = 1e-13
                update["parameter_signature_changed"] = True
            elif self.correctness_fault == "missing_final_witness":
                del correctness["final_optimizer_update"]
            elif self.correctness_fault == "zero_final_exp_avg_sq":
                correctness["final_optimizer_update"][
                    "global_exp_avg_sq_l2_norm"
                ] = 0.0
            elif self.correctness_fault == "skipped_final_optimizer_step":
                correctness["final_optimizer_update"]["expected_step"] -= 1
            elif self.correctness_fault == "partial_final_optimizer_state":
                correctness["final_optimizer_update"][
                    "global_entries_at_expected_step"
                ] -= 1
            elif self.correctness_fault == "tampered_final_capture_signature":
                correctness["final_optimizer_update"][
                    "parameter_at_capture_after"
                ]["global_sum"] += 1.0
            elif self.correctness_fault == "zero_final_update":
                final = correctness["final_optimizer_update"]
                final["parameter_final"] = copy.deepcopy(
                    final["parameter_at_capture_after"]
                )
                final["cumulative_parameter_global_sum_delta"] = 0.0
                final["cumulative_parameter_global_l2_sq_delta"] = 0.0
                final["cumulative_parameter_signature_changed"] = False
            elif self.correctness_fault == "epsilon_final_update_spoof":
                final = correctness["final_optimizer_update"]
                final["parameter_final"] = copy.deepcopy(
                    final["parameter_at_capture_after"]
                )
                final["cumulative_parameter_global_sum_delta"] = 1e-13
                final["cumulative_parameter_global_l2_sq_delta"] = 1e-13
                final["cumulative_parameter_signature_changed"] = True
            elif self.correctness_fault == "numeric_aux_mismatch":
                correctness["aux"][0]["global_sum"] += 1.0
            elif self.correctness_fault == "numeric_final_mismatch":
                correctness["final_optimizer_update"][
                    "global_exp_avg_sq_l2_norm"
                ] += 1.0
        payload = {
            "schema_version": 1,
            "status": "PASS",
            "classification": "PUBLIC_FULL_MODEL_SMOKE_DIAGNOSTIC_ONLY",
            "formal": False,
            "run_id": str(uuid.UUID(int=identity_index + 1)),
            "objective_kind": kind,
            "candidate_version": (
                int(_option(command, "--candidate-version"))
                if kind == "candidate"
                else None
            ),
            "model_optimization": (
                patch_report
                if kind == "candidate" and not self.candidate_no_patch
                else no_hook_report
            ),
            "world_size": 2,
            "fsdp": {
                "api": "FSDP1",
                "sharding_strategy": "FULL_SHARD",
                "use_orig_params": True,
                "parameter_dtype": "bfloat16",
                "reduce_dtype": "bfloat16",
            },
            "model": {
                "repo_id": "Qwen/Qwen3-1.7B",
                "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                "resolved_path": _option(command, "--model-path"),
                "logical_weight_bytes": 3_000,
                "physical_weight_bytes": 3_100,
                "shards": ["model-00001-of-00001.safetensors"],
                "unique_parameters": 1_720_574_976,
                "parameter_dtype": "torch.bfloat16",
                "tied_lm_head": True,
            },
            "batch": batch,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 1e-6,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
            **(
                {"correctness_telemetry": correctness}
                if correctness is not None
                else {}
            ),
            "warmup_steps": warmup_steps,
            "timed_steps": timed_steps,
            "warmup_rank_mean_losses": warmup_losses,
            "timed_rank_mean_losses": timed_losses,
            "rank_max_step_seconds": durations,
            "global_scored_tokens_per_step": scored,
            "global_non_padding_tokens_per_step": non_padding,
            "total_global_scored_tokens": sum(scored),
            "total_global_non_padding_tokens": sum(non_padding),
            "total_rank_max_seconds": total_seconds,
            "throughput_scored_tokens_per_second": sum(scored) / total_seconds,
            "throughput_non_padding_tokens_per_second_diagnostic": (
                sum(non_padding) / total_seconds
            ),
            "peak_allocated_bytes_max_rank": allocated,
            "peak_reserved_bytes_max_rank": reserved,
            "per_rank_memory": [
                {
                    "rank": rank,
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                }
                for rank in (0, 1)
            ],
            "rank_identities": [
                {
                    "rank": rank,
                    "pid": first_pid + rank,
                    "local_rank": rank,
                    "cuda_device_count": 2,
                    "device_name": self.device_name,
                    "compute_capability": self.device_capability,
                    "total_memory_bytes": self.device_total_memory,
                }
                for rank in (0, 1)
            ],
            "rank_completion": [
                {"rank": rank, "pid": first_pid + rank, "status": "PASS"}
                for rank in (0, 1)
            ],
        }
        output = Path(_option(command, "--output"))
        output.write_text(json.dumps(payload), encoding="utf-8")
        stdout.write(b"mock stdout\n")
        stderr.write(b"mock stderr\n")
        self.calls.append(
            {
                "command": command,
                "cwd": cwd,
                "environment": dict(environment),
                "timeout_seconds": timeout_seconds,
                "launch_options": dict(launch_options),
                "kind": kind,
                "pair_batch": pair_batch,
            }
        )
        return 0


def _execution_kwargs(tmp_path: Path, runner: FakeProcessRunner) -> dict[str, Any]:
    lab_root = tmp_path / "orpo_lab"
    model_path = tmp_path / "model"
    data_root = tmp_path / "data"
    scratch_parent = tmp_path / "scratch"
    bootstrap_dir = tmp_path / "trusted_bootstrap"
    for path in (
        lab_root,
        model_path,
        data_root / "private_cross_512",
        data_root / "private_cross_1024",
        data_root / "private_cross_2048",
        scratch_parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir()
    rank_bootstrap = bootstrap_dir / "e2e_rank_bootstrap.py"
    rank_bootstrap.write_text("# mock process runner does not execute this file\n")
    rank_bootstrap.chmod(0o555)
    bootstrap_dir.chmod(0o700)
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    return {
        "lab_root": lab_root,
        "model_path": model_path,
        "data_root": data_root,
        "log_dir": tmp_path / "root_logs",
        "rank_bootstrap": rank_bootstrap,
        "candidate_version": 7,
        "candidate_environment": environment,
        "reference_environment": environment,
        "candidate_launch_options": {
            "user": CANDIDATE_UID,
            "group": CANDIDATE_GID,
            "extra_groups": (),
            "umask": 0o077,
            "start_new_session": True,
            "close_fds": True,
        },
        "timeout_seconds": 61.0,
        "process_runner": runner,
        "scratch_parent": scratch_parent,
    }


def test_current_hidden_plan_is_activation_ready(controller):
    plan = controller.inspect_e2e_plan(_hidden_manifest())
    assert plan["schema_ok"] is True
    assert plan["formal_requested"] is True
    assert plan["activation_ready"] is True
    assert plan["pending_gates"] == []


def test_explicit_pending_plan_returns_pending_without_launch(controller):
    def forbidden_runner(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("superseded formal plan must not launch")

    report = controller.run_e2e_controller(
        _pending_manifest(), process_runner=forbidden_runner
    )
    assert report["status"] == "PENDING_QUALIFICATION"
    assert report["gpu_launched"] is False
    assert report["formal_ready"] is False
    assert report["formal_score_enabled"] is False
    assert report["score"] == 0.0
    assert report["reference_role"] == "official_hf_trl_reference"


def test_fresh_official_hf_trl_fixture_is_activation_ready(controller):
    plan = controller.inspect_e2e_plan(_enabled_manifest())
    assert plan["formal_requested"] is True
    assert plan["activation_ready"] is True
    assert plan["pending_gates"] == []
    assert plan["reference_role"] == "official_hf_trl_reference"


def test_host_reference_attestation_is_not_part_of_runtime_manifest(controller):
    manifest = _enabled_manifest()
    manifest.pop("host_only_qualification", None)
    plan = controller.inspect_e2e_plan(manifest)
    assert plan["activation_ready"] is True
    assert plan["pending_gates"] == []


def test_enabled_plan_with_missing_absolute_cap_stays_fail_closed(controller):
    manifest = _enabled_manifest()
    del manifest["e2e_rows"][0]["memory_gate"][
        "absolute_candidate_peak_reserved_bytes_cap"
    ]

    def forbidden_runner(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("incomplete thresholds must not launch a subprocess")

    report = controller.run_e2e_controller(manifest, process_runner=forbidden_runner)
    assert report["status"] == "BLOCKED_CONFIGURATION"
    assert report["gpu_launched"] is False
    assert report["score"] == 0.0
    assert "e2e_rows[0].memory_gate.absolute_cap" in report["pending_gates"]


def test_enabled_plan_with_unfrozen_correctness_tolerance_never_launches(controller):
    manifest = _enabled_manifest()
    del manifest["e2e_rows"][0]["correctness_tolerance"]

    def forbidden_runner(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("unfrozen correctness thresholds must not launch")

    report = controller.run_e2e_controller(manifest, process_runner=forbidden_runner)
    assert report["status"] == "BLOCKED_CONFIGURATION"
    assert report["gpu_launched"] is False
    assert report["score"] == 0.0
    assert "e2e_rows[0].correctness_tolerance.frozen" in report["pending_gates"]


@pytest.mark.parametrize("invalid", [True, -1.0, math.inf, math.nan])
def test_invalid_grouped_correctness_tolerance_never_launches(
    controller, invalid: Any
):
    manifest = _enabled_manifest()
    manifest["e2e_rows"][0]["correctness_tolerance"]["aux"]["rtol"] = invalid

    def forbidden_runner(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("invalid correctness thresholds must not launch")

    report = controller.run_e2e_controller(manifest, process_runner=forbidden_runner)
    assert report["status"] == "BLOCKED_CONFIGURATION"
    assert report["gpu_launched"] is False
    assert report["score"] == 0.0
    assert "e2e_rows[0].correctness_tolerance.aux" in report["pending_gates"]


def test_formal_candidate_launch_rejects_supplementary_groups(tmp_path, controller):
    runner = FakeProcessRunner()
    kwargs = _execution_kwargs(tmp_path, runner)
    kwargs["candidate_launch_options"]["extra_groups"] = (SUPPLEMENTARY_GID,)
    with pytest.raises(PermissionError, match="supplementary groups"):
        controller.run_e2e_controller(_enabled_manifest(), **kwargs)
    assert runner.calls == []


def test_formal_controller_runs_fresh_counterbalanced_cohorts_and_scores(
    tmp_path, controller
):
    manifest = _enabled_manifest()
    runner = FakeProcessRunner()
    kwargs = _execution_kwargs(tmp_path, runner)
    report = controller.run_e2e_controller(manifest, **kwargs)

    assert report["status"] == "PASS"
    assert report["execution_ok"] is True
    assert report["loss_gate_ok"] is True
    assert report["memory_gate_ok"] is True
    assert report["correctness_evidence_ok"] is True
    assert len(runner.calls) == 40
    expected_roles = [
        "reference",
        "candidate",
        "candidate",
        "reference",
        "candidate",
        "reference",
        "reference",
        "candidate",
    ]
    for offset in range(0, 40, 8):
        assert [call["kind"] for call in runner.calls[offset : offset + 8]] == expected_roles
    expected_medians = [1.5, 1.5, 1.5, 1.2, 1.5]
    assert [row["median_ratio"] for row in report["rows"]] == pytest.approx(
        expected_medians
    )
    assert report["score"] == pytest.approx(
        100.0 * math.prod(expected_medians) ** (1.0 / len(expected_medians))
    )

    private_ids = [row["row_id"] for row in manifest["e2e_rows"]]
    qualification = manifest["host_only_qualification"]
    qualification_markers = list(qualification["raw_artifact_roots"].values()) + [
        entry["sha256"]
        for entries in qualification["paired_report_sha256"].values()
        for entry in entries
    ]
    for call in runner.calls:
        rendered_command = " ".join(call["command"])
        assert "hidden_cases.json" not in rendered_command
        assert all(private_id not in rendered_command for private_id in private_ids)
        assert all(marker not in rendered_command for marker in qualification_markers)
        assert all("hidden_cases" not in value for value in call["environment"].values())
        assert all(
            marker not in value
            for value in call["environment"].values()
            for marker in qualification_markers
        )
        assert call["cwd"] == kwargs["rank_bootstrap"].parent
        assert call["cwd"] != kwargs["lab_root"]
        assert "-I" in call["command"]
        assert "--module" not in call["command"]
        assert "--standalone" not in call["command"]
        assert "--master-addr=127.0.0.1" in call["command"]
        assert "--master-port=29500" in call["command"]
        if call["kind"] == "candidate":
            assert call["launch_options"]["user"] == CANDIDATE_UID
            assert call["launch_options"]["group"] == CANDIDATE_GID
            assert call["launch_options"]["extra_groups"] == ()
            assert call["launch_options"]["start_new_session"] is True
        else:
            assert "user" not in call["launch_options"]

    log_dir = kwargs["log_dir"]
    assert stat_mode(log_dir) == 0o700
    log_files = list(log_dir.iterdir())
    assert len(log_files) == 120
    assert all(stat_mode(path) == 0o600 for path in log_files)


def test_loss_mismatch_blocks_score_after_valid_mock_execution(tmp_path, controller):
    runner = FakeProcessRunner(bad_loss=True)
    report = controller.run_e2e_controller(
        _enabled_manifest(), **_execution_kwargs(tmp_path, runner)
    )
    assert report["status"] == "GATE_FAILED"
    assert report["execution_ok"] is True
    assert report["loss_gate_ok"] is False
    assert report["memory_gate_ok"] is True
    assert report["correctness_evidence_ok"] is True
    assert report["score"] == 0.0
    assert report["failed_gate_groups"] == ["loss"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "wrong_capture_step",
        "tampered_aux",
        "wrong_aux_shape",
        "wrong_aux_dtype",
        "zero_gradient",
        "zero_exp_avg",
        "skipped_optimizer_step",
        "zero_update",
        "epsilon_update_spoof",
        "missing_final_witness",
        "zero_final_exp_avg_sq",
        "skipped_final_optimizer_step",
        "partial_final_optimizer_state",
        "tampered_final_capture_signature",
        "zero_final_update",
        "epsilon_final_update_spoof",
    ],
)
def test_invalid_correctness_evidence_fails_closed(
    tmp_path, controller, fault: str
):
    runner = FakeProcessRunner(correctness_fault=fault)
    report = controller.run_e2e_controller(
        _enabled_manifest(), **_execution_kwargs(tmp_path, runner)
    )
    assert report["status"] == "FAIL"
    assert report["execution_ok"] is False
    assert report["correctness_evidence_ok"] is False
    assert report["score"] == 0.0
    assert report["error"] == "formal e2e execution failed"
    assert report["failure_stage"] == "telemetry_validation"
    assert report["failure_class"] == "invalid_telemetry"
    assert len(runner.calls) == 2


@pytest.mark.parametrize(
    ("role", "expected_call_count"),
    [("reference", 1), ("candidate", 2)],
)
def test_invalid_telemetry_reports_only_the_bounded_failure_role(
    tmp_path, controller, role: str, expected_call_count: int
):
    runner = FakeProcessRunner(
        correctness_fault="missing",
        correctness_fault_role=role,
    )
    report = controller.run_e2e_controller(
        _enabled_manifest(), **_execution_kwargs(tmp_path, runner)
    )

    assert report["failure_stage"] == "telemetry_validation"
    assert report["failure_class"] == "invalid_telemetry"
    assert report["failure_role"] == role
    assert set(report) >= {"failure_stage", "failure_class", "failure_role"}
    assert len(runner.calls) == expected_call_count


def test_frozen_absolute_peak_memory_cap_is_provenance_not_a_gate(
    tmp_path, controller
):
    """The manifest's absolute byte cap was measured on one card; it must not gate.

    Before the capability-envelope fix a candidate whose peak reserved bytes
    exceeded ``absolute_candidate_peak_reserved_bytes_cap`` failed the memory
    gate, so the same code scored differently on a card with a different total
    memory.  The number stays in the frozen manifest as machine-readable
    provenance, but the enforced ceiling is now derived from the device.
    """

    manifest = _enabled_manifest()
    for row in manifest["e2e_rows"]:
        row["memory_gate"]["absolute_candidate_peak_reserved_bytes_cap"] = 1
    runner = FakeProcessRunner()
    report = controller.run_e2e_controller(
        manifest, **_execution_kwargs(tmp_path, runner)
    )
    assert report["status"] == "PASS"
    assert report["memory_gate_ok"] is True
    assert report["score"] > 0.0
    ceiling = (
        MOCK_DEVICE_TOTAL_MEMORY - controller.DEVICE_MEMORY_HEADROOM_BYTES
    )
    assert all(
        row["device_peak_reserved_ceiling_bytes"] == ceiling
        for row in report["rows"]
    )


def test_peak_reserved_above_the_device_ceiling_fails_the_memory_gate(
    tmp_path, controller
):
    """The replacement ceiling is real: it is total memory minus the headroom."""

    runner = FakeProcessRunner(peak_reserved_over_device_ceiling=True)
    report = controller.run_e2e_controller(
        _enabled_manifest(), **_execution_kwargs(tmp_path, runner)
    )
    assert report["status"] == "GATE_FAILED"
    assert report["execution_ok"] is True
    assert report["memory_gate_ok"] is False
    assert report["failed_gate_groups"] == ["memory"]
    assert report["score"] == 0.0


def test_the_device_ceiling_scales_with_the_card(tmp_path, controller):
    """The identical peak that failed on one card passes on a larger one."""

    over = MOCK_DEVICE_TOTAL_MEMORY - 2 * 1024**3
    runner = FakeProcessRunner(
        device_total_memory=MOCK_DEVICE_TOTAL_MEMORY + 8 * 1024**3,
        peak_reserved_over_device_ceiling=False,
    )
    report = controller.run_e2e_controller(
        _enabled_manifest(), **_execution_kwargs(tmp_path, runner)
    )
    assert report["memory_gate_ok"] is True
    assert report["rows"][0]["device_peak_reserved_ceiling_bytes"] > over


def test_the_paired_reference_ratio_still_gates_peak_memory(tmp_path, controller):
    """The device ceiling did not replace the same-run relative gate."""

    manifest = _enabled_manifest()
    for row in manifest["e2e_rows"]:
        row["memory_gate"]["max_candidate_over_reference_peak_reserved_ratio"] = 0.5
    runner = FakeProcessRunner()
    report = controller.run_e2e_controller(
        manifest, **_execution_kwargs(tmp_path, runner)
    )
    assert report["status"] == "GATE_FAILED"
    assert report["memory_gate_ok"] is False
    assert report["failed_gate_groups"] == ["memory"]


@pytest.mark.parametrize(
    "device_kwargs",
    [
        {"device_capability": "7.5"},
        {"device_total_memory": 16 * 1024**3},
    ],
)
def test_a_below_envelope_device_never_produces_a_score(
    tmp_path, controller, device_kwargs: dict[str, Any]
):
    """The controller re-checks the envelope on the trusted rank identity.

    The runtime refuses to start on such a device; this covers the case where
    a telemetry file claiming one is replayed into the controller.
    """

    runner = FakeProcessRunner(**device_kwargs)
    report = controller.run_e2e_controller(
        _enabled_manifest(), **_execution_kwargs(tmp_path, runner)
    )
    assert report["status"] != "PASS"
    assert report["failure_stage"] == "telemetry_validation"
    assert report["failure_class"] == "invalid_telemetry"
    assert report["score"] == 0.0


@pytest.mark.parametrize("fault", ["numeric_aux_mismatch", "numeric_final_mismatch"])
def test_correctness_numeric_mismatch_blocks_every_score(
    tmp_path, controller, fault: str
):
    runner = FakeProcessRunner(correctness_fault=fault)
    report = controller.run_e2e_controller(
        _enabled_manifest(), **_execution_kwargs(tmp_path, runner)
    )
    assert report["status"] == "GATE_FAILED"
    assert report["execution_ok"] is True
    assert report["loss_gate_ok"] is True
    assert report["memory_gate_ok"] is True
    assert report["correctness_evidence_ok"] is False
    assert report["score"] == 0.0
    assert report["failed_gate_groups"] == ["correctness_update"]
    assert len(runner.calls) == 40


def test_near_zero_warmup_l2_delta_is_structural_evidence_not_a_cross_impl_gate(
    controller,
):
    row = controller.inspect_e2e_plan(_enabled_manifest())["rows"][0]
    reference_raw = _correctness_telemetry(2, 5, 10)
    candidate_raw = copy.deepcopy(reference_raw)

    reference_update = reference_raw["optimizer_update"]
    reference_update["parameter_after"]["global_l2_sq"] = (
        reference_update["parameter_before"]["global_l2_sq"] - 2.27e-7
    )
    reference_update["parameter_global_l2_sq_delta"] = (
        reference_update["parameter_after"]["global_l2_sq"]
        - reference_update["parameter_before"]["global_l2_sq"]
    )
    candidate_update = candidate_raw["optimizer_update"]
    candidate_update["parameter_after"]["global_l2_sq"] = (
        candidate_update["parameter_before"]["global_l2_sq"] + 5.09e-7
    )
    candidate_update["parameter_global_l2_sq_delta"] = (
        candidate_update["parameter_after"]["global_l2_sq"]
        - candidate_update["parameter_before"]["global_l2_sq"]
    )
    for telemetry in (reference_raw, candidate_raw):
        update = telemetry["optimizer_update"]
        final = telemetry["final_optimizer_update"]
        final["parameter_at_capture_after"] = copy.deepcopy(
            update["parameter_after"]
        )
        final["cumulative_parameter_global_l2_sq_delta"] = (
            final["parameter_final"]["global_l2_sq"]
            - final["parameter_at_capture_after"]["global_l2_sq"]
        )

    reference = controller._validate_correctness_telemetry(
        {"correctness_telemetry": reference_raw}, row
    )
    candidate = controller._validate_correctness_telemetry(
        {"correctness_telemetry": candidate_raw}, row
    )
    comparison = controller._compare_correctness(reference, candidate, row)
    assert comparison == {"ok": True, "numeric_checks": 43}

    candidate["optimizer_update"]["parameter_global_sum_delta"] += 1.0
    assert controller._compare_correctness(reference, candidate, row)["ok"] is False


def test_worker_timeout_returns_only_safe_stage_and_class(tmp_path, controller):
    def timeout_runner(*_args, **_kwargs):
        raise controller.WorkerTimeoutError("root-only timeout detail")

    report = controller.run_e2e_controller(
        _enabled_manifest(), **_execution_kwargs(tmp_path, timeout_runner)
    )
    assert report["status"] == "FAIL"
    assert report["execution_ok"] is False
    assert report["failure_stage"] == "reference_worker"
    assert report["failure_class"] == "timeout"
    assert "failure_role" not in report
    assert "root-only timeout detail" not in json.dumps(report)


def test_seed_style_candidate_without_model_patch_remains_formal_eligible(
    tmp_path, controller
):
    runner = FakeProcessRunner(candidate_no_patch=True)
    report = controller.run_e2e_controller(
        _enabled_manifest(), **_execution_kwargs(tmp_path, runner)
    )
    assert report["status"] == "PASS"
    assert report["correctness_evidence_ok"] is True
    assert report["score"] > 0.0


def test_reused_worker_identity_fails_execution_without_error_detail(tmp_path, controller):
    runner = FakeProcessRunner(reused_identity=True)
    report = controller.run_e2e_controller(
        _enabled_manifest(), **_execution_kwargs(tmp_path, runner)
    )
    assert report["status"] == "FAIL"
    assert report["execution_ok"] is False
    assert report["score"] == 0.0
    assert report["error"] == "formal e2e execution failed"
    assert len(runner.calls) == 2


def test_imperative_model_optimizer_is_not_formal_eligible(tmp_path, controller):
    runner = FakeProcessRunner(imperative_hook=True)
    report = controller.run_e2e_controller(
        _enabled_manifest(), **_execution_kwargs(tmp_path, runner)
    )
    assert report["status"] == "FAIL"
    assert report["execution_ok"] is False
    assert report["score"] == 0.0
    assert len(runner.calls) == 2


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777
