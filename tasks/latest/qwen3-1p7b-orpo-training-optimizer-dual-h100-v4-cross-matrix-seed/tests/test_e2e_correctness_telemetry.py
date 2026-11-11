from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import TASK_ROOT


CALIBRATION_PATH = TASK_ROOT / "host_reference/run_paired_calibration.py"


@pytest.fixture()
def calibration_module():
    name = f"orpo_paired_calibration_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, CALIBRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        warmup_steps=5,
        timed_steps=10,
        correctness_rtol=None,
        correctness_atol=None,
    )


def _telemetry() -> dict[str, object]:
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
    return {
        "schema_version": 1,
        "capture_point": "warmup_after_backward_before_optimizer_step",
        "capture_warmup_step_index": 3,
        "capture_expected_optimizer_step": 4,
        "memory_qualification": "settled_after_final_plain_warmup",
        "post_capture_allocator_settled": True,
        "optimizer_class": "AdamW",
        "aux": [
            {
                "name": name,
                "shape_per_rank": [4] if index < 2 or 5 <= index <= 6 else [],
                "dtype_per_rank": "torch.float32",
                "global_numel": 8 if index < 2 or 5 <= index <= 6 else 2,
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
            "global_trainable_parameter_elements": 100,
            "global_gradient_tensors": 10,
            "global_gradient_elements": 100,
            "finite": True,
        },
        "optimizer_update": {
            "method": "trusted_AdamW_state_and_compact_parameter_signature",
            "expected_step": 4,
            "global_active_parameter_tensors": 10,
            "global_state_entries": 10,
            "global_entries_at_expected_step": 10,
            "minimum_step": 4.0,
            "maximum_step": 4.0,
            "global_exp_avg_l2_norm": 0.1,
            "parameter_before": {
                "global_numel": 100,
                "global_sum": 2.0,
                "global_l2_sq": 4.0,
            },
            "parameter_after": {
                "global_numel": 100,
                "global_sum": 1.9,
                "global_l2_sq": 3.9,
            },
            "parameter_global_sum_delta": -0.1,
            "parameter_global_l2_sq_delta": -0.1,
            "parameter_signature_changed": True,
            "finite": True,
        },
        "final_optimizer_update": {
            "capture_point": "after_timed_window_and_peak_reads",
            "method": "trusted_AdamW_final_state_and_compact_parameter_signature",
            "expected_step": 15,
            "global_active_parameter_tensors": 10,
            "global_state_entries": 10,
            "global_entries_at_expected_step": 10,
            "minimum_step": 15.0,
            "maximum_step": 15.0,
            "global_exp_avg_l2_norm": 0.2,
            "global_exp_avg_sq_l2_norm": 0.03,
            "parameter_at_capture_after": {
                "global_numel": 100,
                "global_sum": 1.9,
                "global_l2_sq": 3.9,
            },
            "parameter_final": {
                "global_numel": 100,
                "global_sum": 1.8,
                "global_l2_sq": 3.8,
            },
            "cumulative_parameter_global_sum_delta": 1.8 - 1.9,
            "cumulative_parameter_global_l2_sq_delta": 3.8 - 3.9,
            "cumulative_parameter_signature_changed": True,
            "finite": True,
        },
    }


def test_nonformal_calibration_requires_all_task_owned_correctness_evidence(
    calibration_module,
) -> None:
    payload = {"correctness_telemetry": _telemetry()}
    normalized = calibration_module._validate_correctness_telemetry(payload, _args())
    assert len(normalized["aux"]) == 9
    assert normalized["gradients"]["global_l2_norm"] == 3.0
    assert normalized["final_optimizer_update"]["expected_step"] == 15

    payload["correctness_telemetry"]["optimizer_update"]["global_entries_at_expected_step"] = 9
    with pytest.raises(RuntimeError, match="optimizer-update evidence"):
        calibration_module._validate_correctness_telemetry(payload, _args())


def test_calibration_rejects_final_optimizer_witness_without_exp_avg_sq(
    calibration_module,
) -> None:
    payload = {"correctness_telemetry": _telemetry()}
    payload["correctness_telemetry"]["final_optimizer_update"][
        "global_exp_avg_sq_l2_norm"
    ] = 0.0
    with pytest.raises(RuntimeError, match="final optimizer-update witness"):
        calibration_module._validate_correctness_telemetry(payload, _args())


def test_pair_comparator_records_numeric_deltas_without_inventing_formal_tolerance(
    calibration_module,
) -> None:
    reference = {"correctness_telemetry": _telemetry()}
    candidate = {"correctness_telemetry": _telemetry()}
    candidate["correctness_telemetry"]["aux"][0]["global_sum"] = 1.25
    report = calibration_module._compare_correctness(reference, candidate, _args())
    assert report["mode"] == "structural_and_finite_only_pending_freeze"
    assert any(row["absolute_error"] == 0.25 for row in report["numeric_deltas"])


@pytest.mark.parametrize(
    ("order", "expected"),
    (
        (
            "reference-candidate",
            (("reference", "reference"), ("reference", "candidate")),
        ),
        (
            "candidate-reference",
            (("reference", "candidate"), ("reference", "reference")),
        ),
    ),
)
def test_hf_self_launches_two_hf_workers_but_reverses_comparator_labels(
    calibration_module,
    order: str,
    expected: tuple[tuple[str, str], ...],
) -> None:
    assert calibration_module._launch_plan("hf_self", order) == expected


def test_treatment_launch_plan_keeps_worker_role_and_label_aligned(
    calibration_module,
) -> None:
    assert calibration_module._launch_plan("full_liger", "candidate-reference") == (
        ("full_liger", "candidate"),
        ("reference", "reference"),
    )


def test_one_warmup_telemetry_is_explicitly_diagnostic_not_memory_qualified(
    calibration_module,
) -> None:
    args = SimpleNamespace(
        warmup_steps=1,
        timed_steps=2,
        correctness_rtol=None,
        correctness_atol=None,
    )
    payload = {"correctness_telemetry": _telemetry()}
    telemetry = payload["correctness_telemetry"]
    telemetry["capture_warmup_step_index"] = 0
    telemetry["capture_expected_optimizer_step"] = 1
    telemetry["memory_qualification"] = "diagnostic_capture_on_only_warmup"
    telemetry["post_capture_allocator_settled"] = False
    telemetry["optimizer_update"]["expected_step"] = 1
    telemetry["optimizer_update"]["minimum_step"] = 1.0
    telemetry["optimizer_update"]["maximum_step"] = 1.0
    telemetry["final_optimizer_update"]["expected_step"] = 3
    telemetry["final_optimizer_update"]["minimum_step"] = 3.0
    telemetry["final_optimizer_update"]["maximum_step"] = 3.0
    normalized = calibration_module._validate_correctness_telemetry(payload, args)
    assert normalized["optimizer_update"]["expected_step"] == 1
