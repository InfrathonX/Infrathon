from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest

from conftest import TASK_ROOT


CONTROLLER_PATH = TASK_ROOT / "images/judge/verifier/e2e_formal_controller.py"
STUDENT_PARAMETER_ELEMENTS = 1_720_574_976
TEACHER_PARAMETER_ELEMENTS = 4_022_468_096


@pytest.fixture()
def controller_module():
    name = f"gkd_e2e_correctness_telemetry_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, CONTROLLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def _row() -> dict[str, Any]:
    return {
        "warmup_steps": 5,
        "timed_steps": 10,
        "batch_per_rank": 4,
        "loss_rtol": 0.005,
        "loss_atol": 0.005,
        "memory_ratio_cap": 1.02,
        # Provenance only since the frozen byte cap left the gate path; the
        # enforced ceiling is derived from the device that ran the pair.
        "memory_absolute_cap_provenance": 88_046_829_568,
        "correctness_tolerance": {
            "aux": {"rtol": 0.06, "atol": 0.015},
            "gradient": {"rtol": 0.02, "atol": 0.01},
            "optimizer": {"rtol": 0.7, "atol": 1e-7},
        },
    }


def _telemetry() -> dict[str, object]:
    names = (
        "student_completion_logps",
        "teacher_completion_logps",
        "student_logits_mean",
        "teacher_logits_mean",
        "jsd_per_sequence",
        "kl_teacher_mixture_mean",
        "kl_student_mixture_mean",
        "student_entropy_mean",
        "teacher_entropy_mean",
    )
    vector_names = {
        "student_completion_logps",
        "teacher_completion_logps",
        "jsd_per_sequence",
    }
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
                "shape_per_rank": [4] if name in vector_names else [],
                "dtype_per_rank": (
                    "torch.bfloat16"
                    if name in {"student_logits_mean", "teacher_logits_mean"}
                    else "torch.float32"
                ),
                "global_numel": 8 if name in vector_names else 2,
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
            "global_l2_norm_fp64": 3.0000000001,
            "global_trainable_parameter_tensors": 10,
            "global_trainable_parameter_elements": STUDENT_PARAMETER_ELEMENTS,
            "global_gradient_tensors": 10,
            "global_gradient_elements": STUDENT_PARAMETER_ELEMENTS,
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
                "global_numel": STUDENT_PARAMETER_ELEMENTS,
                "global_sum": 2.0,
                "global_l2_sq": 4.0,
            },
            "parameter_after": {
                "global_numel": STUDENT_PARAMETER_ELEMENTS,
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
            "expected_step": 15,
            "global_active_parameter_tensors": 10,
            "global_state_entries": 10,
            "global_entries_at_expected_step": 10,
            "minimum_step": 15.0,
            "maximum_step": 15.0,
            "global_exp_avg_l2_norm": 0.2,
            "global_exp_avg_sq_l2_norm": 0.03,
            "parameter_at_capture_after": {
                "global_numel": STUDENT_PARAMETER_ELEMENTS,
                "global_sum": 1.9,
                "global_l2_sq": 3.9,
            },
            "parameter_final": {
                "global_numel": STUDENT_PARAMETER_ELEMENTS,
                "global_sum": 1.8,
                "global_l2_sq": 3.8,
            },
            "cumulative_parameter_global_sum_delta": 1.8 - 1.9,
            "cumulative_parameter_global_l2_sq_delta": 3.8 - 3.9,
            "cumulative_parameter_signature_changed": True,
            "finite": True,
        },
        "teacher_witness": {
            "method": "compact_frozen_parameter_signature_start_end_exact_equality",
            "signature": {
                "numel": TEACHER_PARAMETER_ELEMENTS,
                "sum": -1234.5,
                "l2_sq": 987.25,
            },
            "identical_across_ranks": True,
            "unchanged": True,
            "no_gradients": True,
            "trainable_parameter_tensors": 0,
            "finite": True,
        },
    }


def _role_payload(controller_module) -> dict[str, Any]:
    correctness = controller_module._validate_correctness_telemetry(
        {"correctness_telemetry": _telemetry()}, _row()
    )
    return {
        "model": {
            "repo_id": "Qwen/Qwen3-1.7B",
            "unique_parameters": STUDENT_PARAMETER_ELEMENTS,
        },
        "teacher_model": {
            "repo_id": "Qwen/Qwen3-4B",
            "unique_parameters": TEACHER_PARAMETER_ELEMENTS,
            "trainable": False,
        },
        "fsdp": {"api": "FSDP1", "sharding_strategy": "FULL_SHARD"},
        "batch": {"batch_per_rank": 2, "source": "frozen_tldr"},
        "scored_tokens": 100,
        "non_padding_tokens": 200,
        "correctness": correctness,
        "warmup_losses": [1.5] * 5,
        "timed_losses": [1.25] * 10,
        "peak_reserved": 1_000,
        "device_total_memory": 84 * 1024**3,
        "throughput": 10.0,
    }


def test_formal_validator_requires_all_task_owned_correctness_evidence(
    controller_module,
) -> None:
    payload = {"correctness_telemetry": _telemetry()}
    normalized = controller_module._validate_correctness_telemetry(payload, _row())
    assert len(normalized["aux"]) == 9
    assert [entry["name"] for entry in normalized["aux"]] == list(
        controller_module.EXPECTED_CORRECTNESS_AUX
    )
    assert normalized["gradients"]["global_l2_norm"] == 3.0
    assert normalized["optimizer_update"]["expected_step"] == 4
    assert normalized["final_optimizer_update"]["expected_step"] == 15

    payload["correctness_telemetry"]["optimizer_update"]["method"] = "handwritten"
    with pytest.raises(RuntimeError, match="optimizer-update evidence"):
        controller_module._validate_correctness_telemetry(payload, _row())

    payload = {"correctness_telemetry": _telemetry()}
    payload["correctness_telemetry"]["optimizer_update"][
        "global_entries_at_expected_step"
    ] = 9
    with pytest.raises(RuntimeError, match="AdamW state did not record"):
        controller_module._validate_correctness_telemetry(payload, _row())


def test_validator_rejects_final_optimizer_witness_without_exp_avg_sq(
    controller_module,
) -> None:
    payload = {"correctness_telemetry": _telemetry()}
    payload["correctness_telemetry"]["final_optimizer_update"][
        "global_exp_avg_sq_l2_norm"
    ] = 0.0
    with pytest.raises(RuntimeError, match="final optimizer-update witness"):
        controller_module._validate_correctness_telemetry(payload, _row())


def test_validator_normalizes_the_frozen_teacher_witness(controller_module) -> None:
    normalized = controller_module._validate_correctness_telemetry(
        {"correctness_telemetry": _telemetry()}, _row()
    )
    assert normalized["teacher_witness"] == {
        "numel": TEACHER_PARAMETER_ELEMENTS,
        "sum": -1234.5,
        "l2_sq": 987.25,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda witness: witness.update(unchanged=False), "frozen-teacher witness"),
        (lambda witness: witness.update(no_gradients=False), "frozen-teacher witness"),
        (
            lambda witness: witness.update(trainable_parameter_tensors=2),
            "frozen-teacher witness",
        ),
        (
            lambda witness: witness.update(identical_across_ranks=False),
            "frozen-teacher witness",
        ),
        (
            lambda witness: witness.update(
                method="compact_frozen_parameter_signature_sampled"
            ),
            "frozen-teacher witness",
        ),
        (lambda witness: witness.pop("signature"), "frozen-teacher signature"),
        (
            lambda witness: witness["signature"].update(
                numel=STUDENT_PARAMETER_ELEMENTS
            ),
            "frozen-teacher signature",
        ),
        (
            lambda witness: witness["signature"].update(l2_sq=0.0),
            "frozen-teacher signature",
        ),
    ),
)
def test_validator_gates_every_frozen_teacher_witness_field(
    controller_module, mutate, message: str
) -> None:
    payload = {"correctness_telemetry": _telemetry()}
    mutate(payload["correctness_telemetry"]["teacher_witness"])
    with pytest.raises(RuntimeError, match=message):
        controller_module._validate_correctness_telemetry(payload, _row())


def test_pair_comparator_applies_only_frozen_grouped_tolerances(
    controller_module,
) -> None:
    row = _row()
    reference = controller_module._validate_correctness_telemetry(
        {"correctness_telemetry": _telemetry()}, row
    )
    near = controller_module._validate_correctness_telemetry(
        {"correctness_telemetry": _telemetry()}, row
    )
    # aux[0] global_sum reference value is 1.0: 0.05 sits inside the frozen
    # 0.015 + 0.06 * |1.0| aux tolerance, 0.25 does not.
    near["aux"][0]["global_sum"] = 1.05
    passing = controller_module._compare_correctness(reference, near, row)
    assert passing["ok"] is True
    assert passing["numeric_checks"] == 41
    assert passing["failed_checks"] == []
    # Every reported check names itself so a maintainer can tell which witness
    # consumed the budget; the participant never sees this.
    assert all(item["check"] for item in passing["worst_checks"])

    far = controller_module._validate_correctness_telemetry(
        {"correctness_telemetry": _telemetry()}, row
    )
    far["aux"][0]["global_sum"] = 1.25
    failing = controller_module._compare_correctness(reference, far, row)
    assert failing["ok"] is False
    assert failing["numeric_checks"] == 41
    # The offending witness is named, so a failure is diagnosable rather than
    # just a red boolean.
    assert failing["failed_checks"] == ["aux.student_completion_logps.global_sum"]

    mismatched = controller_module._validate_correctness_telemetry(
        {"correctness_telemetry": _telemetry()}, row
    )
    mismatched["aux"][0]["dtype_per_rank"] = "torch.float64"
    with pytest.raises(RuntimeError, match="auxiliary structure differs"):
        controller_module._compare_correctness(reference, mismatched, row)


def test_pair_comparator_requires_exactly_equal_teacher_witness_across_roles(
    controller_module,
) -> None:
    row = _row()
    reference = _role_payload(controller_module)
    reference.update(run_id="reference-run", pids={101, 102})
    candidate = _role_payload(controller_module)
    candidate.update(run_id="candidate-run", pids={201, 202})

    report = controller_module._compare_pair(reference, candidate, row)
    assert report["loss_ok"] is True
    assert report["memory_ok"] is True
    assert report["correctness_evidence_ok"] is True
    assert report["ratio"] == 1.0

    # The frozen-teacher signature comparison is exact equality, never a
    # tolerance: even a sub-tolerance drift between the roles fails closed.
    drifted = _role_payload(controller_module)
    drifted.update(run_id="candidate-run", pids={201, 202})
    drifted["correctness"]["teacher_witness"]["sum"] = -1234.500000000001
    with pytest.raises(RuntimeError, match="frozen-teacher signatures differ"):
        controller_module._compare_pair(reference, drifted, row)


def test_pair_comparator_requires_fresh_rank_processes(controller_module) -> None:
    row = _row()
    reference = _role_payload(controller_module)
    reference.update(run_id="shared-run", pids={101, 102})
    candidate = _role_payload(controller_module)
    candidate.update(run_id="shared-run", pids={201, 202})
    with pytest.raises(RuntimeError, match="fresh rank processes"):
        controller_module._compare_pair(reference, candidate, row)

    candidate.update(run_id="candidate-run", pids={102, 202})
    with pytest.raises(RuntimeError, match="fresh rank processes"):
        controller_module._compare_pair(reference, candidate, row)


def test_diagnostic_only_warmup_capture_is_never_formal_evidence(
    controller_module,
) -> None:
    # The probe's one-warmup diagnostic mode labels its capture explicitly;
    # the formal validator accepts only the allocator-settled formal capture.
    payload = {"correctness_telemetry": _telemetry()}
    telemetry = payload["correctness_telemetry"]
    telemetry["memory_qualification"] = "diagnostic_capture_on_only_warmup"
    telemetry["post_capture_allocator_settled"] = False
    with pytest.raises(RuntimeError, match="re-settle the allocator"):
        controller_module._validate_correctness_telemetry(payload, _row())
