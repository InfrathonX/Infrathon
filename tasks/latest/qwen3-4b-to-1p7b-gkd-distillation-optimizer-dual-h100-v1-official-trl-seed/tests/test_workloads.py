from __future__ import annotations

import ast
import importlib.util
import json

from conftest import TASK_ROOT, hidden_cases_path


def load_public_eval():
    path = TASK_ROOT / "images/work/public_tests/public_eval.py"
    spec = importlib.util.spec_from_file_location("gkd_public_eval_workload_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_private_cases_are_disjoint_and_nonempty():
    public = json.loads((TASK_ROOT / "images/work/public_tests/public_cases.json").read_text())
    hidden = json.loads(hidden_cases_path().read_text())
    visible = public["correctness"] + public["gpu_objective"]
    private = hidden["cases"]
    assert visible and private
    assert {row["case_id"] for row in visible}.isdisjoint(row["case_id"] for row in private)
    assert {row["seed"] for row in visible}.isdisjoint(row["seed"] for row in private)
    for row in visible + private:
        assert row["batch"] > 0
        assert row["sequence_length"] > 1
        assert row["mask_stride"] > 1
        assert row["ignore_index"] == -100
        assert row["student_hidden_size"] > 0
        assert row["teacher_hidden_size"] > 0
        assert row["vocab_size"] > 1
        assert 0.0 <= row["beta"] <= 1.0
        assert row["temperature"] > 0.0
        assert isinstance(row["bias"], bool)


def test_private_judge_plan_is_frozen_v1_cross_matrix():
    hidden = json.loads(hidden_cases_path().read_text())
    assert hidden["schema_version"] == 4
    assert hidden["formal_ready"] is True
    assert hidden["formal_score_enabled"] is True
    assert hidden["workload_plan_status"] == "FROZEN_FORMAL_CROSS_MATRIX"
    assert hidden["formal_thresholds_frozen"] is True
    assert hidden["role_legend"] == {
        "R": "official_hf_trl_reference",
        "C": "candidate",
    }
    assert "v1 GKD cross-matrix data artifacts are frozen" in hidden["reason"]
    assert hidden["matrix_contract"]["unique_row_count"] == 5
    assert "host_only_qualification" not in hidden


def test_frozen_thresholds_carry_the_measurement_that_justifies_them():
    """A frozen threshold without its measurement is an unreviewable number."""

    hidden = json.loads(hidden_cases_path().read_text())
    calibration = hidden["calibration"]
    assert calibration["method"] == "official_reference_self_pair"
    assert calibration["schedule"] == "5_warmup_10_timed"
    assert calibration["launches_per_row"] == 4
    assert calibration["role_orders"] == ["RC", "CR", "CR", "RC"]
    assert len(calibration["per_row_throughput_ratio_median"]) == 5

    measured_peak = calibration["measured_peak_reserved_bytes"]
    worst_loss = calibration["worst_loss_absolute_error"]
    worst_reserved = calibration["worst_peak_reserved_ratio"]
    for row in hidden["e2e_rows"]:
        evidence = row["calibration_evidence"]
        assert evidence["peak_reserved_bytes_max"] <= measured_peak
        # Each gate must clear the noise that the same code produced against
        # itself; a gate tighter than its own reference is a false-failure trap.
        assert row["loss_tolerance"]["atol"] >= worst_loss * 4
        assert (
            row["memory_gate"]["max_candidate_over_reference_peak_reserved_ratio"]
            >= worst_reserved
        )
        assert (
            row["memory_gate"]["absolute_candidate_peak_reserved_bytes_cap"]
            > measured_peak
        )


def test_public_gpu_suite_has_large_real_distillation_path_row():
    public = json.loads((TASK_ROOT / "images/work/public_tests/public_cases.json").read_text())
    rows = [
        row
        for row in public["gpu_objective"]
        if row.get("workload_path") == "qwen3_full_distillation_objective"
    ]
    assert rows
    for row in rows:
        assert row["batch"] == 3
        assert row["sequence_length"] == 383
        assert row["student_hidden_size"] == 2048
        assert row["teacher_hidden_size"] == 2560
        assert row["vocab_size"] == 151936
        assert row["bias"] is False
        assert row["beta"] == 0.5
        assert row["temperature"] == 0.9
        assert row["formal"] is False
        assert row["score_eligible"] is False
        assert row["classification"] == "PUBLIC_OBJECTIVE_STRESS_DIAGNOSTIC_ONLY"


def test_public_gpu_protocol_pins_the_dual_model_gkd_input_contract():
    public = json.loads((TASK_ROOT / "images/work/public_tests/public_cases.json").read_text())
    row = next(
        item
        for item in public["gpu_objective"]
        if item.get("workload_path") == "qwen3_full_distillation_objective"
    )
    protocol = load_public_eval().gpu_row_protocol(row)
    rendered = json.loads(json.dumps(protocol, allow_nan=False))
    assert rendered["formal"] is False
    assert rendered["score_eligible"] is False
    assert rendered["workload_path"] == "qwen3_full_distillation_objective"
    assert rendered["dtype"] == "bfloat16"
    assert rendered["input_contract"]["teacher_requires_grad"] is False
    assert rendered["input_contract"]["beta"] == 0.5
    assert rendered["input_contract"]["temperature"] == 0.9
    assert rendered["input_contract"]["student_weight_shape"] == [151936, 2048]
    assert rendered["input_contract"]["teacher_weight_shape"] == [151936, 2560]
    assert rendered["input_contract"]["student_hidden_shape"] == [3, 383, 2048]
    assert rendered["input_contract"]["teacher_hidden_shape"] == [3, 383, 2560]
    assert rendered["input_contract"]["labels_shape"] == [3, 383]


def test_public_objective_inputs_mask_labels_and_keep_the_teacher_frozen():
    public = json.loads((TASK_ROOT / "images/work/public_tests/public_cases.json").read_text())
    case = public["correctness"][0]
    module = load_public_eval()
    inputs = module.make_inputs(case, module.torch.device("cpu"), module.torch.float32)
    (
        student_weight,
        student_hidden,
        teacher_weight,
        teacher_hidden,
        labels,
        student_bias,
        teacher_bias,
    ) = module.clone_trainable(inputs)
    assert labels.eq(case["ignore_index"]).any()
    assert not labels.eq(case["ignore_index"]).all(dim=-1).any()
    assert student_weight.requires_grad and student_hidden.requires_grad
    assert not teacher_weight.requires_grad and not teacher_hidden.requires_grad
    if case["bias"]:
        assert student_bias is not None and student_bias.requires_grad
        assert teacher_bias is not None and not teacher_bias.requires_grad
    else:
        assert student_bias is None and teacher_bias is None


def test_hidden_e2e_rows_are_exactly_the_frozen_five_row_cross_matrix():
    hidden = json.loads(hidden_cases_path().read_text())
    assert hidden["objective_stress_rows"] == []

    e2e = hidden["e2e_rows"]
    assert {(row["batch_per_rank"], row["sequence_length"]) for row in e2e} == {
        (2, 512),
        (2, 1024),
        (2, 2048),
        (4, 512),
        (8, 512),
    }
    for row in e2e:
        assert row["classification"] == "HIDDEN_TLDR_E2E_FORMAL"
        assert row["status"] == "FROZEN_FORMAL_CROSS_MATRIX"
        assert row["formal"] is True
        assert row["score_eligible"] is True
        assert row["seed"] == 20260822
        assert row["loss_tolerance"]["frozen"] is True
        assert row["correctness_tolerance"]["frozen"] is True
        assert row["memory_gate"]["frozen"] is True
        assert row["dataset_partition"] == "private"
        assert row["world_size"] == 2
        assert row["warmup_steps"] == 5
        assert row["timed_steps"] == 10
        assert row["paired_repeats"] == 4
        assert row["paired_role_order"] == ["RC", "CR", "CR", "RC"]
        records_required = (
            (row["warmup_steps"] + row["timed_steps"])
            * row["world_size"]
            * row["batch_per_rank"]
        )
        assert records_required <= 240
        assert row["dataset_subdir"] == f"private_gkd_{row['sequence_length']}"
        assert row["dataset_artifact_status"] == "FROZEN"
        assert len(row["dataset_manifest_sha256"]) == 64
        assert len(row["tensors_sha256"]) == 64

    # Row contracts do not select a host-only implementation.  The separate
    # Judge-private provenance block may name the qualification arm.
    row_contract = json.dumps(e2e, sort_keys=True).lower()
    assert "liger" not in row_contract
    assert "oracle" not in row_contract


def test_frozen_loader_reports_the_validated_artifact_sequence_length():
    source = (
        TASK_ROOT
        / "images/work/workdir/gkd_lab/runtime/train_step.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    loader = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "load_frozen_sft_dataset"
    )
    dataset_returns = [
        node.value
        for node in ast.walk(loader)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "FrozenSFTDataset"
    ]
    assert len(dataset_returns) == 1
    sequence_length = next(
        keyword.value
        for keyword in dataset_returns[0].keywords
        if keyword.arg == "sequence_length"
    )
    assert isinstance(sequence_length, ast.Name)
    assert sequence_length.id == "max_length"
