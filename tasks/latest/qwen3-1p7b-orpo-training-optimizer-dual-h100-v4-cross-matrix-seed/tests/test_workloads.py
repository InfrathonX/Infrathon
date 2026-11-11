from __future__ import annotations

import ast
import importlib.util
import json

from conftest import TASK_ROOT, judge_only_file


def load_public_eval():
    path = TASK_ROOT / "images/work/public_tests/public_eval.py"
    spec = importlib.util.spec_from_file_location("orpo_public_eval_workload_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_private_cases_are_disjoint_and_nonempty():
    public = json.loads((TASK_ROOT / "images/work/public_tests/public_cases.json").read_text())
    hidden = json.loads(judge_only_file().read_text())
    visible = public["correctness"] + public["gpu_objective"]
    private = hidden["cases"]
    assert visible and private
    assert {row["case_id"] for row in visible}.isdisjoint(row["case_id"] for row in private)
    assert {row["seed"] for row in visible}.isdisjoint(row["seed"] for row in private)
    for row in visible + private:
        assert row["pair_batch"] > 0
        assert row["sequence_length"] > 1
        assert row["mask_stride"] > 1
        assert row["ignore_index"] == -100
        assert row["separate_nll_target"] is True
        assert row["nll_target_contract"] == "separate_full_sequence_tensor"


def test_private_judge_plan_is_frozen_v4_cross_matrix():
    hidden = json.loads(judge_only_file().read_text())
    assert hidden["schema_version"] == 4
    assert hidden["formal_ready"] is True
    assert hidden["formal_score_enabled"] is True
    assert hidden["workload_plan_status"] == "FROZEN_FORMAL_CROSS_MATRIX"
    assert hidden["formal_thresholds_frozen"] is True
    assert hidden["role_legend"] == {
        "R": "official_hf_trl_reference",
        "C": "candidate",
    }
    assert "frozen five-row sequence-length and pair-batch cross matrix" in hidden["reason"]
    assert "host_only_qualification" not in hidden


def test_public_gpu_suite_has_large_real_training_path_row():
    public = json.loads((TASK_ROOT / "images/work/public_tests/public_cases.json").read_text())
    rows = [
        row
        for row in public["gpu_objective"]
        if row.get("workload_path") == "qwen3_full_training_objective"
    ]
    assert rows
    for row in rows:
        assert row["pair_batch"] == 3
        assert row["sequence_length"] == 383
        assert row["hidden_size"] == 2048
        assert row["vocab_size"] == 151936
        assert row["bias"] is False
        assert row["separate_nll_target"] is True
        assert row["nll_target_contract"] == "separate_full_sequence_tensor"
        assert row["formal"] is False
        assert row["score_eligible"] is False
        assert row["classification"] == "PUBLIC_OBJECTIVE_STRESS_DIAGNOSTIC_ONLY"


def test_public_gpu_protocol_requires_separate_full_sequence_nll_target():
    public = json.loads((TASK_ROOT / "images/work/public_tests/public_cases.json").read_text())
    row = next(
        item
        for item in public["gpu_objective"]
        if item.get("workload_path") == "qwen3_full_training_objective"
    )
    protocol = load_public_eval().gpu_row_protocol(row)
    rendered = json.loads(json.dumps(protocol, allow_nan=False))
    assert rendered["formal"] is False
    assert rendered["score_eligible"] is False
    assert rendered["workload_path"] == "qwen3_full_training_objective"
    assert rendered["dtype"] == "bfloat16"
    assert (
        rendered["input_contract"]["nll_target"]
        == "separate_full_sequence_tensor"
    )
    assert rendered["input_contract"]["weight_shape"] == [151936, 2048]
    assert rendered["input_contract"]["hidden_states_shape"] == [6, 383, 2048]
    assert rendered["input_contract"]["labels_shape"] == [6, 383]


def test_public_objective_inputs_never_fall_back_to_or_labels_for_nll():
    public = json.loads((TASK_ROOT / "images/work/public_tests/public_cases.json").read_text())
    case = public["correctness"][0]
    module = load_public_eval()
    _, _, labels, _, nll_target = module.make_inputs(
        case, module.torch.device("cpu"), module.torch.float32
    )
    assert nll_target is not None
    assert nll_target.data_ptr() != labels.data_ptr()
    assert not nll_target.eq(case["ignore_index"]).any()
    assert labels.eq(case["ignore_index"]).any()


def test_hidden_e2e_rows_are_exactly_the_frozen_five_row_cross_matrix():
    hidden = json.loads(judge_only_file().read_text())
    assert hidden["objective_stress_rows"] == []

    e2e = hidden["e2e_rows"]
    assert {(row["pair_batch_per_rank"], row["sequence_length"]) for row in e2e} == {
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
        assert row["seed"] == 20260814
        assert row["loss_tolerance"]["frozen"] is True
        assert row["correctness_tolerance"]["frozen"] is True
        assert row["memory_gate"]["frozen"] is True
        assert row["dataset_partition"] == "private"
        assert row["world_size"] == 2
        assert row["nll_target_contract"] == "separate_full_sequence_tensor"
        assert row["warmup_steps"] == 5
        assert row["timed_steps"] == 10
        assert row["paired_repeats"] == 4
        assert row["paired_role_order"] == ["RC", "CR", "CR", "RC"]
        records_required = (
            (row["warmup_steps"] + row["timed_steps"])
            * row["world_size"]
            * row["pair_batch_per_rank"]
        )
        assert records_required <= 240
        assert row["dataset_subdir"] == f"private_cross_{row['sequence_length']}"
        assert row["dataset_artifact_status"] == "FROZEN"
        assert len(row["dataset_manifest_sha256"]) == 64
        assert len(row["tensors_sha256"]) == 64

    # Scored row contracts do not select a host-only implementation.  The
    # separate Judge-private provenance block may name the qualification arm.
    scored_contract = json.dumps(e2e, sort_keys=True).lower()
    assert "liger" not in scored_contract
    assert "oracle" not in scored_contract


def test_frozen_loader_reports_the_validated_artifact_sequence_length():
    source = (
        TASK_ROOT
        / "images/work/workdir/orpo_lab/runtime/train_step.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    loader = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "load_frozen_preference_dataset"
    )
    dataset_returns = [
        node.value
        for node in ast.walk(loader)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "FrozenPreferenceDataset"
    ]
    assert len(dataset_returns) == 1
    sequence_length = next(
        keyword.value
        for keyword in dataset_returns[0].keywords
        if keyword.arg == "sequence_length"
    )
    assert isinstance(sequence_length, ast.Name)
    assert sequence_length.id == "max_length"
