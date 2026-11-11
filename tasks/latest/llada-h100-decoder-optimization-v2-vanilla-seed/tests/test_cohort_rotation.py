from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile

import pytest

from conftest import private_data_dir


def load_cohort(task_root: Path):
    path = task_root / "images/judge/verifier/cohort.py"
    private = private_data_dir()
    if private != task_root / "images/judge/private_data":
        # cohort.py reads ../private_data/workload_policy.json at import: give
        # it the fetched Judge-only policy beside a copy of itself.
        staged = Path(tempfile.mkdtemp(prefix="llada-cohort-"))
        (staged / "verifier").mkdir()
        (staged / "private_data").mkdir()
        shutil.copyfile(path, staged / "verifier/cohort.py")
        shutil.copyfile(
            private / "workload_policy.json", staged / "private_data/workload_policy.json"
        )
        path = staged / "verifier/cohort.py"
    name = "h100_fullscale_cohort_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def synthetic_canonical(cohort):
    cases = []
    dataset_offsets = Counter()
    for row_id, plan in cohort.ROW_PLAN.items():
        for index in range(plan.canonical_count):
            # A target is assigned to exactly one score row within its dataset.
            record_id = (
                f"{plan.benchmark}:record:"
                f"{dataset_offsets[plan.benchmark] + index:04d}"
            )
            cases.append(
                {
                    "benchmark": plan.benchmark,
                    "row_id": row_id,
                    "case_id": f"{row_id}:{record_id}",
                    "record_id": record_id,
                }
            )
        dataset_offsets[plan.benchmark] += plan.canonical_count
    assert len(cases) == 520
    assert dataset_offsets == Counter({category: 130 for category in dataset_offsets})
    return cases


def test_final_selects_all_520_generations(task_root):
    cohort = load_cohort(task_root)
    cases = synthetic_canonical(cohort)
    selected = cohort.select_cohort(cases, cohort.EvaluationContext("final"))
    assert selected.mode == "final"
    assert selected.submission_number is None
    assert selected.cycle is None
    assert selected.slot is None
    assert list(selected.cases) == cases
    assert len(selected.cases) == 520
    assert dict(selected.row_counts) == {
        row_id: plan.canonical_count for row_id, plan in cohort.ROW_PLAN.items()
    }
    assert dict(selected.category_counts) == {
        "gsm8k": 130,
        "math": 130,
        "humaneval": 130,
        "mbpp": 130,
    }
    assert selected.bonus_categories == ()
    assert selected.row_windows == ()


def test_interactive_uses_identical_complete_frozen_final_workload(task_root):
    cohort = load_cohort(task_root)
    cases = synthetic_canonical(cohort)
    final = cohort.select_cohort(cases, cohort.EvaluationContext("final"))
    for submission, run_id, secret in ((1, "run-a", ""), (2, "run-a", "secret-a"), (99, "run-b", "secret-b")):
        selected = cohort.select_cohort(cases, cohort.EvaluationContext("interactive", submission, run_id, secret))
        assert selected.mode == "final"
        assert selected.submission_number == submission
        assert selected.cases == final.cases
        assert selected.row_counts == final.row_counts
        assert selected.category_counts == final.category_counts
        assert len(selected.cases) == len({case["case_id"] for case in selected.cases}) == 520
        assert selected.bonus_categories == ()
        assert selected.row_windows == ()
        assert selected.cycle is None and selected.slot is None


def test_category_is_explicitly_dataset_not_score_row(task_root):
    cohort = load_cohort(task_root)
    assert cohort.WORKLOAD_POLICY["interactive"]["category_definition"] == {
        "kind": "dataset",
        "field": "benchmark",
        "categories": ["gsm8k", "math", "humaneval", "mbpp"],
    }
    interactive = cohort.WORKLOAD_POLICY["interactive"]
    assert interactive["selection"] == "all_frozen_final_v1"
    assert interactive["total_cases"] == 520
    assert interactive["category_base_quota"] == 130
    assert interactive["bonus_categories"] == 0
    assert interactive["minimum_per_row"] == 1
    assert "per_row_quotas" not in interactive
    assert "per_category_quotas" not in interactive


def test_round_and_secret_cannot_change_frozen_membership_or_order(task_root):
    cohort = load_cohort(task_root)
    cases = synthetic_canonical(cohort)
    for round_id in ("agent-1", "agent-2", "agent-100"):
        metadata = {"round_id": round_id, "run_id": "trusted-run"}
        before = dict(metadata)
        context = cohort.context_from_metadata(metadata, "unused-secret")
        selected = cohort.select_cohort(cases, context)
        assert metadata == before
        assert list(selected.cases) == cases
        assert selected.submission_number == int(round_id.removeprefix("agent-"))


def test_root_only_metadata_context_is_fail_safe(task_root):
    cohort = load_cohort(task_root)
    assert cohort.context_from_metadata({}) == cohort.EvaluationContext("final")
    assert cohort.context_from_metadata(
        {cohort.ROUND_ID_KEY: "manual-1"}, calibration=True
    ) == cohort.EvaluationContext("final")
    assert cohort.context_from_metadata(
        {cohort.ROUND_ID_KEY: "agent-11", cohort.RUN_ID_KEY: "run-a"}, "secret-a"
    ) == cohort.EvaluationContext("final", 11, "run-a")
    for metadata, secret in (
        ({cohort.ROUND_ID_KEY: "agent-0"}, "secret-a"),
        ({cohort.ROUND_ID_KEY: "agent-01"}, "secret-a"),
        ({cohort.ROUND_ID_KEY: "agent-1"}, "secret-a"),

    ):
        with pytest.raises(cohort.CohortError):
            cohort.context_from_metadata(metadata, secret)


@pytest.mark.parametrize("round_id", ["agent-1", "auto-1"])
@pytest.mark.parametrize("override", [1, True, "1"])
def test_interactive_override_is_forbidden_even_with_full_workload(task_root, round_id, override):
    cohort = load_cohort(task_root)
    with pytest.raises(cohort.CohortError, match="override"):
        cohort.context_from_metadata({"round_id": round_id, "run_id": "trusted-run", "candidate_version": override}, calibration=True)


def test_rejects_bad_shape_row_duplicates_and_cross_row_target_reuse(task_root):
    cohort = load_cohort(task_root)
    cases = synthetic_canonical(cohort)
    with pytest.raises(cohort.CohortError, match="520"):
        cohort.select_cohort(cases[:-1], cohort.EvaluationContext("final"))
    duplicate_case = [dict(case) for case in cases]
    duplicate_case[1]["case_id"] = duplicate_case[0]["case_id"]
    with pytest.raises(cohort.CohortError, match="repeats case_id"):
        cohort.select_cohort(duplicate_case, cohort.EvaluationContext("final"))
    repeated_target = [dict(case) for case in cases]
    target_index = next(
        i
        for i, case in enumerate(repeated_target)
        if case["row_id"] == "table1_gsm8k_5shot_g512"
    )
    repeated_target[target_index]["record_id"] = cases[0]["record_id"]
    with pytest.raises(cohort.CohortError, match="repeat across score rows"):
        cohort.select_cohort(repeated_target, cohort.EvaluationContext("final"))
