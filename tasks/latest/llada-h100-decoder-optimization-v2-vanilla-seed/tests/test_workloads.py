from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json

from conftest import private_data_dir


EXPECTED = {
    "table1_gsm8k_5shot_g256": 39,
    "table1_gsm8k_5shot_g512": 39,
    "table1_math_4shot_g256": 65,
    "table1_math_4shot_g512": 65,
    "table1_humaneval_0shot_g256": 65,
    "table1_humaneval_0shot_g512": 65,
    "table1_mbpp_3shot_g256": 65,
    "table1_mbpp_3shot_g512": 65,
    "table4_gsm8k_5shot_g1024": 13,
    "table5_gsm8k_8shot_g256": 13,
    "table5_gsm8k_8shot_g512": 13,
    "table5_gsm8k_8shot_g1024": 13,
}
PUBLIC_FULL_EXPECTED = {
    "table1_gsm8k_5shot_g256": 3,
    "table1_gsm8k_5shot_g512": 3,
    "table1_math_4shot_g256": 5,
    "table1_math_4shot_g512": 5,
    "table1_humaneval_0shot_g256": 5,
    "table1_humaneval_0shot_g512": 5,
    "table1_mbpp_3shot_g256": 5,
    "table1_mbpp_3shot_g512": 5,
    "table4_gsm8k_5shot_g1024": 1,
    "table5_gsm8k_8shot_g256": 1,
    "table5_gsm8k_8shot_g512": 1,
    "table5_gsm8k_8shot_g1024": 1,
}
EXPECTED_ROW_METADATA = {
    "table1_gsm8k_5shot_g256": ("gsm8k", 5, 256, 32),
    "table1_gsm8k_5shot_g512": ("gsm8k", 5, 512, 32),
    "table1_math_4shot_g256": ("math", 4, 256, 32),
    "table1_math_4shot_g512": ("math", 4, 512, 32),
    "table1_humaneval_0shot_g256": ("humaneval", 0, 256, 32),
    "table1_humaneval_0shot_g512": ("humaneval", 0, 512, 32),
    "table1_mbpp_3shot_g256": ("mbpp", 3, 256, 32),
    "table1_mbpp_3shot_g512": ("mbpp", 3, 512, 32),
    "table4_gsm8k_5shot_g1024": ("gsm8k", 5, 1024, 32),
    "table5_gsm8k_8shot_g256": ("gsm8k", 8, 256, 32),
    "table5_gsm8k_8shot_g512": ("gsm8k", 8, 512, 32),
    "table5_gsm8k_8shot_g1024": ("gsm8k", 8, 1024, 32),
}
EXPECTED_UNIQUE = {"gsm8k": 130, "math": 130, "humaneval": 130, "mbpp": 130}
EXPECTED_ELIGIBLE = {"gsm8k": 1319, "math": 5000, "humaneval": 164, "mbpp": 499}


def test_stratified_rows_are_disjoint_and_public_profiles_are_balanced(task_root):
    public_path = task_root / "images/work/public_tests/public_cases.json"
    hidden_path = private_data_dir() / "hidden_cases.json.gz"
    assert hidden_path.read_bytes()[9] == 3
    public = json.loads(public_path.read_text())
    hidden_bytes = gzip.decompress(hidden_path.read_bytes())
    hidden = json.loads(hidden_bytes)
    assert len(public) == 44
    assert len(hidden) == 520
    assert Counter(case["row_id"] for case in hidden) == EXPECTED
    assert sum(case["gen_length"] for case in hidden) == 216_320
    assert len({case["case_id"] for case in hidden}) == 520

    row_records = defaultdict(set)
    row_benchmarks = {}
    for case in hidden:
        row_id = case["row_id"]
        row_benchmarks[row_id] = case["benchmark"]
        assert case["record_id"] not in row_records[row_id]
        row_records[row_id].add(case["record_id"])
        assert (
            case["benchmark"], case["fewshot"], case["gen_length"], case["block_length"]
        ) == EXPECTED_ROW_METADATA[row_id]
        assert case["case_id"] == f"{row_id}:{case['record_id']}"
    for benchmark, count in EXPECTED_UNIQUE.items():
        benchmark_sets = [
            row_records[row_id]
            for row_id in EXPECTED
            if row_benchmarks[row_id] == benchmark
        ]
        flattened = [record for records in benchmark_sets for record in records]
        assert len(flattened) == count
        assert len(set(flattened)) == count

    assert Counter(case["public_profile"] for case in public) == {"quick": 4, "full": 40}
    public_full = [case for case in public if case["public_profile"] == "full"]
    public_quick = [case for case in public if case["public_profile"] == "quick"]
    assert Counter(case["row_id"] for case in public_full) == PUBLIC_FULL_EXPECTED
    assert Counter(case["benchmark"] for case in public_full) == {
        benchmark: 10 for benchmark in EXPECTED_UNIQUE
    }
    assert Counter(case["benchmark"] for case in public_quick) == {
        benchmark: 1 for benchmark in EXPECTED_UNIQUE
    }
    assert all(case["gen_length"] == 128 for case in public_quick)
    assert len({(case["benchmark"], case["record_id"]) for case in public}) == 40

    public_targets = {(case["benchmark"], case["record_id"]) for case in public}
    hidden_targets = {(case["benchmark"], case["record_id"]) for case in hidden}
    assert public_targets.isdisjoint(hidden_targets)
    assert all(case["source_split"] in {"train", "dev", "synthetic"} for case in public)
    assert all(case["source_split"] != "test" for case in public)


def test_dataset_manifest_exact_profile_and_hashes(task_root):
    manifest_path = task_root / "docs/dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 5
    assert manifest["task_id"] == "llada_h100_decoder_optimization_v2"
    assert manifest["workload_profile"] == "fast_dllm_v1_llada_12row_stratified520_batch1"
    assert manifest["public_cases"] == 44
    assert manifest["public_unique_targets"] == 40
    assert manifest["public_profile_counts"] == {"quick": 4, "full": 40}
    assert manifest["public_target_counts"] == {
        benchmark: 10 for benchmark in EXPECTED_UNIQUE
    }
    assert manifest["public_full_row_counts"] == PUBLIC_FULL_EXPECTED
    assert manifest["hidden_cases"] == 520
    assert manifest["generation_count"] == 520
    assert manifest["requested_slots"] == 216_320
    assert manifest["hidden_target_counts"] == EXPECTED_UNIQUE
    assert manifest["hidden_row_counts"] == EXPECTED
    assert manifest["standard_test_data"]["implicit_limit"] == 130
    assert manifest["standard_test_data"]["mounted_in_work"] is False
    assert manifest["standard_test_data"]["classification"] == "public_reproducible_judge_only"
    assert manifest["public_fixture_policy"]["official_test_records"] == 0
    assert manifest["public_fixture_policy"]["target_disjoint_from_hidden"] is True
    assert manifest["public_fixture_policy"]["target_disjoint_from_fewshot"] is True
    assert manifest["generator_sha256"] == hashlib.sha256(
        (task_root / "scripts/prepare_benchmark_data.py").read_bytes()
    ).hexdigest()
    for key, relative in (
        ("public_sha256", "images/work/public_tests/public_cases.json"),
        ("hidden_sha256", None),
    ):
        payload = (
            gzip.decompress(
                (private_data_dir() / "hidden_cases.json.gz").read_bytes()
            )
            if relative is None
            else (task_root / relative).read_bytes()
        )
        assert hashlib.sha256(payload).hexdigest() == manifest[key]
    assert manifest["hidden_storage"]["compressed_sha256"] == hashlib.sha256(
        (private_data_dir() / "hidden_cases.json.gz").read_bytes()
    ).hexdigest()


def test_context_exclusion_is_exactly_mbpp_482(task_root):
    manifest = json.loads((task_root / "docs/dataset_manifest.json").read_text())
    hidden = json.loads(
        gzip.decompress(
            (private_data_dir() / "hidden_cases.json.gz").read_bytes()
        )
    )
    context = manifest["context_validation"]
    assert context["validated_hidden_cases"] == 520
    assert context["candidate_pool_after_filter"] == EXPECTED_ELIGIBLE
    excluded = {
        (entry["benchmark"], entry["record_id"])
        for entries in context["excluded_candidates"].values()
        for entry in entries
    }
    assert excluded == {("mbpp", "test:482/task:493")}
    totals = []
    for case in hidden:
        prompt_tokens = case["prompt_token_count"]
        assert isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool)
        total = prompt_tokens + case["gen_length"]
        assert total <= 4096
        totals.append(total)
    assert context["maximum_hidden_total_tokens"] == max(totals)


def test_workload_policy_is_byte_identical_and_manifest_bound(task_root):
    root_policy = task_root / "workload_policy.json"
    private_policy = private_data_dir() / "workload_policy.json"
    assert root_policy.read_bytes() == private_policy.read_bytes()
    policy = json.loads(root_policy.read_text())
    assert policy["manifest"]["row_counts"] == EXPECTED
    assert policy["manifest"]["generation_count"] == 520
    assert policy["manifest"]["requested_slots"] == 216_320
    assert policy["manifest"]["context_exclusions"] == ["mbpp:test:482/task:493"]
    assert policy["manifest"]["hidden_cases_sha256"] == hashlib.sha256(
        gzip.decompress(
            (private_data_dir() / "hidden_cases.json.gz").read_bytes()
        )
    ).hexdigest()
    assert policy["manifest"]["dataset_manifest_sha256"] == hashlib.sha256(
        (task_root / "docs/dataset_manifest.json").read_bytes()
    ).hexdigest()


def test_public_targets_are_disjoint_from_fewshot_sources(task_root):
    public = json.loads(
        (task_root / "images/work/public_tests/public_cases.json").read_text()
    )
    targets = {
        benchmark: {
            case["record_id"] for case in public if case["benchmark"] == benchmark
        }
        for benchmark in EXPECTED_UNIQUE
    }
    assert targets["gsm8k"] == {f"train:{index}" for index in range(8, 18)}
    assert targets["gsm8k"].isdisjoint({f"train:{index}" for index in range(8)})
    assert all("/train:" in target and int(target.rsplit(":", 1)[1]) >= 4 for target in targets["math"])
    assert all(target.startswith("synthetic:mlsysforge_") for target in targets["humaneval"])
    assert all(target.startswith("train:") for target in targets["mbpp"])


def test_public_mbpp_runs_correctness_grading(task_root):
    source = (task_root / "images/work/public_tests/public_eval.py").read_text()
    assert 'quality = {"evaluated": True, **score_completion(case, completion)}' in source
    assert "performance_only" not in source
    assert '"dataset_summary": summarize_datasets(rows)' in source
