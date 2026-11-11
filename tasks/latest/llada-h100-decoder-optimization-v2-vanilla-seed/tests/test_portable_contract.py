"""The portable contract freezes the TaskCore byte-for-byte.

Every digest, path set, row shape and count declared by ``portable/*.json``
and the task JSON is recomputed here from the shipped sources (the committed
stratified workload, the public cases, the frozen policies, the policy
contract, the Judge adapter, the verifier, Dockerfiles and the runner), never
trusted from the declaration.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from fractions import Fraction

import pytest

from conftest import ROOT, load_file, private_data_dir
from sforge.harness.portable_spec import parse_portable_spec, verify_sha256_file

PORTABLE = ROOT / "portable"
# Judge-only data (dataset assets): manifests name the in-repository paths;
# the bytes come from the repository copy or the fetched asset.
PRIVATE_RELATIVE = "images/judge/private_data"
LAB = "images/work/workdir/decoder_lab"
TASK_PATH = ROOT / "tasks/llada_h100_decoder_optimization_v2.json"
SUITE_ID = "llada-decoder-v2"
PROFILE_ID = "single-cuda-sm80-34g"
LEGACY_PROFILE_ID = "dual-cuda-sm80-34g"
METRIC_ID = "primary"
# The public suite runs shorter generation lengths than the Judge: length
# listed in public_cases.json -> length public_eval.py runs.
PUBLIC_GEN_LENGTH = {128: 128, 256: 192, 512: 384, 1024: 768}


@pytest.fixture(scope="module")
def raw_task() -> dict:
    return json.loads(TASK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def portable(raw_task):
    return parse_portable_spec(raw_task["portable"], task_id=raw_task["task_id"])


@pytest.fixture(scope="module")
def contract():
    return load_file("portable_test_policy_contract", ROOT / "scripts/h100_policy_contract.py")


@pytest.fixture(scope="module")
def workload() -> dict:
    return json.loads((ROOT / "workload_policy.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def hidden() -> list[dict]:
    return json.loads(gzip.decompress((private_data_dir() / "hidden_cases.json.gz").read_bytes()))


def _resource_path(relative: str):
    """A manifest resource's bytes: the task file, or the Judge-only asset."""
    if relative.startswith(PRIVATE_RELATIVE + "/") and not (ROOT / relative).is_file():
        return private_data_dir() / relative[len(PRIVATE_RELATIVE) + 1:]
    return ROOT / relative


@pytest.fixture(scope="module")
def public() -> list[dict]:
    return json.loads((ROOT / "images/work/public_tests/public_cases.json").read_text(encoding="utf-8"))


def _manifest(name: str) -> dict:
    return json.loads((PORTABLE / name).read_text(encoding="utf-8"))


def _row_shapes(hidden: list[dict]) -> dict[str, tuple]:
    shapes: dict[str, tuple] = {}
    for case in hidden:
        shape = (case["benchmark"], case["gen_length"], case["block_length"], case["fewshot"])
        assert shapes.setdefault(case["row_id"], shape) == shape
    return shapes


def test_every_declared_digest_is_the_actual_file_digest(raw_task, portable) -> None:
    suite = portable.evaluation_suites[SUITE_ID]
    declarations = {
        "resource-manifest.json": portable.task_core.resource_manifest_sha256,
        "build-resource-manifest.json": raw_task["delivery"]["build_resource_manifest_sha256"],
        "public-suite-manifest.json": suite.roles["public"].manifest_sha256,
        "interactive-suite-manifest.json": suite.roles["interactive"].manifest_sha256,
        "final-suite-manifest.json": suite.roles["final"].manifest_sha256,
        "oracle-protocol.json": suite.oracle.spec_sha256,
        "selection-protocol.json": suite.selection.spec_sha256,
    }
    for profile_id in (PROFILE_ID, LEGACY_PROFILE_ID):
        profile = portable.execution_profiles[profile_id]
        declarations["software-abi.json"] = profile.software_abi.spec_sha256
        declarations["calibration-protocol.json"] = profile.calibration_protocol.spec_sha256
        assert set(declarations) == {path.name for path in PORTABLE.glob("*.json")}
        for filename, declared in declarations.items():
            assert verify_sha256_file(PORTABLE / filename, declared) == declared


def test_resource_manifest_hashes_every_real_scientific_resource() -> None:
    manifest = _manifest("resource-manifest.json")
    expected = {
        "images/work/AGENT_TASK.md", f"{LAB}/README_AGENT.md", f"{LAB}/FINAL_VERSION", f"{LAB}/notes.md",
        *(f"images/work/public_tests/{n}" for n in ("memory_contract.json", "public_cases.json", "public_eval.py", "public_quality.py", "run_public_tests.sh")),
        *(f"{LAB}/runtime/{n}" for n in ("__init__.py", "api.py", "configuration_llada.py", "loader.py", "modeling_llada.py", "source_policy.py")),
        *(f"{LAB}/solutions/decoder_v001/{n}" for n in ("__init__.py", "candidate.py")),
        "images/judge/run_judge.py", "images/judge/protected_hashes.txt",
        *(f"images/judge/verifier/{n}" for n in ("candidate_worker.py", "cohort.py", "nvml_sampler.py", "quality.py", "quality_threshold.py", "row_lifecycle.py", "source_admission.py", "verify_decoder.py")),
        *(f"images/judge/private_data/{n}" for n in ("hidden_cases.json.gz", "memory_policy.json", "quality_controls.json", "reference_quality.json", "workload_policy.json")),
        *(f"{PRIVATE_RELATIVE}/reference_cache/{p.name}" for p in (private_data_dir() / "reference_cache").glob("*.json")),
        "judge/run_judge.py", "workload_policy.json", "memory_policy.json", "quality_controls.json",
        "docs/dataset_manifest.json", "docs/model_manifest.json", "scripts/h100_policy_contract.py",
    }
    assert {resource["path"] for resource in manifest["resources"]} == expected
    for resource in manifest["resources"]:
        verify_sha256_file(_resource_path(resource["path"]), resource["sha256"])
    assert "prepare_submit.sh" not in {resource["path"] for resource in manifest["resources"]}
    protected = (ROOT / "images/judge/protected_hashes.txt").read_text(encoding="utf-8")
    for path in ("runtime/api.py", "runtime/loader.py", "runtime/modeling_llada.py", "runtime/source_policy.py"):
        assert f"/app/workdir/decoder_lab/{path}" in protected
    assert (ROOT / "images/judge/verifier/source_admission.py").read_bytes() == (ROOT / f"{LAB}/runtime/source_policy.py").read_bytes()
    assert (ROOT / "images/judge/run_judge.py").read_bytes() == (ROOT / "judge/run_judge.py").read_bytes()
    for name in ("workload_policy.json", "memory_policy.json", "quality_controls.json"):
        assert (ROOT / name).read_bytes() == (private_data_dir() / name).read_bytes()


def test_build_manifest_covers_exact_portable_delivery_inputs(raw_task) -> None:
    path = PORTABLE / "build-resource-manifest.json"
    declared = raw_task["delivery"]["build_resource_manifest_sha256"]
    assert verify_sha256_file(path, declared) == declared
    manifest = json.loads(path.read_text(encoding="utf-8"))
    declared_paths = {resource["path"] for resource in manifest["resources"]}
    for context_name in ("work", "judge"):
        context = ROOT / f"images/{context_name}"
        actual = {f.relative_to(ROOT).as_posix() for f in context.rglob("*") if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc"}
        if context_name == "judge":
            # The Judge-only private_data/ is staged into its own build context.
            private = private_data_dir()
            actual = {p for p in actual if not p.startswith(PRIVATE_RELATIVE + "/")}
            actual |= {f"{PRIVATE_RELATIVE}/{f.relative_to(private).as_posix()}" for f in private.rglob("*") if f.is_file()}
        assert {r for r in declared_paths if r.startswith(f"images/{context_name}/")} == actual
    extras = {"tasks/BENCHMARK.yaml", "judge/run_judge.py", "scripts/build_images.sh", "scripts/select_reference_profile.py", "scripts/sforge_h100_entry.py",
              "scripts/restricted_connect_proxy.py", "scripts/lib.sh", "scripts/validate_layout.py", "portable/REBUILD_PROMPT.md"}
    assert {p for p in declared_paths if not p.startswith("images/")} == extras
    assert "images/work/public_tests/prepare_submit.sh" in declared_paths
    assert not any(p.startswith("host_reference/") for p in declared_paths)
    assert "memory_qualification_fixtures.json" not in declared_paths
    for resource in manifest["resources"]:
        verify_sha256_file(_resource_path(resource["path"]), resource["sha256"])


def test_host_private_final_manifest_cannot_enter_the_work_image() -> None:
    work = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8").casefold()
    assert "portable" not in work and "final-suite-manifest" not in work and "private_data" not in work and "host_reference" not in work
    judge = (ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    # private_data/ enters only the host-built holdout layer FROM the public Judge.
    assert "COPY private_data/" not in judge
    holdout_layer = (ROOT / "images/judge/Dockerfile.private").read_text(encoding="utf-8")
    assert "FROM ${JUDGE_PUBLIC_IMAGE}" in holdout_layer
    assert "COPY --from=private_data . /opt/decoder_judge/private_data/" in holdout_layer
    assert judge.startswith("ARG WORK_IMAGE=")


def test_public_manifest_is_the_disclosed_public_cases(portable, public) -> None:
    manifest = _manifest("public-suite-manifest.json")
    cases = manifest["cases"]
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["public"].case_count == len(public) == 44
    by_id = {case["case_id"]: case for case in public}
    assert [case["case_id"] for case in cases] == [case["case_id"] for case in public]
    row_counts = Counter(case["row_id"] for case in public)
    for row in cases:
        source = by_id[row["case_id"]]
        for key in ("row_id", "benchmark", "block_length", "fewshot", "prompt_token_count", "public_profile", "record_id", "source_split"):
            assert row[key] == source[key]
        assert row["gen_length"] == PUBLIC_GEN_LENGTH[source["gen_length"]]
        assert row["role"] == "performance" and row["prompt_visibility"] == "disclosed"
        assert Fraction(row["weight"]["numerator"], row["weight"]["denominator"]) == Fraction(1, row_counts[row["row_id"]] * len(row_counts))
        assert "request_text" not in row and "gold" not in row
    assert sum(Fraction(r["weight"]["numerator"], r["weight"]["denominator"]) for r in cases) == 1
    assert manifest["row_counts"] == dict(sorted(row_counts.items()))
    assert manifest["profiles"] == {"quick": 4, "full": 40}
    assert manifest["distinct_targets"] == len({(c["benchmark"], c["record_id"]) for c in public}) == 40
    assert manifest["case_counts"] == {"total": 44, "scored_performance": 44, "zero_weight_hard_gate": 0}
    evaluator = (ROOT / "images/work/public_tests/public_eval.py").read_text(encoding="utf-8")
    assert 'PROFILE_COUNTS = {"quick": 4, "full": 40}' in evaluator
    assert f"PUBLIC_GEN_LENGTH = {PUBLIC_GEN_LENGTH}" in evaluator


def _assert_rows(rows: list[dict], kind: str, contract, shapes: dict[str, tuple]) -> None:
    assert [row["row_id"] for row in rows] == list(contract.ROW_ORDER)
    for row in rows:
        benchmark, gen_length, block_length, fewshot = shapes[row["row_id"]]
        assert row["case_id"] == f"{kind}-{row['row_id']}"
        assert (row["benchmark"], row["gen_length"], row["block_length"], row["fewshot"]) == (benchmark, gen_length, block_length, fewshot)
        assert row["benchmark"] == contract.ROW_BENCHMARKS[row["row_id"]]
        assert row["role"] == "performance" and row["prompt_visibility"] == "hidden_committed"
        assert Fraction(row["weight"]["numerator"], row["weight"]["denominator"]) == Fraction(1, 12)


def test_interactive_manifest_is_the_identical_complete_final_workload(portable, contract, workload, hidden) -> None:
    manifest = _manifest("interactive-suite-manifest.json")
    final = _manifest("final-suite-manifest.json")
    assert len(manifest["cases"]) == portable.evaluation_suites[SUITE_ID].roles["interactive"].case_count == 12
    assert manifest["cases"] == final["cases"]
    assert manifest["suite_id"] == final["suite_id"] == "llada-decoder-final-v2"
    assert manifest["total_generations"] == len(hidden) == contract.INTERACTIVE_TOTAL_CASES == 520
    assert manifest["total_requested_slots"] == final["total_requested_slots"] == 216_320
    assert manifest["hidden_cases_sha256"] == contract.canonical_hidden_sha256(ROOT)
    assert manifest["suite_gate_timeouts"] == final["suite_gate_timeouts"]
    assert manifest["row_atomic"] == final["row_atomic"]
    assert manifest["quality_role"] == workload["quality_policy"]["interactive"] == "same_as_final"
    cohort = (ROOT / "images/judge/verifier/cohort.py").read_text()
    assert 'INTERACTIVE_SELECTION = "all_frozen_final_v1"' in cohort


def test_final_manifest_is_the_committed_stratified_workload(raw_task, portable, contract, workload, hidden) -> None:
    manifest = _manifest("final-suite-manifest.json")
    assert len(manifest["cases"]) == portable.evaluation_suites[SUITE_ID].roles["final"].case_count == 12
    shapes = _row_shapes(hidden)
    _assert_rows(manifest["cases"], "final", contract, shapes)
    counts = Counter(case["row_id"] for case in hidden)
    assert manifest["generations"] == dict(contract.ROW_COUNTS) == counts
    for row in manifest["cases"]:
        assert row["generations"] == counts[row["row_id"]]
        assert row["requested_slots"] == counts[row["row_id"]] * shapes[row["row_id"]][1]
    assert manifest["total_generations"] == len(hidden) == contract.GENERATION_COUNT == 520
    assert manifest["total_requested_slots"] == sum(r["requested_slots"] for r in manifest["cases"]) == contract.REQUESTED_SLOTS == 216_320
    assert manifest["hidden_cases_sha256"] == contract.canonical_hidden_sha256(ROOT)
    assert manifest["dataset_manifest_sha256"] == workload["manifest"]["dataset_manifest_sha256"]
    assert manifest["context_exclusions"] == workload["manifest"]["context_exclusions"]
    assert manifest["category_counts"] == dict(contract.CATEGORY_COUNTS) == Counter(case["benchmark"] for case in hidden)
    final = workload["final"]
    assert manifest["suite_gate_timeouts"] == {
        "case_seconds": final["case_timeout_seconds"], "row_seconds": final["row_timeout_seconds"],
        "inner_seconds": final["inner_timeout_seconds"], "outer_seconds": final["outer_timeout_seconds"],
        "wrapper_margin_seconds": final["wrapper_margin_seconds"], "max_infrastructure_retries_per_row": final["max_infrastructure_retries_per_row"],
    }
    if workload["qualification_status"] == "QUALIFIED":
        assert 0 < final["case_timeout_seconds"] < final["row_timeout_seconds"] < final["inner_timeout_seconds"] < final["outer_timeout_seconds"] <= raw_task["judge"]["eval_timeout"]
    else:
        assert all(final[key] is None for key in ("case_timeout_seconds", "row_timeout_seconds", "inner_timeout_seconds", "outer_timeout_seconds"))
    assert manifest["orchestration_timeouts"]["standalone_final_evaluation_seconds"] == raw_task["judge"]["eval_timeout"] == 7200
    reference = json.loads((private_data_dir() / "reference_quality.json").read_text(encoding="utf-8"))
    assert manifest["quality_reference"]["status"] == reference["status"] and manifest["quality_reference"]["calibrated"] == reference["calibrated"]
    runner = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    assert 'JUDGE_PORT="${JUDGE_PORT:-18384}"' in runner
    assert '--timeout "${TASK_EVAL_TIMEOUT}"' in runner and "TASK_EVAL_TIMEOUT=\"$(jq -er '.judge.eval_timeout'" in runner
    selection = _manifest("selection-protocol.json")
    assert selection["rows"] == list(contract.ROW_ORDER)
    assert selection["row_shapes"] == {r: {"benchmark": s[0], "gen_length": s[1], "block_length": s[2], "fewshot": s[3]} for r, s in shapes.items()}
    assert selection["final"]["generations"] == 520 and selection["final"]["scored_case_count"] == 12 and selection["final"]["hard_gate_case_count"] == 0
    assert selection["interactive"]["sampled_cases_per_round"] == 520


def test_oracle_matches_the_frozen_policies_and_judge_gates(contract, workload) -> None:
    oracle = _manifest("oracle-protocol.json")
    adapter = load_file("portable_test_run_judge", ROOT / "judge/run_judge.py")
    formal = (ROOT / "scripts/validate_formal_artifacts.py").read_text(encoding="utf-8")
    for gate in oracle["gates"]["names"]:
        assert f'"{gate}"' in formal
    assert oracle["gates"]["names"][:3] == ["source_ok", "integrity_ok", "performance_ok"]
    assert set(oracle["gates"]["names"]) == {"source_ok", "integrity_ok", "performance_ok", "memory_leak_free", "quality_ok", "quality_hard_gate"}
    assert oracle["policy"]["row_order"] == list(contract.ROW_ORDER)
    assert oracle["policy"]["row_counts"] == dict(contract.ROW_COUNTS)
    memory_contract = json.loads((ROOT / "images/work/public_tests/memory_contract.json").read_text(encoding="utf-8"))
    memory = json.loads((ROOT / "memory_policy.json").read_text(encoding="utf-8"))
    reserve = None if memory["headroom_mib"] is None else memory["headroom_mib"] + memory["safety_margin_mib"]
    assert oracle["policy"]["memory_reserve_mib"] == memory_contract["memory_reserve_mib"] == reserve
    assert oracle["policy"]["row_sentinel_metric"] == contract.ROW_SENTINEL_METRIC == memory["row_sentinel"]["metric"]
    assert oracle["policy"]["quality_policy_status"] == workload["quality_policy"]["status"]
    verifier = (ROOT / "images/judge/verifier/verify_decoder.py").read_text(encoding="utf-8")
    assert "producer_relative_min_retention_ppm_v1" in verifier
    assert oracle["policy"]["quality_threshold_formula"] == "producer_relative_min_retention_ppm_v1"
    assert "quality_ok" in adapter.__dict__.get("__doc__", "") or "quality" in (ROOT / "judge/run_judge.py").read_text(encoding="utf-8")
    calibration = _manifest("calibration-protocol.json")
    assert calibration["comparability"] == "same_measurement_profile"
    assert calibration["measurement_profile_key"][:3] == ["gpu_name", "compute_capability", "driver_version"]
    # Driver and image ids are recorded provenance; selection/refusal is GPU
    # model + compute capability plus the pinned software ABI.
    assert calibration["reference_profile_key"] == ["gpu_name", "compute_capability"]
    assert calibration["reference_software_abi"] == ["cuda_runtime", "torch", "transformers"]
    selector = (ROOT / "scripts/select_reference_profile.py").read_text(encoding="utf-8")
    assert 'REFERENCE_PROFILE_FIELDS = ("compute_capability", "gpu_name")' in selector
    assert 'REFERENCE_PROFILE_FIELDS = ("compute_capability", "gpu_name")' in verifier


def test_software_abi_matches_the_pinned_images_and_adapter() -> None:
    abi = _manifest("software-abi.json")
    work = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    assert work.startswith("FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    for pin in ("accelerate==1.4.0", "huggingface-hub==0.29.3", "safetensors==0.5.3", "tokenizers==0.21.0", "transformers==4.49.0"):
        assert pin in work
    assert 'ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0"' in work
    assert abi["torch_cuda_arch_list"] == "8.0;8.6;8.9;9.0;10.0"
    assert abi["source_runtime"] == ["pytorch-2.7.1", "cudnn-9", "transformers-4.49.0", "tokenizers-0.21.0", "safetensors-0.5.3", "accelerate-1.4.0", "huggingface-hub-0.29.3"]
    adapter = (ROOT / "scripts/sforge_h100_entry.py").read_text(encoding="utf-8")
    assert 'MODEL_MOUNT = "/opt/models/llada-hf"' in adapter and 'MODEL_REVISION = "6059b30c9531767ce47e0ca6a8161a6bb0c67693"' in adapter
    assert abi["model_binding"]["revision"] == "6059b30c9531767ce47e0ca6a8161a6bb0c67693" and abi["model_binding"]["adapter"] == "scripts/sforge_h100_entry.py"
    prompt = (ROOT / "images/work/AGENT_TASK.md").read_text(encoding="utf-8")
    for root in abi["allowed_import_roots"]:
        assert f"`{root}`" in prompt
    admission = load_file("portable_test_admission", ROOT / "images/judge/verifier/source_admission.py")
    assert set(abi["allowed_import_roots"]) == set(admission.ALLOWED_IMPORT_ROOTS)


def test_portable_layers_preserve_metric_and_use_capabilities(raw_task, portable, contract, hidden) -> None:
    metric = portable.metrics[METRIC_ID]
    assert metric.direction == "maximize" == raw_task["judge"]["score_direction"]
    assert metric.unit == "tokens_per_s" and metric.comparability == "same_measurement_profile" and metric.baseline_policy == "none"
    assert list(metric.aggregation.spec["rows"]) == list(contract.ROW_ORDER) and metric.aggregation.spec["row_count"] == 12
    contract_doc_path = ROOT / "docs/SCORING_CONTRACT.md"
    if contract_doc_path.is_file():
        assert "score = exp(mean(log(TP_r)))" in contract_doc_path.read_text(encoding="utf-8")
    single = portable.execution_profiles[PROFILE_ID]
    legacy = portable.execution_profiles[LEGACY_PROFILE_ID]
    assert single.vendor == legacy.vendor == "nvidia" and single.count == 1 and legacy.count == 2
    for profile in (single, legacy):
        assert profile.min_memory_bytes == contract.MIN_DEVICE_MEMORY_BYTES == 34816 * 1024 * 1024
        assert profile.min_compute_capability == contract.MIN_COMPUTE_CAPABILITY == "8.0"
        assert "nvml" in profile.features and "bf16" in profile.features
        assert profile.software_abi.spec_sha256 == single.software_abi.spec_sha256
    layout = load_file("portable_test_layout", ROOT / "scripts/validate_layout.py")
    assert layout.MIN_DEVICE_MEMORY_MIB * 1024 * 1024 == single.min_memory_bytes and layout.MIN_COMPUTE_CAPABILITY == single.min_compute_capability
    definition = portable.task_core.definition
    assert list(definition["rows"]) == list(contract.ROW_ORDER)
    assert dict(definition["row_counts"]) == dict(contract.ROW_COUNTS) == Counter(case["row_id"] for case in hidden)
    assert definition["generation_count"] == 520 and definition["requested_slots"] == 216_320
    assert definition["workload_profile"] == contract.PROFILE
    assert portable.defaults.evaluation_suite_id == SUITE_ID and portable.defaults.execution_profile_id == PROFILE_ID and portable.defaults.metric_id == METRIC_ID
    portable_text = json.dumps(raw_task["portable"], sort_keys=True)
    assert "h100" not in portable_text.casefold() and "gpu-" not in portable_text.casefold()
    assert raw_task["cwd"] == "/app/workdir/decoder_lab"
    adapter = load_file("portable_test_adapter", ROOT / "scripts/sforge_h100_entry.py")
    assert adapter.SINGLE_EXECUTION_PROFILE_ID == PROFILE_ID and adapter.EXECUTION_PROFILE_ID == LEGACY_PROFILE_ID
    assert set(adapter.EXECUTION_PROFILE_IDS) == set(portable.execution_profiles)


def test_refresh_tool_reports_no_drift() -> None:
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/refresh_portable_manifests.py"), "--check"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
