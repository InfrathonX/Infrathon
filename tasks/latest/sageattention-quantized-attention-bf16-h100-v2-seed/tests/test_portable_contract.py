"""The portable contract freezes the TaskCore byte-for-byte.

Every digest, path set and case count declared by ``portable/*.json`` and the
task JSON is recomputed here from the executable sources (runtime edge
fixtures and cells, public runner, Judge verifier, final-holdout generator,
Dockerfiles, runner), never trusted from the declaration.
"""

from __future__ import annotations

import json
import random
import re
from fractions import Fraction

import pytest

import hashlib

from conftest import TASK_ROOT, judge_only_file
from sforge.harness.portable_spec import parse_portable_spec, verify_sha256_file

ROOT = TASK_ROOT
PORTABLE = ROOT / "portable"
TASK_PATH = ROOT / "tasks/sageattention_quantized_attention_bf16_h100_seed_v2.json"
SUITE_ID = "sageattention-qattn-v2"
PROFILE_ID = "single-cuda-sm80-64g"
METRIC_ID = "family-geomean-effective-tflops"
LAB = "quant_attention_lab"
ENV_PREFIX = "SAGE_QATTN_"
LEN = "l"
EDGE_COUNT = 8
SCORED = 774
REAL_SCORED = 139
TOTAL = EDGE_COUNT + SCORED
MIN_CC = (8, 0)


def _assert_recipe_bound_fixture(document: dict, runtime) -> None:
    """The e8 capture is identified by its recipe, never by a pinned SHA-256.

    Every deployment captures its own copy at build time, so no declaration may
    carry the digest of one host's bytes.
    """
    entries = document["external_fixtures"]
    assert len(entries) == 11
    entry = entries[0]
    _assert_real_layer_fixtures(entries[1:], runtime)
    assert entry["name"] == runtime.FIXTURE_NAME
    assert entry["role"] == "edge_fixture_input"
    assert entry["identity"] == "per-deployment-capture"
    assert entry["recipe"] == "fixtures/recipe/RECIPE.json"
    assert "sha256" not in entry
    verify_sha256_file(ROOT / entry["recipe"], entry["recipe_sha256"])
    assert entry["source"].startswith("fixtures/recipe/build_fixture.sh")


def _assert_real_layer_fixtures(entries: list, runtime) -> None:
    """The scored real-layer captures are recipe-bound exactly like e8."""
    expected = {f"real_layers/{cdir}": f"fixtures/recipe/real_layers/{name}" for name, _rid, cdir, _g in runtime.REAL_SOURCES.values()}
    assert {entry["name"] for entry in entries} == set(expected)
    for entry in entries:
        recipe_path = expected[entry["name"]]
        recipe = json.loads((ROOT / recipe_path).read_text(encoding="utf-8"))
        assert entry["role"] == "scored_real_layer_input"
        assert entry["identity"] == "per-deployment-capture"
        assert entry["recipe"] == recipe_path and entry["layers"] == len(runtime.captured_layers(recipe))
        assert "sha256" not in entry
        verify_sha256_file(ROOT / recipe_path, entry["recipe_sha256"])
        assert entry["source"].startswith("fixtures/recipe/real_layers/capture_scored_layers.sh")


def load_module(name: str, relative: str):
    """Load a task source file as a registered module (dataclasses need sys.modules)."""
    import importlib.util
    import sys

    path = TASK_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    parent = str(path.parent)
    sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(parent)
    return module


@pytest.fixture(scope="module")
def raw_task() -> dict:
    return json.loads(TASK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def portable(raw_task):
    return parse_portable_spec(raw_task["portable"], task_id=raw_task["task_id"])


@pytest.fixture(scope="module")
def runtime():
    return load_module("portable_test_runtime", "runtime/quant_attention_runtime.py")


@pytest.fixture(scope="module")
def verifier(runtime):
    return load_module("portable_test_verifier", "images/judge/verify_quant_attention.py")


@pytest.fixture(scope="module")
def public_runner():
    return load_module("portable_test_public_runner", "images/work/benchmark/public_runner.py")


@pytest.fixture(scope="module")
def generator():
    return load_module("portable_test_generator", "scripts/generate_final_manifest.py")


def _manifest(name: str) -> dict:
    return json.loads((PORTABLE / name).read_text(encoding="utf-8"))


def _judge_env() -> dict[str, int]:
    dockerfile = (ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    return {
        name: int(value)
        for name, value in re.findall(rf"({ENV_PREFIX}[A-Z_]+)=([0-9]+)", dockerfile)
    }


def _plain(value):
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def test_every_declared_digest_is_the_actual_file_digest(raw_task, portable) -> None:
    suite = portable.evaluation_suites[SUITE_ID]
    profile = portable.execution_profiles[PROFILE_ID]
    declarations = {
        "resource-manifest.json": portable.task_core.resource_manifest_sha256,
        "build-resource-manifest.json": raw_task["delivery"]["build_resource_manifest_sha256"],
        "public-suite-manifest.json": suite.roles["public"].manifest_sha256,
        "interactive-suite-manifest.json": suite.roles["interactive"].manifest_sha256,
        "final-suite-manifest.json": suite.roles["final"].manifest_sha256,
        "oracle-protocol.json": suite.oracle.spec_sha256,
        "selection-protocol.json": suite.selection.spec_sha256,
        "software-abi.json": profile.software_abi.spec_sha256,
        "calibration-protocol.json": profile.calibration_protocol.spec_sha256,
    }
    assert set(declarations) == {path.name for path in PORTABLE.glob("*.json")}
    for filename, declared in declarations.items():
        assert verify_sha256_file(PORTABLE / filename, declared) == declared


def test_resource_manifest_hashes_every_real_scientific_resource(runtime) -> None:
    manifest = _manifest("resource-manifest.json")
    assert manifest["schema_version"] == 1
    expected_paths = {
        "images/work/AGENT_TASK.md",
        f"images/work/workdir/{LAB}/README_AGENT.md",
        "images/work/benchmark/quant_attention_runtime.py",
        "images/work/benchmark/public_runner.py",
        "images/work/benchmark/run_public_tests.sh",
        "images/work/benchmark/source_admission.py",
        f"images/work/workdir/{LAB}/tools/run_one.sh",
        f"images/work/workdir/{LAB}/tools/run_public_tests.sh",
        "images/judge/candidate_worker.py",
        "images/judge/run_judge.py",
        "images/judge/source_admission.py",
        "images/judge/verify_quant_attention.py",
        "images/judge/protected_hashes.txt",
        "judge/run_judge.py",
        "runtime/quant_attention_runtime.py",
        "scripts/generate_final_manifest.py",
        "fixtures/recipe/RECIPE.json",
        "fixtures/recipe/capture_e8_wan_qkv.py",
        "fixtures/recipe/real_layers/RECIPE_cogvideox2b.json",
        "fixtures/recipe/real_layers/RECIPE_llama31_8b.json",
        "fixtures/recipe/real_layers/RECIPE_cogvideox2b_s10.json",
        "fixtures/recipe/real_layers/RECIPE_cogvideox2b_s40.json",
        "fixtures/recipe/real_layers/RECIPE_llama31_128k.json",
        "fixtures/recipe/real_layers/RECIPE_llama31_64k.json",
        "fixtures/recipe/real_layers/RECIPE_llama3_262k.json",
        "fixtures/recipe/real_layers/RECIPE_sd35m_1024.json",
        "fixtures/recipe/real_layers/RECIPE_wan13b_480p.json",
        "fixtures/recipe/real_layers/RECIPE_wan14b_720p.json",
        "fixtures/recipe/real_layers/capture_real_layers.py",
        "fixtures/recipe/public_layers/capture_public_layers.py",
    }
    assert {resource["path"] for resource in manifest["resources"]} == expected_paths
    for resource in manifest["resources"]:
        verify_sha256_file(ROOT / resource["path"], resource["sha256"])
    # The edge capture is bound by its recipe, never by one host's bytes.
    _assert_recipe_bound_fixture(manifest, runtime)
    layout = (ROOT / "scripts/validate_layout.py").read_text(encoding="utf-8")
    assert "FIXTURE_SHA256" not in layout


def test_build_manifest_covers_exact_portable_delivery_inputs(raw_task) -> None:
    path = PORTABLE / "build-resource-manifest.json"
    declared = raw_task["delivery"]["build_resource_manifest_sha256"]
    assert verify_sha256_file(path, declared) == declared
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["boundary"]["kind"] == "portable_image_delivery_inputs"
    declared_paths = {resource["path"] for resource in manifest["resources"]}
    for context_name in ("work", "judge"):
        context = ROOT / f"images/{context_name}"
        actual_context_files = {
            file.relative_to(ROOT).as_posix()
            for file in context.rglob("*")
            if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc"
        }
        assert {
            resource for resource in declared_paths
            if resource.startswith(f"images/{context_name}/")
        } == actual_context_files
    extras = {
        "tasks/BENCHMARK.yaml",
        "judge/run_judge.py",
        "runtime/quant_attention_runtime.py",
        "scripts/build_images.sh",
        "scripts/generate_final_manifest.py",
        "fixtures/recipe/RECIPE.json",
        "fixtures/recipe/capture_e8_wan_qkv.py",
        "fixtures/recipe/build_fixture.sh",
        "fixtures/recipe/real_layers/RECIPE_cogvideox2b.json",
        "fixtures/recipe/real_layers/RECIPE_llama31_8b.json",
        "fixtures/recipe/real_layers/RECIPE_cogvideox2b_s10.json",
        "fixtures/recipe/real_layers/RECIPE_cogvideox2b_s40.json",
        "fixtures/recipe/real_layers/RECIPE_llama31_128k.json",
        "fixtures/recipe/real_layers/RECIPE_llama31_64k.json",
        "fixtures/recipe/real_layers/RECIPE_llama3_262k.json",
        "fixtures/recipe/real_layers/RECIPE_sd35m_1024.json",
        "fixtures/recipe/real_layers/RECIPE_wan13b_480p.json",
        "fixtures/recipe/real_layers/RECIPE_wan14b_720p.json",
        "fixtures/recipe/real_layers/capture_scored_layers.sh",
        "fixtures/recipe/real_layers/capture_in_venv.sh",
        "fixtures/recipe/real_layers/capture_real_layers.py",
        "fixtures/recipe/real_layers/Dockerfile.builder",
        "fixtures/recipe/real_layers/build_real_layers.sh",
        "fixtures/recipe/public_layers/capture_public_layers.py",
        "fixtures/recipe/public_layers/build_public_layers.sh",
        "scripts/lib.sh",
        "scripts/validate_layout.py",
        "portable/REBUILD_PROMPT.md",
    }
    assert {p for p in declared_paths if not p.startswith("images/")} == extras
    assert "images/work/benchmark/prepare_submit.sh" in declared_paths
    assert not any(p.startswith("oracle/") for p in declared_paths)
    # The recipe is a delivery input; the capture it produces never is.
    assert {p for p in declared_paths if p.startswith("fixtures/")} == {
        p for p in declared_paths if p.startswith("fixtures/recipe/")
    }
    for resource in manifest["resources"]:
        verify_sha256_file(ROOT / resource["path"], resource["sha256"])
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert build.count('--build-context "fixtures=${fixtures_context}"') == 1


def test_host_private_final_manifest_cannot_enter_the_work_image() -> None:
    work_context = ROOT / "images/work"
    assert not (PORTABLE / "final-suite-manifest.json").is_relative_to(work_context)
    dockerfile = (work_context / "Dockerfile").read_text(encoding="utf-8").casefold()
    assert "portable" not in dockerfile
    assert "final-suite-manifest" not in dockerfile
    assert "final_cases" not in dockerfile
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert build.count('--build-context "final_cases=${final_cases_context}"') == 1


def _assert_edge_rows_match_runtime(cases: list[dict], runtime) -> None:
    rows = [case for case in cases if case["role"] == "correctness"]
    assert [row["case_id"] for row in rows] == list(runtime.EDGE_CASE_IDS)
    assert len(rows) == EDGE_COUNT
    for row, actual in zip(rows, runtime.EDGE_CASES):
        for field, value in actual.items():
            assert row[field] == _plain(value), (row["case_id"], field)
        assert row["measurement_mode"] == "correctness"
        assert row["timing"] == "not_measured"
        assert Fraction(row["weight"]["numerator"], row["weight"]["denominator"]) == 0
    fixture_rows = [row for row in rows if row.get("fixture") is True]
    assert len(fixture_rows) == 1


def test_public_manifest_is_the_actual_edge_plus_public_suite(portable, public_runner) -> None:
    public_runtime = load_module("portable_test_public_runtime", "images/work/benchmark/quant_attention_runtime.py")
    manifest = _manifest("public-suite-manifest.json")
    cases = manifest["cases"]
    assert manifest["suite_id"] == "sageattention-qattn-v3-public"
    total = EDGE_COUNT + len(public_runner.PUBLIC_CASES) + len(public_runtime.PUBLIC_REAL_LAYER_CASES)
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["public"].case_count == total == 29
    rows = [case for case in cases if case["role"] == "correctness"]
    assert [row["case_id"] for row in rows] == list(public_runtime.EDGE_CASE_IDS)
    for row, actual in zip(rows, public_runtime.EDGE_CASES):
        for field, value in actual.items():
            assert row[field] == _plain(value), (row["case_id"], field)
    scored = [case for case in cases if case["role"] == "performance"]
    synthetic, real = scored[:len(public_runner.PUBLIC_CASES)], scored[len(public_runner.PUBLIC_CASES):]
    assert [c["case_id"] for c in synthetic] == [c["case_id"] for c in public_runner.PUBLIC_CASES]
    quick = {*public_runner.QUICK_CASE_IDS, public_runner.QUICK_PERF_ONLY_ID}
    for declared, actual in zip(synthetic, public_runner.PUBLIC_CASES):
        for field in ("b", "d", "family", "h_kv", "h_q", "is_causal", "l", "regime", "seed"):
            assert declared[field] == actual[field], (declared["case_id"], field)
        assert declared["quick_suite"] == (actual["case_id"] in quick)
        assert declared["trials"] == {"full": 3, "quick": 1} and declared["warmups"] == 1
    assert [(c["case_id"], c["capture_dir"]) for c in real] == [
        (c["case_id"], c["capture_dir"]) for c in public_runtime.PUBLIC_REAL_LAYER_CASES]
    weights = sum(Fraction(c["weight"]["numerator"], c["weight"]["denominator"]) for c in scored)
    assert weights == 1
    assert manifest["case_counts"] == {"total": 29, "scored_performance": 21, "zero_weight_hard_gate": 8}
    source = (ROOT / "images/work/benchmark/public_runner.py").read_text(encoding="utf-8")
    assert 'trials=3 if a.suite == "full" or a.case else 1' in source
    assert "warmups=1" in source


def test_interactive_manifest_is_the_complete_frozen_final_suite(portable, runtime, verifier) -> None:
    manifest = _manifest("interactive-suite-manifest.json")
    final = _manifest("final-suite-manifest.json")
    assert manifest["suite_id"] == final["suite_id"] == verifier.FINAL_SUITE_ID
    assert manifest["evaluation_policy"] == verifier.EVALUATION_POLICY == "full-final-shared-v3"
    assert manifest["selection_protocol_id"] == verifier.SELECTION_PROTOCOL_ID
    for field in ("cases", "case_counts", "suite_gate_timeouts", "frozen_instance", "draw_policy", "external_fixtures"):
        assert manifest[field] == final[field], field
    assert len(manifest["cases"]) == portable.evaluation_suites[SUITE_ID].roles["interactive"].case_count


def test_final_manifest_withholds_shapes_and_matches_the_frozen_instance(raw_task, portable, runtime, verifier, generator) -> None:
    manifest = _manifest("final-suite-manifest.json")
    cases = manifest["cases"]
    assert manifest["suite_id"] == generator.SUITE_ID == verifier.FINAL_SUITE_ID == "final-holdout-v3"
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["final"].case_count == TOTAL
    _assert_edge_rows_match_runtime(cases, runtime)
    _assert_recipe_bound_fixture(manifest, runtime)
    scored = [case for case in cases if case["role"] == "performance"]
    frozen_path = judge_only_file("judge/private/final_cases.json")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    verifier.validate_final_manifest(frozen)
    assert frozen == generator.generate_manifest()
    assert [(c["case_id"], c["family"], c["tier"]) for c in frozen["cases"]] == [
        (c["case_id"], c["family"], c["tier"]) for c in scored]
    assert manifest["frozen_instance"]["sha256"] == hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    env = _judge_env()
    weights = Fraction(0)
    for case in scored:
        assert case["seed"] == "private_root_only" and case["shape"] == "withheld_root_only"
        assert case["trials"] == env[f"{ENV_PREFIX}VERIFY_TRIALS"]
        assert case["warmups"] == env[f"{ENV_PREFIX}VERIFY_WARMUPS"] == 1
        weights += Fraction(case["weight"]["numerator"], case["weight"]["denominator"])
    assert weights == 1
    assert manifest["case_counts"] == {"total": TOTAL, "scored_performance": SCORED, "zero_weight_hard_gate": EDGE_COUNT, "real_scored": REAL_SCORED}
    # No hidden length, model name or seed appears in the committed manifest.
    text = json.dumps(manifest)
    for case in frozen["cases"]:
        assert f'"l": {case["l"]},' not in text or case["l"] in {c["l"] for c in runtime.EDGE_CASES}
        assert str(case["seed"]) not in text
    assert manifest["draw_policy"]["generator"] == "scripts/generate_final_manifest.py"
    assert manifest["suite_gate_timeouts"] == {
        "candidate_seconds": env[f"{ENV_PREFIX}CANDIDATE_TIMEOUT"],
        "verifier_total_seconds": env[f"{ENV_PREFIX}TOTAL_TIMEOUT"],
        "judge_adapter_seconds": env[f"{ENV_PREFIX}JUDGE_TIMEOUT"],
    }
    final_timeout = raw_task["judge"]["eval_timeout"]
    assert manifest["orchestration_timeouts"]["standalone_final_evaluation_seconds"] == final_timeout
    runner = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    assert f'FINAL_EVAL_TIMEOUT="${{FINAL_EVAL_TIMEOUT:-{final_timeout}}}"' in runner
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert f'"${{final_cases_count}}" -eq {SCORED}' in build
    selection = _manifest("selection-protocol.json")
    for role in ("final", "interactive"):
        assert selection[role]["case_count"] == SCORED and selection[role]["hard_gate_case_count"] == EDGE_COUNT
        assert selection[role]["real_case_count"] == REAL_SCORED and selection[role]["trials"] == 3
        assert selection[role]["warmup_wall_limit_seconds"] == runtime.WARMUP_WALL_LIMIT_SECONDS
    assert selection["edge_fixtures"]["case_count"] == len(runtime.EDGE_CASE_IDS)
    public_runner = load_module("portable_test_public_runner_quick", "images/work/benchmark/public_runner.py")
    assert selection["public"]["quick"]["case_ids"] == [*public_runner.QUICK_CASE_IDS, public_runner.QUICK_PERF_ONLY_ID]



def test_oracle_protocol_matches_runtime_gates(runtime) -> None:
    oracle = _manifest("oracle-protocol.json")
    constants = oracle["tolerances"]["constants"]
    for name in ("L1_MULT", "L1_FLOOR", "HEAD_MULT", "HEAD_FLOOR", "P999_MULT", "P999_FLOOR",
                 "ROWMAX_MULT", "ROWMAX_FLOOR", "COS_FLOOR", "ROW_NORM_FLOOR"):
        assert float(constants[name]) == getattr(runtime, name), name
    assert oracle["reference"]["chunk"] == runtime.REFERENCE_CHUNK
    assert oracle["reference"]["provenance"] == runtime.GENERATOR_PROVENANCE
    assert oracle["tolerances"]["rel_l1"] == f"max({runtime.L1_MULT!r}*reference, {runtime.L1_FLOOR!r})"
    assert float(oracle["tolerances"]["cossim_floor"]) == runtime.COS_FLOOR
    _assert_recipe_bound_fixture(oracle, runtime)



def test_software_abi_matches_the_pinned_images() -> None:
    abi = _manifest("software-abi.json")
    for context in ("work", "judge"):
        dockerfile = (ROOT / f"images/{context}/Dockerfile").read_text(encoding="utf-8")
        assert dockerfile.startswith("FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
        assert "2.8.0" in dockerfile and "12.9" in dockerfile and "3.4.0" in dockerfile
    assert abi["source_runtime"] == ["pytorch-2.8.0", "triton-3.4.0"]
    work = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    absent = re.search(r"for absent in \(([^)]*)\)", work, re.S).group(1)
    assert [x.strip().strip('"') for x in absent.split(",") if x.strip()] == abi["forbidden_packages"]


def test_portable_layers_preserve_metric_and_use_capabilities(raw_task, portable, runtime) -> None:
    metric = portable.metrics[METRIC_ID]
    assert metric.direction == "maximize" == raw_task["judge"]["score_direction"]
    assert metric.unit == "effective_tflops"
    assert metric.comparability == "same_measurement_profile"
    assert metric.baseline_policy == "none"
    assert list(metric.aggregation.spec["families"]) == list(runtime.FAMILIES)
    profile = portable.execution_profiles[PROFILE_ID]
    assert profile.vendor == "nvidia"
    assert profile.count == 1
    assert profile.min_memory_bytes == runtime.MIN_DEVICE_MEMORY_BYTES
    assert tuple(int(part) for part in profile.min_compute_capability.split(".")) == runtime.MIN_COMPUTE_CAPABILITY == MIN_CC
    definition = portable.task_core.definition
    assert [x["family"] for x in definition["families"]] == list(runtime.FAMILIES)
    assert definition["max_length"] == runtime.MAX_L and tuple(definition["head_dims"]) == runtime.SUPPORTED_HEAD_DIMS
    assert portable.defaults.evaluation_suite_id == SUITE_ID
    assert portable.defaults.execution_profile_id == PROFILE_ID
    assert portable.defaults.metric_id == METRIC_ID
    portable_text = json.dumps(raw_task["portable"], sort_keys=True)
    assert "h100" not in portable_text.casefold()
    assert "gpu-" not in portable_text.casefold()
    assert "h100" not in raw_task["name"].casefold()
    assert raw_task["cwd"] == f"/app/workdir/{LAB}"


def test_refresh_tool_reports_no_drift() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/refresh_portable_manifests.py"), "--check"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
