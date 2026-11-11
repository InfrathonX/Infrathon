"""The portable contract freezes the TaskCore byte-for-byte.

Every digest, path set and case count declared by ``portable/*.json`` and the
task JSON is recomputed here from the executable sources (runtime constants,
public runner, Judge verifier, final-holdout generator, Dockerfiles, runner),
never trusted from the declaration.
"""

from __future__ import annotations

import itertools
import json
import random
import re
from fractions import Fraction

import pytest

from conftest import TASK_ROOT, judge_only_file
from sforge.harness.portable_spec import parse_portable_spec, verify_sha256_file


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

ROOT = TASK_ROOT
PORTABLE = ROOT / "portable"
TASK_PATH = ROOT / "tasks/standalone_dynamic_grouped_fp8_h100_scratch_v3.json"
SUITE_ID = "moe-serving-grouped-fp8-v4"
PROFILE_ID = "single-cuda-sm89-56g"
METRIC_ID = "four-family-geomean-effective-tflops"
LAB = "grouped_fp8_lab"
ENV_PREFIX = "STANDALONE_GROUPED_FP8_"
FIXED_FIELDS = ("g", "m_max", "expected_m")
DRAWN_FIELD = "nk"


@pytest.fixture(scope="module")
def raw_task() -> dict:
    return json.loads(TASK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def portable(raw_task):
    return parse_portable_spec(raw_task["portable"], task_id=raw_task["task_id"])


@pytest.fixture(scope="module")
def runtime():
    return load_module("portable_test_runtime", "runtime/grouped_fp8_runtime.py")


@pytest.fixture(scope="module")
def verifier(runtime):
    return load_module("portable_test_verifier", "images/judge/verify_grouped.py")


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


def test_resource_manifest_hashes_every_real_scientific_resource() -> None:
    manifest = _manifest("resource-manifest.json")
    assert manifest["schema_version"] == 1
    expected_paths = {
        "images/work/AGENT_TASK.md",
        f"images/work/workdir/{LAB}/README_AGENT.md",
        "images/work/benchmark/grouped_fp8_runtime.py",
        "images/work/benchmark/public_runner.py",
        "images/work/benchmark/run_public_tests.sh",
        "images/work/benchmark/source_admission.py",
        "images/work/benchmark/public_edge_cases.json",
        "images/work/benchmark/public_cases.json",
        f"images/work/workdir/{LAB}/tools/run_one.sh",
        f"images/work/workdir/{LAB}/tools/run_public_tests.sh",
        "images/judge/candidate_worker.py",
        "images/judge/run_judge.py",
        "images/judge/source_admission.py",
        "images/judge/verify_grouped.py",
        "images/judge/protected_hashes.txt",
        "judge/run_judge.py",
        "runtime/grouped_fp8_runtime.py",
        "scripts/generate_final_manifest.py",
        "scripts/generate_edge_manifest.py",
    }
    assert {resource["path"] for resource in manifest["resources"]} == expected_paths
    for resource in manifest["resources"]:
        verify_sha256_file(ROOT / resource["path"], resource["sha256"])


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
        "runtime/grouped_fp8_runtime.py",
        "scripts/build_images.sh",
        "scripts/generate_final_manifest.py",
        "scripts/generate_edge_manifest.py",
        "scripts/lib.sh",
        "scripts/validate_layout.py",
        "portable/REBUILD_PROMPT.md",
    }
    assert {p for p in declared_paths if not p.startswith("images/")} == extras
    assert "images/work/benchmark/prepare_submit.sh" in declared_paths
    assert not any(p.startswith(("oracle/", "baseline/")) for p in declared_paths)
    for resource in manifest["resources"]:
        verify_sha256_file(ROOT / resource["path"], resource["sha256"])


def test_host_private_final_manifest_cannot_enter_the_work_image() -> None:
    work_context = ROOT / "images/work"
    assert not (PORTABLE / "final-suite-manifest.json").is_relative_to(work_context)
    dockerfile = (work_context / "Dockerfile").read_text(encoding="utf-8").casefold()
    assert "portable" not in dockerfile
    assert "final-suite-manifest" not in dockerfile
    assert "final_cases" not in dockerfile
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert build.count('--build-context "final_cases=${final_cases_context}"') == 1


def _scored(cases: list[dict]) -> list[dict]:
    assert all(case["role"] == "performance" for case in cases)
    return cases


def _family_weight(cases: list[dict], case: dict) -> Fraction:
    families = {item["family"] for item in cases}
    return Fraction(1, len(families) * sum(1 for item in cases if item["family"] == case["family"]))


def test_public_manifest_is_the_actual_public_suite(portable, runtime, public_runner, generator) -> None:
    manifest = _manifest("public-suite-manifest.json")
    cases = _scored(manifest["cases"])
    assert manifest["suite_id"] == SUITE_ID
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["public"].case_count == 60
    assert [case["case_id"] for case in cases] == [c["case_id"] for c in public_runner.PUBLIC_CASES]
    source = (ROOT / "images/work/benchmark/public_runner.py").read_text(encoding="utf-8")
    assert "FULL_TRIALS, FULL_ROUNDS = 3, 1" in source
    for declared, actual in zip(cases, public_runner.PUBLIC_CASES):
        for field in generator.SCORED_KEYS:
            assert declared[field] == actual[field], (declared["case_id"], field)
        runtime.validate_scored_case(actual)
        assert declared["trials"] == (1 if actual["family"] == "sequence" else 3)
        assert declared["family"] in runtime.FAMILIES
        weight = Fraction(declared["weight"]["numerator"], declared["weight"]["denominator"])
        assert weight == _family_weight(list(public_runner.PUBLIC_CASES), actual)
    assert {case["family"] for case in cases} == set(runtime.FAMILIES)
    assert max(case["seed"] for case in cases) < generator.FINAL_SEED_MINIMUM
    public_path = ROOT / "images/work/benchmark/public_cases.json"
    assert manifest["case_manifest"]["sha256"] == __import__("hashlib").sha256(public_path.read_bytes()).hexdigest()
    edge = json.loads((ROOT / "images/work/benchmark/public_edge_cases.json").read_text(encoding="utf-8"))
    assert manifest["case_counts"] == {
        "total": 60 + len(edge["cases"]),
        "scored_performance": 60,
        "zero_weight_hard_gate": len(edge["cases"]),
    }
    assert manifest["edge_suite"]["suite_id"] == edge["suite_id"] == "edge-public-v4"
    assert manifest["edge_suite"]["case_count"] == len(edge["cases"])
    assert manifest["edge_suite"]["sha256"] == __import__("hashlib").sha256(
        (ROOT / "images/work/benchmark/public_edge_cases.json").read_bytes()
    ).hexdigest()


def test_interactive_manifest_is_the_complete_frozen_final_suite(portable, runtime, verifier) -> None:
    manifest = _manifest("interactive-suite-manifest.json")
    final = _manifest("final-suite-manifest.json")
    assert manifest["suite_id"] == final["suite_id"] == verifier.FINAL_SUITE_ID
    assert manifest["evaluation_policy"] == "full-final-shared-v4"
    for field in ("cases", "case_counts", "suite_gate_timeouts", "frozen_instance", "draw_policy", "edge_suite", "workload"):
        assert manifest[field] == final[field], field
    assert len(manifest["cases"]) == portable.evaluation_suites[SUITE_ID].roles["interactive"].case_count


def test_final_manifest_lists_the_frozen_cases_without_hidden_shapes(
    raw_task, portable, runtime, verifier, generator
) -> None:
    manifest = _manifest("final-suite-manifest.json")
    cases = _scored(manifest["cases"])
    assert manifest["suite_id"] == generator.SUITE_ID == verifier.FINAL_SUITE_ID == "final-holdout-v4"
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["final"].case_count == 1000
    edge = json.loads(judge_only_file("judge/private/edge_cases.json").read_text(encoding="utf-8"))
    assert manifest["case_counts"] == {
        "total": 1000 + len(edge["cases"]),
        "scored_performance": 1000,
        "zero_weight_hard_gate": len(edge["cases"]),
    }
    raw = judge_only_file("judge/private/final_cases.json").read_bytes()
    frozen = json.loads(raw)
    assert manifest["frozen_instance"]["sha256"] == __import__("hashlib").sha256(raw).hexdigest()
    assert len(verifier.validate_final_manifest(frozen)) == 1000
    env = _judge_env()
    assert [c["case_id"] for c in cases] == [c["case_id"] for c in frozen["cases"]]
    for declared, actual in zip(cases, frozen["cases"]):
        assert declared["family"] == actual["family"]
        # Hidden shapes, distributions and seeds never appear in a portable manifest.
        assert not {"n", "k", "g", "m_max", "expected_m", "alpha", "hint_scale"} & set(declared)
        assert declared["seed"] == declared["shape"] == "frozen_root_only"
        expected_trials = (
            env[f"{ENV_PREFIX}VERIFY_ROUNDS"] if actual["family"] == "sequence" else env[f"{ENV_PREFIX}VERIFY_TRIALS"]
        )
        assert declared["trials"] == expected_trials
        weight = Fraction(declared["weight"]["numerator"], declared["weight"]["denominator"])
        assert weight == _family_weight(frozen["cases"], actual)
    assert env[f"{ENV_PREFIX}VERIFY_TRIALS"] == 5 and env[f"{ENV_PREFIX}VERIFY_ROUNDS"] == 3
    assert env[f"{ENV_PREFIX}VERIFY_WARMUPS"] == 1
    edge_suite = manifest["edge_suite"]
    assert edge_suite["suite_id"] == verifier.EDGE_SUITE_ID
    assert edge_suite["case_count"] == verifier.EDGE_CASE_COUNT == len(edge["cases"])
    assert edge_suite["sanitize_case_count"] == verifier.EDGE_SANITIZE_COUNT
    assert edge_suite["frozen_instance"]["sha256"] == __import__("hashlib").sha256(
        judge_only_file("judge/private/edge_cases.json").read_bytes()
    ).hexdigest()
    assert manifest["suite_gate_timeouts"] == {
        "candidate_seconds": env[f"{ENV_PREFIX}CANDIDATE_TIMEOUT"],
        "sanitizer_seconds": env[f"{ENV_PREFIX}SANITIZER_TIMEOUT"],
        "verifier_total_seconds": env[f"{ENV_PREFIX}TOTAL_TIMEOUT"],
        "judge_adapter_seconds": env[f"{ENV_PREFIX}JUDGE_TIMEOUT"],
    }
    assert (
        env[f"{ENV_PREFIX}CANDIDATE_TIMEOUT"] + env[f"{ENV_PREFIX}SANITIZER_TIMEOUT"] + 60
        <= env[f"{ENV_PREFIX}TOTAL_TIMEOUT"]
        < env[f"{ENV_PREFIX}JUDGE_TIMEOUT"]
        < raw_task["judge"]["eval_timeout"]
    )
    final_timeout = raw_task["judge"]["eval_timeout"]
    assert manifest["orchestration_timeouts"]["standalone_final_evaluation_seconds"] == final_timeout
    runner = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    assert f'FINAL_EVAL_TIMEOUT="${{FINAL_EVAL_TIMEOUT:-{final_timeout}}}"' in runner
    lib = (ROOT / "scripts/lib.sh").read_text(encoding="utf-8")
    assert "FINAL_CASE_COUNT=1000" in lib and f"EDGE_CASE_COUNT={len(edge['cases'])}" in lib
    selection = _manifest("selection-protocol.json")
    assert selection["final"]["case_count"] == 1000 and selection["final"]["trials"] == 5
    assert selection["final"]["sequence_rounds"] == 3
    assert selection["final"]["edge_case_count"] == len(edge["cases"])


def test_oracle_protocol_matches_runtime_limit(runtime) -> None:
    oracle = _manifest("oracle-protocol.json")
    assert float(oracle["tolerances"]["normalized_error_limit"]) == runtime.SOURCE_ERROR_LIMIT == 1e-3
    assert float(oracle["tolerances"]["exact_error_limit"]) == runtime.EXACT_ERROR_LIMIT == 5e-5
    assert oracle["tolerances"]["source_check_min_elements"] == runtime.SOURCE_CHECK_MIN_ELEMENTS
    assert oracle["reference"]["provenance"] == runtime.GENERATOR_PROVENANCE
    assert oracle["tolerances"]["scale_block"] == runtime.SCALE_BLOCK == 128


def test_software_abi_matches_the_pinned_images() -> None:
    abi = _manifest("software-abi.json")
    for context in ("work", "judge"):
        dockerfile = (ROOT / f"images/{context}/Dockerfile").read_text(encoding="utf-8")
        assert dockerfile.startswith("FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
        assert "2.8.0" in dockerfile and "12.9" in dockerfile and "3.4.0" in dockerfile
    assert abi["source_runtime"] == ["pytorch-2.8.0", "triton-3.4.0"]
    work = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    for absent in abi["forbidden_packages"]:
        assert f'"{absent}"' in work, absent
    assert "fp8-e4m3" in abi["required_dtypes"]


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
    assert tuple(int(part) for part in profile.min_compute_capability.split(".")) == runtime.MIN_COMPUTE_CAPABILITY == (8, 9)
    assert "fp8-e4m3" in profile.features
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
