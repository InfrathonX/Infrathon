"""The portable contract freezes the TaskCore byte-for-byte.

Every digest, path set and case count declared by ``portable/*.json`` and the
task JSON is recomputed here from the executable sources (runtime constants,
public runner, Judge verifier, final-holdout and edge-case generators,
Dockerfiles, runner), never trusted from the declaration.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from fractions import Fraction

import pytest

from conftest import TASK_ROOT, judge_only_file, load_module
from sforge.harness.portable_spec import parse_portable_spec, verify_sha256_file

ROOT = TASK_ROOT
PORTABLE = ROOT / "portable"
TASK_PATH = ROOT / "tasks/standalone_paged_gqa_decode_bf16_h100_scratch_v2.json"
SUITE_ID = "flash-attention-paged-gqa-v2"
PROFILE_ID = "single-cuda-sm80-32g"
METRIC_ID = "family-geomean-effective-tflops"
LAB = "paged_gqa_decode_lab"
ENV_PREFIX = "STANDALONE_PAGED_GQA_"
PUBLIC_EDGE = "images/work/benchmark/public_edge_cases.json"
PRIVATE_EDGE = "judge/private/edge_cases.json"


@pytest.fixture(scope="module")
def raw_task() -> dict:
    return json.loads(TASK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def portable(raw_task):
    return parse_portable_spec(raw_task["portable"], task_id=raw_task["task_id"])


@pytest.fixture(scope="module")
def runtime():
    return load_module("portable_test_runtime", "runtime/paged_gqa_decode_runtime.py")


@pytest.fixture(scope="module")
def verifier(runtime):
    return load_module("portable_test_verifier", "images/judge/verify_paged_gqa.py")


@pytest.fixture(scope="module")
def public_runner():
    return load_module("portable_test_public_runner", "images/work/benchmark/public_runner.py")


@pytest.fixture(scope="module")
def generator():
    return load_module("portable_test_generator", "scripts/generate_final_manifest.py")


@pytest.fixture(scope="module")
def edge_generator():
    return load_module("portable_test_edge_generator", "scripts/generate_edge_manifest.py")


def _manifest(name: str) -> dict:
    return json.loads((PORTABLE / name).read_text(encoding="utf-8"))


def _cases(relative: str) -> list[dict]:
    path = judge_only_file(relative) if relative.startswith("judge/private/") else ROOT / relative
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def _judge_env() -> dict[str, int]:
    dockerfile = (ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    return {
        name: int(value)
        for name, value in re.findall(rf"({ENV_PREFIX}[A-Z_]+)=([0-9]+)", dockerfile)
    }


def _geometry(runtime, cell: str, family: str) -> dict:
    g = runtime.CELL_GEOMETRY[cell]
    return {
        "band": g["band"], "h_q": g["h_q"], "h_kv": g["h_kv"], "model": g["model"],
        "page_size": g["page_size"], "s_q": runtime.FAMILY_S_Q[family], "b": g["b"][family],
    }


def _assert_scored_rows(cases: list[dict], runtime) -> None:
    assert all(case["role"] == "performance" for case in cases)
    for case in cases:
        for key, value in _geometry(runtime, case["cell"], case["family"]).items():
            assert case[key] == value, (case["case_id"], key)
        # sqrt(geomean5(q1) * geomean5(q16)): every case carries exponent 1/10.
        assert Fraction(case["weight"]["numerator"], case["weight"]["denominator"]) == Fraction(1, 10)
        assert case["measurement_mode"] == "decoy_precall_then_fresh_measured_inputs"


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
    assert suite.oracle.protocol_id == _manifest("oracle-protocol.json")["protocol_id"]
    assert suite.selection.protocol_id == _manifest("selection-protocol.json")["protocol_id"]
    assert profile.software_abi.protocol_id == _manifest("software-abi.json")["protocol_id"]
    assert profile.calibration_protocol.protocol_id == _manifest("calibration-protocol.json")["protocol_id"]


def test_resource_manifest_hashes_every_real_scientific_resource() -> None:
    manifest = _manifest("resource-manifest.json")
    assert manifest["schema_version"] == 1
    expected_paths = {
        "images/work/AGENT_TASK.md",
        f"images/work/workdir/{LAB}/README_AGENT.md",
        PUBLIC_EDGE,
        "images/work/benchmark/public_cases.json",
        "images/work/benchmark/paged_gqa_decode_runtime.py",
        "images/work/benchmark/public_runner.py",
        "images/work/benchmark/run_public_tests.sh",
        "images/work/benchmark/source_admission.py",
        f"images/work/workdir/{LAB}/tools/run_one.sh",
        f"images/work/workdir/{LAB}/tools/run_public_tests.sh",
        "images/judge/candidate_worker.py",
        "images/judge/run_judge.py",
        "images/judge/source_admission.py",
        "images/judge/verify_paged_gqa.py",
        "images/judge/protected_hashes.txt",
        "judge/run_judge.py",
        "runtime/paged_gqa_decode_runtime.py",
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
    assert f"tasks/{TASK_PATH.name} (digest-binding root document)" in manifest["boundary"]["excludes"]
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
        "runtime/paged_gqa_decode_runtime.py",
        "scripts/build_images.sh",
        "scripts/generate_final_manifest.py",
        "scripts/generate_edge_manifest.py",
        "scripts/lib.sh",
        "scripts/validate_layout.py",
        "portable/REBUILD_PROMPT.md",
    }
    assert {p for p in declared_paths if not p.startswith("images/")} == extras
    assert "images/work/benchmark/prepare_submit.sh" in declared_paths
    assert PUBLIC_EDGE in declared_paths
    assert not any(p.startswith(("oracle/", "judge/private/")) for p in declared_paths)
    for resource in manifest["resources"]:
        verify_sha256_file(ROOT / resource["path"], resource["sha256"])


def test_host_private_manifests_cannot_enter_the_work_image() -> None:
    work_context = ROOT / "images/work"
    assert not (PORTABLE / "final-suite-manifest.json").is_relative_to(work_context)
    assert not (ROOT / PRIVATE_EDGE).resolve().is_relative_to(work_context.resolve())
    dockerfile = (work_context / "Dockerfile").read_text(encoding="utf-8").casefold()
    assert "portable" not in dockerfile
    assert "final-suite-manifest" not in dockerfile
    assert "final_cases" not in dockerfile and "edge_cases.json" not in dockerfile
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert build.count('--build-context "final_cases=${final_cases_context}"') == 1


def test_public_manifest_is_the_actual_public_suite(portable, runtime, public_runner, edge_generator) -> None:
    manifest = _manifest("public-suite-manifest.json")
    cases = manifest["cases"]
    assert manifest["suite_id"] == SUITE_ID
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["public"].case_count == 127
    assert [case["case_id"] for case in cases] == [c["case_id"] for c in public_runner.PUBLIC_CASES]
    for declared, actual in zip(cases, public_runner.PUBLIC_CASES):
        for field in ("family", "s_q", "h_q", "h_kv", "page_size", "seed"):
            assert declared[field] == actual[field], (declared["case_id"], field)
        runtime.validate_scored_case(actual)
    public_cases = ROOT / "images/work/benchmark/public_cases.json"
    assert manifest["public_cases"]["sha256"] == hashlib.sha256(public_cases.read_bytes()).hexdigest()
    assert manifest["public_cases"]["families"] == {
        family: sum(case["family"] == family for case in cases) for family in runtime.FAMILIES
    }
    source = (ROOT / "images/work/benchmark/public_runner.py").read_text(encoding="utf-8")
    assert 'trials = 1 if args.suite in ("one", "quick") else 3' in source
    edge = json.loads((ROOT / PUBLIC_EDGE).read_text(encoding="utf-8"))
    assert manifest["case_counts"] == {
        "total": 127 + len(edge["cases"]),
        "scored_performance": 127,
        "zero_weight_hard_gate": len(edge["cases"]),
    }
    edge_suite = manifest["edge_suite"]
    assert edge_suite["suite_id"] == edge["suite_id"] == public_runner.PUBLIC_EDGE_SUITE_ID == "edge-public-v4"
    assert edge_suite["case_count"] == len(edge["cases"]) == 44
    assert edge_suite["sanitize_case_count"] == sum(case["sanitize"] for case in edge["cases"])
    assert edge_suite["categories"] == sorted(name for name, *_ in edge_generator.CATEGORY_COUNTS)
    assert edge_suite["path"] == PUBLIC_EDGE
    assert edge_suite["sha256"] == hashlib.sha256((ROOT / PUBLIC_EDGE).read_bytes()).hexdigest()
    selection = _manifest("selection-protocol.json")["public"]
    assert selection["one"]["case_ids"] == [public_runner.ONE_CASE_ID]
    assert selection["quick"]["case_ids"] == list(public_runner.QUICK_CASE_IDS)
    assert selection["edge"]["case_count"] == len(edge["cases"])
    assert selection["sanitize"]["case_count"] == edge_suite["sanitize_case_count"]


def test_interactive_manifest_is_the_complete_frozen_final_suite(portable, verifier) -> None:
    manifest = _manifest("interactive-suite-manifest.json")
    final = _manifest("final-suite-manifest.json")
    assert manifest["role"] == "interactive" and final["role"] == "final"
    assert manifest["suite_id"] == final["suite_id"] == verifier.FINAL_SUITE_ID
    for document in (manifest, final):
        assert document["evaluation_policy"] == verifier.EVALUATION_POLICY == "full-final-shared-v4"
        assert document["selection_protocol_id"] == verifier.SELECTION_PROTOCOL_ID == "paged-gqa-shared-final-v4"
    for field in ("cases", "case_counts", "suite_gate_timeouts", "frozen_instance", "draw_policy", "edge_suite", "workload"):
        assert manifest[field] == final[field], field
    assert len(manifest["cases"]) == portable.evaluation_suites[SUITE_ID].roles["interactive"].case_count
    selection = _manifest("selection-protocol.json")
    assert selection["protocol_id"] == "paged-gqa-shared-final-v4"
    assert selection["evaluation_policy"] == "full-final-shared-v4"
    for key in ("suite_id", "case_count", "edge_case_count", "edge_suite_id", "sanitize_case_count", "trials", "warmups",
                "stream_warmups", "stream_time_limit_seconds"):
        assert selection["interactive"][key] == selection["final"][key], key
    assert selection["interactive"]["candidate_version_override"].startswith("forbidden")


def test_final_manifest_matches_the_holdout_and_edge_generators(raw_task, portable, runtime, verifier, generator, edge_generator) -> None:
    manifest = _manifest("final-suite-manifest.json")
    cases = manifest["cases"]
    assert manifest["suite_id"] == generator.SUITE_ID == verifier.FINAL_SUITE_ID == "final-holdout-v4"
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["final"].case_count == 1114
    assert all(case["role"] == "performance" and set(case) == {"case_id", "family", "role", "weight"}
               for case in cases)
    edge = _cases(PRIVATE_EDGE)
    assert manifest["case_counts"] == {
        "total": 1114 + len(edge),
        "scored_performance": 1114,
        "zero_weight_hard_gate": len(edge),
    }
    frozen = _cases("judge/private/final_cases.json")
    assert [(c["case_id"], c["family"]) for c in cases] == [(c["case_id"], c["family"]) for c in frozen]
    env = _judge_env()
    assert env[f"{ENV_PREFIX}VERIFY_TRIALS"] == verifier.EXPECTED_TRIALS == 3
    assert env[f"{ENV_PREFIX}VERIFY_WARMUPS"] == verifier.EXPECTED_WARMUPS == 1
    assert set(manifest["workload"]["families"]) == set(runtime.FAMILIES)
    assert manifest["draw_policy"]["seed_range"] == [verifier.FINAL_SEED_MINIMUM, verifier.MAX_SEED]
    assert manifest["draw_policy"]["generator"].startswith("scripts/generate_final_manifest.py")
    edge_suite = manifest["edge_suite"]
    assert edge_suite["suite_id"] == verifier.EDGE_SUITE_ID == edge_generator.PRIVATE_SUITE_ID
    assert edge_suite["case_count"] == verifier.EDGE_CASE_COUNT == len(edge)
    assert edge_suite["sanitize_case_count"] == verifier.EDGE_SANITIZE_COUNT == sum(c["sanitize"] for c in edge)
    assert edge_suite["sanitize_only_case_count"] == verifier.EDGE_SANITIZE_ONLY_COUNT == sum(not c["native"] for c in edge)
    assert edge_suite["native_case_count"] == sum(c["native"] for c in edge)
    assert edge_suite["categories"] == sorted(name for name, *_ in edge_generator.CATEGORY_COUNTS)
    assert edge_suite["generator"] == "scripts/generate_edge_manifest.py"
    assert manifest["suite_gate_timeouts"] == {
        "candidate_seconds": env[f"{ENV_PREFIX}CANDIDATE_TIMEOUT"],
        "sanitizer_seconds": env[f"{ENV_PREFIX}SANITIZER_TIMEOUT"],
        "verifier_total_seconds": env[f"{ENV_PREFIX}TOTAL_TIMEOUT"],
        "judge_adapter_seconds": env[f"{ENV_PREFIX}JUDGE_TIMEOUT"],
    }
    # The verifier's total budget must cover the sanitizer pass plus the
    # native worker, and the adapter and orchestration ceilings must nest.
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
    assert "FINAL_CASE_COUNT=1114" in lib and f"EDGE_CASE_COUNT={len(edge)}" in lib
    selection = _manifest("selection-protocol.json")
    assert selection["final"]["case_count"] == selection["final"]["scored_case_count"] == 1114
    assert selection["final"]["trials"] == 3 and selection["final"]["warmups"] == 1
    assert selection["final"]["edge_case_count"] == selection["final"]["hard_gate_case_count"] == len(edge)
    assert selection["final"]["sanitize_case_count"] == verifier.EDGE_SANITIZE_COUNT


def test_oracle_protocol_matches_runtime_tolerances(runtime) -> None:
    oracle = _manifest("oracle-protocol.json")
    constants = oracle["tolerances"]["constants"]
    assert float(constants["max_abs_mult"]) == runtime.MAX_ABS_MULT
    assert float(constants["mean_abs_mult"]) == runtime.MEAN_ABS_MULT
    assert float(constants["max_abs_floor"]) == runtime.MAX_ABS_FLOOR
    assert float(constants["norm_l2_tol"]) == runtime.NORM_L2_TOL
    assert float(oracle["tolerances"]["normalized_l2"]) == runtime.NORM_L2_TOL
    assert oracle["reference"]["provenance"] == runtime.GENERATOR_PROVENANCE
    safety = oracle["memory_safety"]
    assert safety["guard_bytes"] == runtime.GUARD_BYTES
    assert safety["soft_landing"] == {"page_table": runtime.SOFT_PAGE_ID, "cache_seqlens": "s_q"}
    assert "compute_sanitizer_memcheck_subset" in oracle["integrity"]


def test_software_abi_matches_the_pinned_images() -> None:
    abi = _manifest("software-abi.json")
    for context in ("work", "judge"):
        dockerfile = (ROOT / f"images/{context}/Dockerfile").read_text(encoding="utf-8")
        assert dockerfile.startswith("FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
        assert "2.8.0" in dockerfile and "12.9" in dockerfile and "3.4.0" in dockerfile
    assert abi["source_runtime"] == ["pytorch-2.8.0", "triton-3.4.0"]
    assert abi["required_tools"] == ["compute-sanitizer"]
    judge = (ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    assert "test -x /usr/local/cuda/compute-sanitizer/compute-sanitizer" in judge
    work = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    for absent in abi["forbidden_packages"]:
        assert f'"{absent}"' in work, absent


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
    assert profile.min_memory_bytes == runtime.MIN_DEVICE_MEMORY_BYTES == 32 * 1024**3
    assert tuple(int(part) for part in profile.min_compute_capability.split(".")) == runtime.MIN_COMPUTE_CAPABILITY == (8, 0)
    definition = portable.task_core.definition
    assert portable.task_core.contract_id == "paged-gqa-decode-bf16-v2"
    assert list(definition["families"]) == list(runtime.FAMILIES)
    assert definition["support_domain"]["cache_lengths"] == f"s_q <= l_b <= l_max <= {runtime.MAX_L_MAX}"
    domain = definition["support_domain"]
    assert list(domain["b"]) == [1, runtime.MAX_B]
    assert list(domain["s_q"]) == [1, runtime.MAX_S_Q]
    assert domain["h_q_max"] == runtime.MAX_H_Q
    assert tuple(domain["page_sizes"]) == runtime.PAGE_SIZES
    assert portable.defaults.evaluation_suite_id == SUITE_ID
    assert portable.defaults.execution_profile_id == PROFILE_ID
    assert portable.defaults.metric_id == METRIC_ID
    portable_text = json.dumps(raw_task["portable"], sort_keys=True)
    assert "h100" not in portable_text.casefold()
    assert "gpu-" not in portable_text.casefold()
    assert "h100" not in raw_task["name"].casefold()
    assert raw_task["cwd"] == f"/app/workdir/{LAB}"


def test_refresh_tool_reports_no_drift() -> None:
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/refresh_portable_manifests.py"), "--check"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
