"""The portable contract freezes the TaskCore byte-for-byte.

Every digest, path set and case count declared by ``portable/*.json`` and the
task JSON is recomputed here from the executable sources (runtime families,
edge fixtures and suite builders, Judge verifier, final-holdout generator,
Dockerfiles, runner), never trusted from the declaration.
"""

from __future__ import annotations

import json
import random
import re
from fractions import Fraction

import pytest

from conftest import TASK_ROOT, judge_only_file
from sforge.harness.portable_spec import parse_portable_spec, verify_sha256_file

ROOT = TASK_ROOT
PORTABLE = ROOT / "portable"
TASK_PATH = ROOT / "tasks/tilelang_mamba2_ssd_forward_h100_seed_v2.json"
SUITE_ID = "tilelang-ssd-v2"
PROFILE_ID = "single-cuda-sm80-73g"
METRIC_ID = "family-geomean-effective-tflops"
LAB = "mamba2_ssd_lab"
ENV_PREFIX = "TILELANG_SSD_"
EDGE_COUNT = 24
SCORED = 138
TOTAL = EDGE_COUNT + SCORED


def load_module(name: str, relative: str):
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
    return load_module("portable_test_runtime", "runtime/mamba2_ssd_runtime.py")


@pytest.fixture(scope="module")
def verifier(runtime):
    return load_module("portable_test_verifier", "images/judge/verify_ssd.py")


@pytest.fixture(scope="module")
def generator():
    return load_module("portable_test_generator", "scripts/generate_final_manifest.py")


def _manifest(name: str) -> dict:
    return json.loads((PORTABLE / name).read_text(encoding="utf-8"))


def _judge_env() -> dict[str, int]:
    dockerfile = (ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    return {name: int(value) for name, value in re.findall(rf"({ENV_PREFIX}[A-Z_]+)=([0-9]+)", dockerfile)}


def _geometry(runtime, name: str) -> dict:
    return {k: int(v) for k, v in runtime.GEOMETRIES[name].items()}


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
    expected_paths = {
        "images/work/AGENT_TASK.md",
        f"images/work/workdir/{LAB}/README_AGENT.md",
        "images/work/benchmark/public_runner.py",
        "images/work/benchmark/run_public_tests.sh",
        "images/work/benchmark/source_admission.py",
        f"images/work/workdir/{LAB}/tools/run_one.sh",
        f"images/work/workdir/{LAB}/tools/run_public_tests.sh",
        "images/work/benchmark/mamba2_ssd_runtime.py",
        "images/judge/candidate_worker.py",
        "images/judge/run_judge.py",
        "images/judge/source_admission.py",
        "images/judge/verify_ssd.py",
        "images/judge/protected_hashes.txt",
        "judge/run_judge.py",
        "runtime/mamba2_ssd_runtime.py",
        "scripts/generate_final_manifest.py",
        "fixtures/recipe/real_layers/REAL_LAYERS.json",
        "fixtures/recipe/real_layers/capture_real_layers.py",
    }
    assert {resource["path"] for resource in manifest["resources"]} == expected_paths
    for resource in manifest["resources"]:
        verify_sha256_file(ROOT / resource["path"], resource["sha256"])


def test_protected_hashes_are_current() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/generate_protected_hashes.py"), "--check"], check=False
    )
    assert result.returncode == 0


def test_build_manifest_covers_exact_portable_delivery_inputs(raw_task) -> None:
    path = PORTABLE / "build-resource-manifest.json"
    declared = raw_task["delivery"]["build_resource_manifest_sha256"]
    assert verify_sha256_file(path, declared) == declared
    manifest = json.loads(path.read_text(encoding="utf-8"))
    declared_paths = {resource["path"] for resource in manifest["resources"]}
    for context_name in ("work", "judge"):
        context = ROOT / f"images/{context_name}"
        actual = {f.relative_to(ROOT).as_posix() for f in context.rglob("*")
                  if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc"}
        assert {r for r in declared_paths if r.startswith(f"images/{context_name}/")} == actual
    extras = {"tasks/BENCHMARK.yaml", "judge/run_judge.py", "runtime/mamba2_ssd_runtime.py", "scripts/build_images.sh",
              "scripts/generate_final_manifest.py", "scripts/lib.sh", "scripts/validate_layout.py", "portable/REBUILD_PROMPT.md",
              "fixtures/recipe/real_layers/REAL_LAYERS.json"}
    assert {p for p in declared_paths if not p.startswith("images/")} == extras
    assert "images/work/benchmark/prepare_submit.sh" in declared_paths
    # The seeded implementation ships in the Work context and is therefore bound.
    assert f"images/work/workdir/{LAB}/candidates/kernel_v001/submission.py" in declared_paths
    assert not any(p.startswith("oracle/") for p in declared_paths)
    for resource in manifest["resources"]:
        verify_sha256_file(ROOT / resource["path"], resource["sha256"])


def test_host_private_final_manifest_cannot_enter_the_work_image() -> None:
    work_context = ROOT / "images/work"
    assert not (PORTABLE / "final-suite-manifest.json").is_relative_to(work_context)
    dockerfile = (work_context / "Dockerfile").read_text(encoding="utf-8").casefold()
    assert "portable" not in dockerfile and "final-suite-manifest" not in dockerfile and "final_cases" not in dockerfile
    assert "real_layers=" not in dockerfile
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert build.count('--build-context "final_cases=${final_cases_context}"') == 1
    assert build.count('--build-context "real_layers=${REAL_LAYERS_DIR}"') == 1


def _assert_edge_rows(cases: list[dict], runtime) -> None:
    rows = [case for case in cases if case["role"] == "correctness"]
    fixtures = runtime.edge_fixtures()
    assert [row["case_id"] for row in rows] == [f["case_id"] for f in fixtures] == [c["case_id"] for c in runtime.EDGE_FIXTURES]
    for row, actual in zip(rows, fixtures):
        for field, value in actual.items():
            if field in ("scored", "check_trial", "layer", "tier"):
                continue
            assert row[field] == value, (row["case_id"], field)
        for key, value in _geometry(runtime, actual["geometry"]).items():
            assert row[key] == value
        assert row["timing"] == "not_measured"
        assert Fraction(row["weight"]["numerator"], row["weight"]["denominator"]) == 0


def test_public_manifest_is_the_actual_edge_plus_public_suite(portable) -> None:
    work = load_module("portable_test_work_runtime", "images/work/benchmark/mamba2_ssd_runtime.py")
    manifest = _manifest("public-suite-manifest.json")
    cases = manifest["cases"]
    public = work.public_suite()
    total = len(work.EDGE_FIXTURES) + len(public)
    assert manifest["suite_id"] == work.PUBLIC_SUITE_ID
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["public"].case_count == total == 45
    _assert_edge_rows(cases, work)
    scored = [case for case in cases if case["role"] == "performance"]
    assert [c["case_id"] for c in scored] == [c["case_id"] for c in public]
    weights = Fraction(0)
    for declared, actual in zip(scored, public):
        for field, value in actual.items():
            if field == "scored":
                continue
            assert declared[field] == value, (declared["case_id"], field)
        for key, value in _geometry(work, actual["geometry"]).items():
            assert declared[key] == value
        expected_trials = 1 if actual["family"] == "real" else 3
        assert declared["trials"] == {"full": expected_trials, "quick": expected_trials}
        assert declared["warmups"] == 1
        assert declared["quick_suite"] == (actual["case_id"] in ("public-p1", "public-p20"))
        weights += Fraction(declared["weight"]["numerator"], declared["weight"]["denominator"])
    assert weights == 1
    assert manifest["case_counts"] == {"total": 45, "scored_performance": 24, "zero_weight_hard_gate": 21}
    source = (ROOT / "images/work/benchmark/public_runner.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--trials", type=int, default=3)' in source
    assert 'parser.add_argument("--warmups", type=int, default=1)' in source
    assert 'QUICK_CASES = ("public-p1", "public-p20")' in source


def test_interactive_manifest_reuses_the_complete_frozen_final_suite(portable, runtime, verifier) -> None:
    interactive = _manifest("interactive-suite-manifest.json")
    final = _manifest("final-suite-manifest.json")
    assert interactive["role"] == "interactive" and final["role"] == "final"
    assert interactive["suite_id"] == final["suite_id"] == verifier.FINAL_SUITE_ID
    assert interactive["cases"] == final["cases"]
    assert interactive["case_counts"] == final["case_counts"]
    assert interactive["suite_gate_timeouts"] == final["suite_gate_timeouts"]
    assert interactive["frozen_instance"] == final["frozen_instance"]
    assert interactive["evaluation_policy"] == verifier.EVALUATION_POLICY
    assert interactive["selection_protocol_id"] == verifier.SELECTION_PROTOCOL_ID
    assert len(interactive["cases"]) == portable.evaluation_suites[SUITE_ID].roles["interactive"].case_count == TOTAL
    scored = [case for case in interactive["cases"] if case["role"] == "performance"]
    assert len(scored) == SCORED
    env = _judge_env()
    assert env[f"{ENV_PREFIX}VERIFY_TRIALS"] == 3
    assert all(
        case["trials"] == (runtime.REAL_TRIALS if case["family"] == "real" else env[f"{ENV_PREFIX}VERIFY_TRIALS"])
        for case in scored
    )
    assert all(case["warmups"] == env[f"{ENV_PREFIX}VERIFY_WARMUPS"] == 1 for case in scored)


def test_final_manifest_withholds_shapes_and_matches_the_frozen_instance(raw_task, portable, runtime, verifier, generator) -> None:
    manifest = _manifest("final-suite-manifest.json")
    cases = manifest["cases"]
    assert manifest["suite_id"] == generator.SUITE_ID == runtime.FINAL_SUITE_ID == verifier.FINAL_SUITE_ID
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["final"].case_count == TOTAL
    _assert_edge_rows(cases, runtime)
    scored = [case for case in cases if case["role"] == "performance"]
    frozen = json.loads(judge_only_file("judge/private/final_cases.json").read_text(encoding="utf-8"))
    runtime.validate_manifest_cases(frozen["cases"])
    assert [(c["case_id"], c["family"], c["tier"]) for c in frozen["cases"]] == [
        (c["case_id"], c["family"], c["tier"]) for c in scored
    ]
    env = _judge_env()
    weights = Fraction(0)
    for case in scored:
        assert case["seed"] == "private_root_only"
        assert case["geometry"] == case["shape"] == case["checked_trial"] == "withheld_root_only"
        assert case["regime"] == "mixed"
        expected_trials = runtime.REAL_TRIALS if case["family"] == "real" else env[f"{ENV_PREFIX}VERIFY_TRIALS"]
        assert case["trials"] == expected_trials and case["warmups"] == env[f"{ENV_PREFIX}VERIFY_WARMUPS"]
        share = Fraction(case["weight"]["numerator"], case["weight"]["denominator"])
        assert share == Fraction(1, 4 * runtime.FINAL_FAMILY_COUNTS[case["family"]])
        weights += share
    assert weights == 1
    assert manifest["frozen_instance"]["sha256"] == __import__("hashlib").sha256(
        judge_only_file("judge/private/final_cases.json").read_bytes()
    ).hexdigest()
    text = json.dumps(manifest)
    for case in frozen["cases"]:
        assert f'"seqlen": {case["seqlen"]},' not in text or case["seqlen"] in {f["seqlen"] for f in runtime.EDGE_FIXTURES}
    assert manifest["draw_policy"]["seed_range"] == [1, 2**63 - 1]
    plan = generator.plan_from_manifest(frozen)
    recipe = json.loads((ROOT / "fixtures/recipe/real_layers/REAL_LAYERS.json").read_text(encoding="utf-8"))
    for seed in (1, 20261006, 424242):
        sample = generator.generate_manifest(plan, recipe, random.Random(seed))
        runtime.validate_manifest_cases(sample["cases"])
    assert manifest["suite_gate_timeouts"] == {
        "candidate_seconds": env[f"{ENV_PREFIX}CANDIDATE_TIMEOUT"],
        "verifier_total_seconds": env[f"{ENV_PREFIX}TOTAL_TIMEOUT"],
        "judge_adapter_seconds": env[f"{ENV_PREFIX}JUDGE_TIMEOUT"],
    }
    assert float(manifest["budget_gates"]["warmup_wall_cap_seconds"]) == verifier.WARMUP_WALL_CAP_SECONDS
    assert float(manifest["budget_gates"]["workspace_multiple"]) == verifier.WORKSPACE_MULTIPLE == runtime.WORKSPACE_MULTIPLE
    assert int(manifest["budget_gates"]["workspace_cap_bytes"]) == verifier.WORKSPACE_CAP_BYTES
    final_timeout = raw_task["judge"]["eval_timeout"]
    assert manifest["orchestration_timeouts"]["standalone_final_evaluation_seconds"] == final_timeout == 5400
    assert final_timeout > env[f"{ENV_PREFIX}JUDGE_TIMEOUT"]
    runner = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    assert f'FINAL_EVAL_TIMEOUT="${{FINAL_EVAL_TIMEOUT:-{final_timeout}}}"' in runner
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert f'"{generator.SUITE_ID}"' in build and '"${final_cases_count}" -eq 138' in build
    selection = _manifest("selection-protocol.json")
    assert selection["final"]["case_count"] == SCORED and selection["final"]["hard_gate_case_count"] == EDGE_COUNT
    assert selection["edge_fixtures"]["case_count"] == len(runtime.EDGE_FIXTURES)
    assert selection["family_weights"] == {family: "1/4" for family in runtime.SCORE_FAMILIES}


def test_oracle_protocol_matches_runtime_tolerances(runtime, verifier) -> None:
    oracle = _manifest("oracle-protocol.json")
    tolerances = oracle["tolerances"]
    assert float(tolerances["rel_l2"]) == runtime.OUT_REL_L2_TOL == verifier.OUT_REL_L2_TOL
    assert float(tolerances["scaled_abs"]) == runtime.OUT_SCALED_ABS_TOL == verifier.OUT_SCALED_ABS_TOL
    assert float(tolerances["rel"]) == runtime.OUT_REL_TOL == verifier.OUT_REL_TOL
    assert oracle["reference"]["decay_regimes"] == list(runtime.REGIMES)


def test_software_abi_matches_the_pinned_images() -> None:
    abi = _manifest("software-abi.json")
    for context in ("work", "judge"):
        dockerfile = (ROOT / f"images/{context}/Dockerfile").read_text(encoding="utf-8")
        assert dockerfile.startswith("FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    work = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    assert "pip install --no-cache-dir tilelang==0.1.13" in work
    assert abi["source_runtime"] == ["pytorch-2.8.0", "triton-3.4.0", "tilelang-0.1.13"]
    absent = re.search(r"for absent in \(([^)]*)\)", work, re.S).group(1)
    assert [x.strip().strip('"') for x in absent.split(",") if x.strip()] == abi["forbidden_packages"]


def test_portable_layers_preserve_metric_and_use_capabilities(raw_task, portable, runtime) -> None:
    metric = portable.metrics[METRIC_ID]
    assert metric.direction == "maximize" == raw_task["judge"]["score_direction"]
    assert metric.unit == "effective_tflops"
    assert metric.comparability == "same_measurement_profile"
    assert metric.baseline_policy == "none"
    assert list(metric.aggregation.spec["families"]) == list(runtime.SCORE_FAMILIES)
    assert list(metric.aggregation.spec["family_weights"]) == ["1/4"] * 4
    profile = portable.execution_profiles[PROFILE_ID]
    assert profile.vendor == "nvidia" and profile.count == 1
    assert profile.min_memory_bytes == runtime.MIN_DEVICE_MEMORY_BYTES == 73 * 1024**3
    assert tuple(int(p) for p in profile.min_compute_capability.split(".")) == runtime.MIN_COMPUTE_CAPABILITY == (8, 0)
    assert "nvcc" in profile.features and "tilelang" in profile.features
    definition = portable.task_core.definition
    assert {name: dict(value) for name, value in definition["geometries"].items()} == {
        name: _geometry(runtime, name) for name in runtime.GEOMETRIES
    }
    assert list(definition["workload_families"]) == list(runtime.SCORE_FAMILIES)
    assert dict(definition["final_family_counts"]) == runtime.FINAL_FAMILY_COUNTS
    assert definition["max_tensor_elements"] == runtime.MAX_TENSOR_ELEMENTS
    assert definition["max_batch"] == runtime.MAX_BATCH
    assert portable.defaults.evaluation_suite_id == SUITE_ID
    assert portable.defaults.execution_profile_id == PROFILE_ID
    portable_text = json.dumps(raw_task["portable"], sort_keys=True)
    assert "h100" not in portable_text.casefold() and "gpu-" not in portable_text.casefold()
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

