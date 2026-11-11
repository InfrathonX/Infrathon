"""The portable contract freezes the TaskCore byte-for-byte.

Every digest, path set and case count declared by ``portable/*.json`` and the
task JSON is recomputed here from the executable sources (runtime edge
fixtures and cells, public runner, Judge verifier, final-holdout generator,
Dockerfiles, runner), never trusted from the declaration.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from fractions import Fraction

import pytest

from conftest import TASK_ROOT
from sforge.harness.portable_spec import parse_portable_spec, verify_sha256_file

ROOT = TASK_ROOT
PORTABLE = ROOT / "portable"
TASK_PATH = ROOT / "tasks/attn_qat_nvfp4_training_attention_h100_seed_v1.json"
SUITE_ID = "attn-qat-train-v1"
PROFILE_ID = "single-cuda-sm89-40g"
METRIC_ID = "family-geomean-effective-tflops"
LAB = "attn_qat_lab"
ENV_PREFIX = "ATTN_QAT_"
EDGE_COUNT = 15
RECIPE_RELATIVE = "fixtures/recipe/RECIPE.json"
SCORED = 287
PUBLIC_SCORED = 29
TOTAL = EDGE_COUNT + SCORED
PUBLIC_TOTAL = EDGE_COUNT + PUBLIC_SCORED
MIN_CC = (8, 9)


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
    return load_module("portable_test_runtime", "runtime/attn_qat_runtime.py")


@pytest.fixture(scope="module")
def verifier(runtime):
    return load_module("portable_test_verifier", "images/judge/verify_attn_qat.py")


@pytest.fixture(scope="module")
def public_runner():
    return load_module("portable_test_public_runner", "images/work/benchmark/public_runner.py")


@pytest.fixture(scope="module")
def generator():
    return load_module("portable_test_generator", "scripts/generate_final_manifest.py")


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _case_weight(table, family: str, tier: str, domains: dict) -> Fraction:
    """A case's share of the primary score: two domains, then families, then tiers, then cases."""
    domain = next(name for name, members in domains.items() if family in members)
    families = {key[0] for key in table if key[0] in domains[domain]}
    tiers = [key for key in table if key[0] == family]
    return (Fraction(1, len(domains)) * Fraction(1, len(families)) * Fraction(1, len(tiers))
            * Fraction(1, table[(family, tier)]))


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
        "images/work/benchmark/attn_qat_runtime.py",
        "images/work/benchmark/public_runner.py",
        "images/work/benchmark/run_public_tests.sh",
        "images/work/benchmark/source_admission.py",
        f"images/work/workdir/{LAB}/tools/run_one.sh",
        f"images/work/workdir/{LAB}/tools/run_public_tests.sh",
        "images/judge/candidate_worker.py",
        "images/judge/run_judge.py",
        "images/judge/source_admission.py",
        "images/judge/verify_attn_qat.py",
        "images/judge/protected_hashes.txt",
        "judge/run_judge.py",
        "runtime/attn_qat_runtime.py",
        "scripts/generate_final_manifest.py",
        "fixtures/recipe/RECIPE.json",
        "fixtures/recipe/capture_e7_wan_grad.py",
        "fixtures/recipe/real_layers/REAL_LAYERS.json",
        "fixtures/recipe/real_layers/PUBLIC_LAYER.json",
        "fixtures/recipe/real_layers/capture_real_layers.py",
        "fixtures/recipe/real_layers/REAL_LAYERS_V2.json",
        "fixtures/recipe/real_layers/PUBLIC_LLM_LAYER.json",
        "fixtures/recipe/real_layers/capture_real_layers_v2.py",
    }
    assert {resource["path"] for resource in manifest["resources"]} == expected_paths
    for resource in manifest["resources"]:
        verify_sha256_file(ROOT / resource["path"], resource["sha256"])
    # The capture is drawn per deployment, so the manifest binds its recipe, not its bytes.
    assert manifest["external_fixtures"] == [
        {
            "name": runtime.FIXTURE_NAME,
            "role": "edge_fixture_input",
            "identity": runtime.FIXTURE_IDENTITY,
            "recipe": RECIPE_RELATIVE,
            "recipe_sha256": "sha256:" + _sha256(ROOT / RECIPE_RELATIVE),
            "source": (
                "captured at build time by fixtures/recipe/build_fixture.sh "
                "(docs/SOURCE_PROVENANCE.md); the bytes are per-deployment and are not "
                "pinned by digest"
            ),
        }
    ]
    layout = (ROOT / "scripts/validate_layout.py").read_text(encoding="utf-8")
    assert f'FIXTURE_RECIPE_ID = "{runtime.FIXTURE_RECIPE_ID}"' in layout
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
        "runtime/attn_qat_runtime.py",
        "scripts/build_images.sh",
        "scripts/generate_final_manifest.py",
        "fixtures/recipe/RECIPE.json",
        "fixtures/recipe/capture_e7_wan_grad.py",
        "fixtures/recipe/build_fixture.sh",
        "fixtures/recipe/real_layers/REAL_LAYERS.json",
        "fixtures/recipe/real_layers/PUBLIC_LAYER.json",
        "fixtures/recipe/real_layers/capture_real_layers.py",
        "fixtures/recipe/real_layers/REAL_LAYERS_V2.json",
        "fixtures/recipe/real_layers/PUBLIC_LLM_LAYER.json",
        "fixtures/recipe/real_layers/capture_real_layers_v2.py",
        "fixtures/recipe/real_layers/build_real_layers.sh",
        "fixtures/recipe/real_layers/Dockerfile.builder",
        "scripts/sforge_judge_entry.py",
        "scripts/lib.sh",
        "scripts/validate_layout.py",
        "portable/REBUILD_PROMPT.md",
    }
    assert {p for p in declared_paths if not p.startswith("images/")} == extras
    assert "images/work/benchmark/prepare_submit.sh" in declared_paths
    # The capture itself is never a declared delivery input: it is drawn per deployment.
    assert not any(p.startswith("oracle/") for p in declared_paths)
    assert not any(
        p.startswith("fixtures/") and not p.startswith("fixtures/recipe/")
        for p in declared_paths
    )
    for resource in manifest["resources"]:
        verify_sha256_file(ROOT / resource["path"], resource["sha256"])
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert build.count('--build-context "fixtures=${MIGRATION_ROOT}/fixtures"') == 1


def test_host_private_final_manifest_cannot_enter_the_work_image() -> None:
    work_context = ROOT / "images/work"
    assert not (PORTABLE / "final-suite-manifest.json").is_relative_to(work_context)
    dockerfile = (work_context / "Dockerfile").read_text(encoding="utf-8").casefold()
    assert "portable" not in dockerfile
    assert "final-suite-manifest" not in dockerfile
    assert "final_cases" not in dockerfile
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert build.count('--build-context "final_cases=${final_cases_context}"') == 1


def _assert_edge_rows_match_runtime(cases: list[dict], runtime, fixtures: int = 1) -> None:
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
    assert len(fixture_rows) == fixtures


def test_public_manifest_is_the_actual_edge_plus_public_suite(portable, public_runner) -> None:
    manifest = _manifest("public-suite-manifest.json")
    cases = manifest["cases"]
    assert manifest["suite_id"] == SUITE_ID
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["public"].case_count == PUBLIC_TOTAL
    work_runtime = load_module("portable_test_work_runtime", "images/work/benchmark/attn_qat_runtime.py")
    _assert_edge_rows_match_runtime(cases, work_runtime, fixtures=0)
    scored = [case for case in cases if case["role"] == "performance"]
    assert [case["case_id"] for case in scored] == [c["case_id"] for c in public_runner.PUBLIC_CASES]
    table = {}
    real = set(public_runner.REAL_FAMILIES)
    domains = {
        domain: [f for f in work_runtime.FAMILIES if work_runtime.FAMILY_DOMAIN[f] == domain]
        for domain in work_runtime.DOMAINS
    }
    for family, *_rest in public_runner.GEOMETRY.values():
        key = (family, "real" if family in real else "public")
        table[key] = table.get(key, 0) + 1
    quick_ids = set(public_runner.QUICK_CASE_IDS)
    for declared, actual in zip(scored, public_runner.PUBLIC_CASES):
        family, b, h, h_kv, d, s_q, s_kv, causal = public_runner.GEOMETRY[actual["cell"]]
        assert declared["cell"] == actual["cell"] and declared["seed"] == actual["seed"]
        assert (declared["family"], declared["b"], declared["h"], declared["h_kv"], declared["d"]) == (
            family, b, h, h_kv, d)
        assert (declared["s_q"], declared["s_kv"], declared["causal"]) == (s_q, s_kv, causal)
        assert declared["domain"] == work_runtime.FAMILY_DOMAIN[family]
        assert declared["trials"] == (1 if family in real else 3)
        assert declared["warmups"] == 1
        assert declared["quick_suite"] == (actual["case_id"] in quick_ids)
        assert Fraction(declared["weight"]["numerator"], declared["weight"]["denominator"]) == (
            _case_weight(table, family, declared["tier"], domains)
        )
    assert sum(
        Fraction(c["weight"]["numerator"], c["weight"]["denominator"]) for c in scored
    ) == 1
    assert manifest["case_counts"] == {"total": PUBLIC_TOTAL, "scored_performance": PUBLIC_SCORED, "zero_weight_hard_gate": EDGE_COUNT}
    source = (ROOT / "images/work/benchmark/public_runner.py").read_text(encoding="utf-8")
    assert 'trials=3 if a.suite == "full" else 1' in source
    assert "warmups=1" in source
    assert public_runner.GEOMETRY == work_runtime.PUBLIC_GRID


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
    assert all(not row["case_id"].startswith("anchor-") for row in scored)
    env = _judge_env()
    assert all(
        case["trials"] == (1 if case["tier"] == "real" else env[f"{ENV_PREFIX}VERIFY_TRIALS"])
        for case in scored
    )
    assert env[f"{ENV_PREFIX}VERIFY_TRIALS"] == 3
    assert {case["domain"] for case in scored} == set(verifier.DOMAINS)
    assert all(case["warmups"] == env[f"{ENV_PREFIX}VERIFY_WARMUPS"] == 1 for case in scored)


def test_final_manifest_matches_the_holdout_generator(raw_task, portable, runtime, generator) -> None:
    manifest = _manifest("final-suite-manifest.json")
    cases = manifest["cases"]
    assert manifest["suite_id"] == "final-holdout-v3" == generator.SUITE_ID
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["final"].case_count == TOTAL
    _assert_edge_rows_match_runtime(cases, runtime)
    scored = [case for case in cases if case["role"] == "performance"]
    assert manifest["case_counts"] == {"total": TOTAL, "scored_performance": SCORED, "zero_weight_hard_gate": EDGE_COUNT}
    assert manifest["draw_policy"]["generator"] == "scripts/generate_final_manifest.py"
    env = _judge_env()
    rows = generator.grid_cases()
    table = {}
    for row in rows:
        key = (row["family"], row["tier"])
        table[key] = table.get(key, 0) + 1
    assert [(c["family"], c["tier"]) for c in scored] == [(r["family"], r["tier"]) for r in rows]
    domains = {
        domain: [f for f in runtime.FAMILIES if runtime.FAMILY_DOMAIN[f] == domain]
        for domain in runtime.DOMAINS
    }
    for case in scored:
        # The manifest declares no geometry: shapes, seeds and sampling stay Judge-only.
        assert not {"b", "h", "h_kv", "d", "s_q", "s_kv", "causal", "layer"} & set(case)
        assert case["domain"] == runtime.FAMILY_DOMAIN[case["family"]]
        assert case["geometry"] == "withheld_root_only" and case["seed"] == "frozen_root_only"
        assert case["warmups"] == env[f"{ENV_PREFIX}VERIFY_WARMUPS"]
        assert case["trials"] == (1 if case["tier"] == "real" else env[f"{ENV_PREFIX}VERIFY_TRIALS"])
        assert Fraction(case["weight"]["numerator"], case["weight"]["denominator"]) == (
            _case_weight(table, case["family"], case["tier"], domains)
        )
    assert sum(
        Fraction(c["weight"]["numerator"], c["weight"]["denominator"]) for c in scored
    ) == 1
    low, high = manifest["draw_policy"]["seed_range"]
    for seed in (1, 20260910, 424242):
        sample = generator.generate_manifest(random.Random(seed))
        assert sample["suite_id"] == manifest["suite_id"]
        assert len(sample["cases"]) == SCORED == generator.EXPECTED_CASE_COUNT
        assert [c["case_id"] for c in sample["cases"]] == [c["case_id"] for c in scored]
        assert all(low <= drawn["seed"] <= high for drawn in sample["cases"])
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
    assert selection["final"]["case_count"] == SCORED and selection["final"]["hard_gate_case_count"] == EDGE_COUNT
    assert selection["edge_fixtures"]["case_count"] == len(runtime.EDGE_CASE_IDS)
    assert selection["public"]["quick"]["case_ids"] == list(public_runner_quick_ids())


def public_runner_quick_ids():
    return load_module("portable_test_public_runner_quick", "images/work/benchmark/public_runner.py").QUICK_CASE_IDS


def test_oracle_protocol_matches_runtime_gates(runtime) -> None:
    oracle = _manifest("oracle-protocol.json")
    tolerances = oracle["tolerances"]
    assert {k: float(v) for k, v in tolerances["scored_rel_l2"].items()} == runtime.SCORED_GATE_LIMITS
    assert {k: float(v) for k, v in tolerances["fixture_floors"].items()} == runtime.GATE_FLOORS
    assert {k: float(v) for k, v in tolerances["fixture_caps"].items()} == runtime.GATE_CAPS
    assert {cid: {k: float(v) for k, v in lim.items()} for cid, lim in tolerances["fixture_rel_l2"].items()} == runtime.FIXTURE_GATE_LIMITS
    assert set(tolerances["fixture_rel_l2"]) == set(runtime.EDGE_CASE_IDS)
    assert oracle["reference"]["provenance"] == runtime.GENERATOR_PROVENANCE
    assert oracle["external_fixtures"][0]["identity"] == runtime.FIXTURE_IDENTITY
    assert oracle["external_fixtures"][0]["recipe"] == RECIPE_RELATIVE
    assert oracle["external_fixtures"][0]["recipe_sha256"] == (
        "sha256:" + _sha256(ROOT / RECIPE_RELATIVE)
    )
    assert "sha256" not in oracle["external_fixtures"][0]
    assert runtime.GATE_TENSORS == ("o", "dq", "dk", "dv")



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
    assert {k: list(v) for k, v in metric.aggregation.spec["domains"].items()} == {
        domain: [f for f in runtime.FAMILIES if runtime.FAMILY_DOMAIN[f] == domain]
        for domain in runtime.DOMAINS
    }
    profile = portable.execution_profiles[PROFILE_ID]
    assert profile.vendor == "nvidia"
    assert profile.count == 1
    assert profile.min_memory_bytes == runtime.MIN_DEVICE_MEMORY_BYTES
    assert tuple(int(part) for part in profile.min_compute_capability.split(".")) == runtime.MIN_COMPUTE_CAPABILITY == MIN_CC
    assert list(portable.task_core.definition["families"]) == list(runtime.FAMILIES)
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
