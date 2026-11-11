"""The portable contract freezes the TaskCore byte-for-byte.

Every digest, path set and case count declared by ``portable/*.json`` and the
task JSON is recomputed here from the executable sources (runtime edge
fixtures, the protected public case table, the Judge verifier, the frozen
holdout, Dockerfiles, runner), never trusted from the declaration.
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
TASK_PATH = ROOT / "tasks/lmcache_paged_kv_block_transfer_h100_seed_v1.json"
SUITE_ID = "lmcache-kv-transfer-v2"
PROFILE_ID = "single-cuda-sm80-64g"
METRIC_ID = "weighted-group-geomean-payload-gbps"
LAB = "kv_transfer_lab"
ENV_PREFIX = "LMCACHE_KV_"
EDGE_COUNT = 25
PUBLIC_SCORED = 155
FINAL_SCORED = 843


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
    return load_module("portable_test_runtime", "runtime/kv_transfer_runtime.py")


@pytest.fixture(scope="module")
def verifier(runtime):
    return load_module("portable_test_verifier", "images/judge/verify_kv_transfer.py")


@pytest.fixture(scope="module")
def public_runner(runtime):
    return load_module("portable_test_public_runner", "images/work/benchmark/public_runner.py")


@pytest.fixture(scope="module")
def generator(runtime):
    return load_module("portable_test_generator", "scripts/generate_final_manifest.py")


def _manifest(name: str) -> dict:
    return json.loads((PORTABLE / name).read_text(encoding="utf-8"))


def _judge_env() -> dict[str, int]:
    dockerfile = (ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    return {name: int(value) for name, value in re.findall(rf"({ENV_PREFIX}[A-Z_]+)=([0-9]+)", dockerfile)}


def _case_file(name: str) -> list[dict]:
    return json.loads((ROOT / "images/judge" / name).read_text(encoding="utf-8"))["cases"]


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
        "images/work/benchmark/kv_transfer_runtime.py",
        "images/work/benchmark/transfer_abi.h",
        "images/work/benchmark/run_public_tests.sh",
        "images/work/benchmark/source_admission.py",
        f"images/work/workdir/{LAB}/tools/run_one.sh",
        f"images/work/workdir/{LAB}/tools/run_public_tests.sh",
        "images/judge/candidate_worker.py",
        "images/judge/run_judge.py",
        "images/judge/source_admission.py",
        "images/judge/verify_kv_transfer.py",
        "images/judge/protected_hashes.txt",
        "images/judge/public_cases.json",
        "images/work/benchmark/public_cases.json",
        "judge/run_judge.py",
        "runtime/kv_transfer_runtime.py",
        "runtime/transfer_abi.h",
        "scripts/generate_final_manifest.py",
        "scripts/make_public_runtime.py",
    }
    assert {resource["path"] for resource in manifest["resources"]} == expected_paths
    for resource in manifest["resources"]:
        verify_sha256_file(ROOT / resource["path"], resource["sha256"])
    # The protected-hash manifest covers the same case files and ABI header.
    protected = (ROOT / "images/judge/protected_hashes.txt").read_text(encoding="utf-8")
    for label in ("judge:public_cases.json", "worker:transfer_abi.h"):
        assert label in protected


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
    extras = {"tasks/BENCHMARK.yaml", "judge/run_judge.py", "runtime/kv_transfer_runtime.py", "runtime/transfer_abi.h",
              "scripts/build_images.sh", "scripts/generate_final_manifest.py", "scripts/lib.sh", "scripts/validate_layout.py",
              "portable/REBUILD_PROMPT.md"}
    assert {p for p in declared_paths if not p.startswith("images/")} == extras
    assert "images/work/benchmark/prepare_submit.sh" in declared_paths
    assert f"images/work/workdir/{LAB}/candidates/kernel_v001/submission.cu" in declared_paths
    assert not any(p.startswith("oracle/") for p in declared_paths)
    for resource in manifest["resources"]:
        verify_sha256_file(ROOT / resource["path"], resource["sha256"])


def test_host_private_final_manifest_cannot_enter_the_work_image() -> None:
    work_context = ROOT / "images/work"
    assert not (PORTABLE / "final-suite-manifest.json").is_relative_to(work_context)
    dockerfile = (work_context / "Dockerfile").read_text(encoding="utf-8").casefold()
    assert "portable" not in dockerfile and "final-suite-manifest" not in dockerfile and "final_cases" not in dockerfile
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert build.count('--build-context "final_cases=${final_cases_context}"') == 1


def _envelope(runtime) -> dict:
    return {
        "protocol": runtime.ENVELOPE_PROTOCOL, "margin": repr(runtime.ENVELOPE_MARGIN),
        "probe_bytes": runtime.ENVELOPE_PROBE_BYTES, "probe_repeats": runtime.ENVELOPE_PROBE_REPEATS,
        "min_device_gbps": repr(runtime.ENVELOPE_MIN_DEVICE_GBPS), "min_host_gbps": repr(runtime.ENVELOPE_MIN_HOST_GBPS),
    }


def _assert_envelope(manifest: dict, runtime, verifier) -> None:
    declared = manifest["bandwidth_envelope"]
    for key, value in _envelope(runtime).items():
        assert declared[key] == value, key
    assert float(declared["margin"]) == verifier.ENVELOPE_MARGIN
    assert declared["protocol"] == verifier.ENVELOPE_PROTOCOL
    assert declared["probe_bytes"] == verifier.ENVELOPE_PROBE_BYTES
    assert declared["probe_repeats"] == verifier.ENVELOPE_PROBE_REPEATS


def _assert_edge_rows(cases: list[dict], runtime, verifier) -> None:
    rows = [case for case in cases if case["role"] == "correctness"]
    fixtures = runtime.edge_fixtures()
    assert [row["case_id"] for row in rows] == [f["case_id"] for f in fixtures] == list(verifier.EDGE_CASE_IDS)
    assert len(rows) == EDGE_COUNT
    for row, actual in zip(rows, fixtures):
        assert row["arm"] == actual["arm"] and row["family"] == actual["family"]
        assert row["wave_count"] == len(actual["waves"])
        assert row["seed"] == actual["seed"]
        assert row["trials"] == actual["trials"] == 1
        assert row["measurement_mode"] == "correctness_bit_exact"
        assert Fraction(row["weight"]["numerator"], row["weight"]["denominator"]) == 0


def _case_weight(case: dict, cases: list[dict], runtime) -> Fraction:
    family = case["family"]
    group = next(g for g, families in runtime.GROUP_FAMILIES.items() if family in families)
    same = [c for c in cases if c["family"] == family]
    tiers = {c["tier"] for c in same}
    strata = {c["stratum"] for c in same if c["tier"] == case["tier"]}
    count = sum(1 for c in same if (c["tier"], c["stratum"]) == (case["tier"], case["stratum"]))
    return (Fraction(runtime.GROUP_WEIGHTS[group]).limit_denominator(100)
            / len(runtime.GROUP_FAMILIES[group]) / len(tiers) / len(strata) / count)


def _assert_scored_rows(declared_rows: list[dict], actual_cases: list[dict], runtime, *, hidden: bool) -> None:
    assert [r["case_id"] for r in declared_rows] == [c["case_id"] for c in actual_cases]
    total = Fraction(0)
    for declared, actual in zip(declared_rows, actual_cases):
        assert (declared["family"], declared["arm"]) == (actual["family"], actual["arm"])
        assert declared["trials"] == actual["trials"] == runtime.DEFAULT_TRIALS == 3
        weight = Fraction(declared["weight"]["numerator"], declared["weight"]["denominator"])
        assert weight == _case_weight(actual, actual_cases, runtime)
        total += weight
        assert (declared["stratum"], declared["tier"]) == (actual["stratum"], actual["tier"])
        if hidden:
            assert declared["seed"] == declared["shape"] == "frozen_root_only"
            assert "waves" not in declared and "wave_count" not in declared
        else:
            assert declared["seed"] == actual["seed"]
            assert declared["wave_count"] == len(actual["waves"])
            assert declared["foreground"] == actual["foreground"]
    assert total == 1


def test_public_manifest_is_the_actual_edge_plus_public_suite(portable, public_runner) -> None:
    work = load_module("portable_work_runtime", "images/work/benchmark/kv_transfer_runtime.py")
    manifest = _manifest("public-suite-manifest.json")
    cases = manifest["cases"]
    public = work.public_suite()
    total = EDGE_COUNT + PUBLIC_SCORED
    assert manifest["suite_id"] == work.PUBLIC_SUITE_ID
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["public"].case_count == total
    assert manifest["case_counts"] == {"total": total, "scored_performance": PUBLIC_SCORED,
                                       "zero_weight_hard_gate": EDGE_COUNT}
    table = ROOT / manifest["case_table"]["path"]
    assert verify_sha256_file(table, "sha256:" + manifest["case_table"]["sha256"])
    edges = [case for case in cases if case["role"] == "correctness"]
    scored = [case for case in cases if case["role"] == "performance"]
    for row, actual in zip(edges, work.edge_fixtures(), strict=True):
        assert (row["case_id"], row["arm"], row["seed"]) == (actual["case_id"], actual["arm"], actual["seed"])
    _assert_scored_rows(scored, public, work, hidden=False)
    quick = set(public_runner.QUICK_CASES)
    assert {case["family"] for case in public if case["case_id"] in quick} == set(work.FAMILIES)
    for declared in scored:
        assert declared["quick_suite"] == (declared["case_id"] in quick)
    assert "bandwidth_envelope" not in manifest
    source = (ROOT / "images/work/benchmark/public_runner.py").read_text(encoding="utf-8")
    assert '"--trials", type=int, default=3' in source


def test_interactive_manifest_is_the_complete_frozen_final_suite(portable, runtime, verifier) -> None:
    manifest = _manifest("interactive-suite-manifest.json")
    final = _manifest("final-suite-manifest.json")
    assert manifest["suite_id"] == final["suite_id"] == verifier.FINAL_SUITE_ID
    assert manifest["evaluation_policy"] == "full-final-shared-v3"
    for field in ("cases", "case_counts", "family_counts", "suite_gate_timeouts", "frozen_instance", "draw_policy"):
        assert manifest[field] == final[field], field
    assert len(manifest["cases"]) == portable.evaluation_suites[SUITE_ID].roles["interactive"].case_count


def test_final_manifest_matches_the_frozen_holdout(raw_task, portable, runtime, verifier, generator) -> None:
    manifest = _manifest("final-suite-manifest.json")
    cases = manifest["cases"]
    assert manifest["suite_id"] == generator.SUITE_ID == runtime.FINAL_SUITE_ID == verifier.FINAL_SUITE_ID
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["final"].case_count == EDGE_COUNT + FINAL_SCORED
    _assert_edge_rows(cases, runtime, verifier)
    scored = [case for case in cases if case["role"] == "performance"]
    assert len(scored) == FINAL_SCORED == verifier.EXPECTED_FINAL_COUNT
    assert manifest["family_counts"] == verifier.FINAL_FAMILY_COUNTS
    instance = judge_only_file("judge/private/final_cases.json")
    assert verify_sha256_file(instance, "sha256:" + manifest["frozen_instance"]["sha256"])
    frozen = runtime.load_manifest(instance)["cases"]
    _assert_scored_rows(scored, frozen, runtime, hidden=True)
    text = json.dumps(manifest)
    for case in frozen[:20]:
        assert str(case["seed"]) not in text
    env = _judge_env()
    assert manifest["suite_gate_timeouts"]["candidate_seconds"] == env[f"{ENV_PREFIX}CANDIDATE_TIMEOUT"]
    assert manifest["suite_gate_timeouts"]["verifier_total_seconds"] == env[f"{ENV_PREFIX}TOTAL_TIMEOUT"]
    assert manifest["suite_gate_timeouts"]["judge_adapter_seconds"] == env[f"{ENV_PREFIX}JUDGE_TIMEOUT"]
    _assert_envelope(manifest, runtime, verifier)
    final_timeout = raw_task["judge"]["eval_timeout"]
    assert manifest["orchestration_timeouts"]["standalone_final_evaluation_seconds"] == final_timeout
    assert final_timeout > env[f"{ENV_PREFIX}JUDGE_TIMEOUT"]
    runner = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    assert f'FINAL_EVAL_TIMEOUT="${{FINAL_EVAL_TIMEOUT:-{final_timeout}}}"' in runner
    assert 'JUDGE_PORT="${JUDGE_PORT:-18496}"' in runner
    assert 'score_unit: "effective_gbps"' in runner
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert f'"{generator.SUITE_ID}"' in build and f'"${{final_cases_count}}" -eq {FINAL_SCORED}' in build
    selection = _manifest("selection-protocol.json")
    assert selection["final"]["case_count"] == FINAL_SCORED and selection["final"]["hard_gate_case_count"] == EDGE_COUNT
    assert {g: float(w) for g, w in selection["group_weights"].items()} == runtime.GROUP_WEIGHTS == verifier.GROUP_WEIGHTS


def test_oracle_and_calibration_match_runtime_constants(runtime, verifier) -> None:
    oracle = _manifest("oracle-protocol.json")
    assert oracle["reference"]["constants"] == {"bs": runtime.BS, "head_sizes": list(runtime.HEAD_SIZES), "esize": runtime.ESIZE}
    assert oracle["reference"]["scratch_bytes"] == runtime.SCRATCH_BYTES == verifier.EXPECTED_CONSTANTS["scratch_bytes"]
    assert f"0x{runtime.POISON:04X}" in oracle["reference"]["untargeted_bytes"]
    assert oracle["tolerances"] == {"any": "zero"}
    _assert_envelope(oracle, runtime, verifier)
    calibration = _manifest("calibration-protocol.json")
    assert float(calibration["settle_seconds"]) == runtime.SETTLE_SECONDS
    assert float(calibration["device_settle_seconds"]) == runtime.DEVICE_SETTLE_SECONDS
    assert calibration["comparability"] == "same_measurement_profile_including_host_platform"
    _assert_envelope(calibration, runtime, verifier)


def test_software_abi_matches_the_pinned_images_and_compile_command(runtime) -> None:
    abi = _manifest("software-abi.json")
    for context in ("work", "judge"):
        dockerfile = (ROOT / f"images/{context}/Dockerfile").read_text(encoding="utf-8")
        assert dockerfile.startswith("FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    work = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    assert 'assert "release 12.9" in release' in work
    assert abi["entry_point"] == runtime.ENTRYPOINT == "lmcache_paged_kv_block_transfer"
    assert abi["compile"]["abi_header"] == "runtime/transfer_abi.h"
    argv = runtime.compile_argv(ROOT / "x.cu", ROOT / "x.so", ROOT / "runtime", "sm_80")
    for flag in ("-O3", "-std=c++17", "-arch=sm_80", "--shared", "--cudart=shared"):
        assert flag in argv, flag
    assert runtime.native_arch.__doc__ and "sm_XYa" in runtime.native_arch.__doc__


def test_portable_layers_preserve_metric_and_use_capabilities(raw_task, portable, runtime, verifier) -> None:
    metric = portable.metrics[METRIC_ID]
    assert metric.direction == "maximize" == raw_task["judge"]["score_direction"]
    assert metric.unit == "effective_gbps"
    assert metric.comparability == "same_measurement_profile"
    assert metric.baseline_policy == "none"
    assert dict(metric.aggregation.spec["group_weight_basis_points"]) == {g: int(w * 10000) for g, w in runtime.GROUP_WEIGHTS.items()}
    assert metric.aggregation.spec["host_platform_in_comparability_key"] is True
    profile = portable.execution_profiles[PROFILE_ID]
    assert profile.vendor == "nvidia" and profile.count == 1
    assert profile.min_memory_bytes == runtime.MIN_DEVICE_MEMORY_BYTES == verifier.MIN_DEVICE_MEMORY_BYTES
    assert tuple(int(p) for p in profile.min_compute_capability.split(".")) == runtime.MIN_COMPUTE_CAPABILITY == verifier.MIN_COMPUTE_CAPABILITY == (8, 0)
    assert "pinned-host-memory" in profile.features and "nvcc" in profile.features
    definition = portable.task_core.definition
    assert list(definition["arms"]) == list(runtime.ARMS)
    domain = {key: (list(value) if isinstance(value, tuple) else value) for key, value in dict(definition["support_domain"]).items()}
    assert domain["nl"] == list(runtime.NL_RANGE) and domain["nh"] == list(runtime.NH_RANGE)
    assert domain["nb"] == list(runtime.NB_RANGE) and domain["head_sizes"] == list(runtime.HEAD_SIZES)
    assert dict(definition["families"]) == runtime.FAMILIES
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
