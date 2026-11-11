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
TASK_PATH = ROOT / "tasks/standalone_sparse_mla_prefill_h100_scratch_v2.json"
SUITE_ID = "flashmla-sparse-prefill-v2"
PROFILE_ID = "single-cuda-sm80-32g"
METRIC_ID = "family-geomean-effective-tflops"
LAB = "sparse_mla_prefill_lab"
ENV_PREFIX = "STANDALONE_SPARSE_MLA_"
PUBLIC_EDGE = "images/work/benchmark/public_edge_cases.json"
PUBLIC_CASES = "images/work/benchmark/public_cases.json"
PRIVATE_EDGE = "judge/private/edge_cases.json"


@pytest.fixture(scope="module")
def raw_task() -> dict:
    return json.loads(TASK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def portable(raw_task):
    return parse_portable_spec(raw_task["portable"], task_id=raw_task["task_id"])


@pytest.fixture(scope="module")
def runtime():
    return load_module("portable_test_runtime", "runtime/sparse_mla_prefill_runtime.py")


@pytest.fixture(scope="module")
def verifier(runtime):
    return load_module("portable_test_verifier", "images/judge/verify_sparse_mla.py")


@pytest.fixture(scope="module")
def public_runtime():
    return load_module("portable_test_public_runtime", "images/work/benchmark/sparse_mla_prefill_runtime.py")


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


def _expected_weights(cases: list[dict], group_tier: dict[str, str]) -> dict[str, Fraction]:
    """The exponent of each case in the four-level geometric mean.

    score = geomean_f sqrt(geomean_{g in evidence} G * geomean_{g in sweep} G),
    G = geomean of the group's cases, so a case of family f, group g, tier t
    carries 1 / (families * tiers_of_f * groups_of_(f,t) * cases_of_(f,g)).
    """
    cells: dict[tuple[str, str], int] = {}
    for case in cases:
        cells[(case["family"], case["group"])] = cells.get((case["family"], case["group"]), 0) + 1
    families = {family for family, _group in cells}
    weights = {}
    for case in cases:
        family, group = case["family"], case["group"]
        tier = group_tier[group]
        tiers = {group_tier[g] for f, g in cells if f == family}
        groups = sum(1 for f, g in cells if f == family and group_tier[g] == tier)
        weights[case["case_id"]] = Fraction(1, len(families) * len(tiers) * groups * cells[(family, group)])
    return weights


def _weight(case: dict) -> Fraction:
    return Fraction(case["weight"]["numerator"], case["weight"]["denominator"])


def _assert_scored_rows(cases: list[dict], runtime) -> None:
    """Rows of the final and interactive manifests: identities only, shapes root-only."""
    assert all(case["role"] == "performance" for case in cases)
    assert {case["group"] for case in cases} == set(runtime.GROUPS)
    assert {case["family"] for case in cases} == set(runtime.FAMILIES)
    for case in cases:
        assert set(case) == {"case_id", "family", "group", "measurement", "role", "seed", "shape", "tier",
                             "trials", "weight"}
        assert case["seed"] == case["shape"] == "frozen_root_only"
        assert case["tier"] == runtime.GROUP_TIER[case["group"]]
        assert case["measurement"] == ("stream" if case["group"] in runtime.STREAM_GROUPS else "trials")
    # Every anchor's weight follows from the disclosed anchor grid alone: four families,
    # two tiers, five evidence groups and the family's anchor shapes.
    assert sum(tier == "evidence" for tier in runtime.GROUP_TIER.values()) == 5
    for case in cases:
        if case["group"] == "anchor":
            assert _weight(case) == Fraction(1, 4 * 2 * 5 * len(runtime.ANCHOR_S_KV[case["family"]]))
    assert sum(_weight(case) for case in cases) == 1


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
        PUBLIC_CASES,
        "images/work/benchmark/public_runner.py",
        "images/work/benchmark/run_public_tests.sh",
        "images/work/benchmark/source_admission.py",
        "images/work/benchmark/sparse_mla_prefill_runtime.py",
        f"images/work/workdir/{LAB}/tools/run_one.sh",
        f"images/work/workdir/{LAB}/tools/run_public_tests.sh",
        "images/judge/candidate_worker.py",
        "images/judge/run_judge.py",
        "images/judge/source_admission.py",
        "images/judge/verify_sparse_mla.py",
        "images/judge/protected_hashes.txt",
        "judge/run_judge.py",
        "runtime/sparse_mla_prefill_runtime.py",
        "scripts/generate_final_manifest.py",
        "scripts/generate_edge_manifest.py",
        "scripts/generate_public_cases.py",
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
        "runtime/sparse_mla_prefill_runtime.py",
        "scripts/build_images.sh",
        "scripts/generate_final_manifest.py",
        "scripts/generate_edge_manifest.py",
        "scripts/generate_public_cases.py",
        "scripts/lib.sh",
        "scripts/validate_layout.py",
        "portable/REBUILD_PROMPT.md",
    }
    assert {p for p in declared_paths if not p.startswith("images/")} == extras
    assert "images/work/benchmark/prepare_submit.sh" in declared_paths
    assert PUBLIC_EDGE in declared_paths and PUBLIC_CASES in declared_paths
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


def test_public_manifest_is_the_actual_public_suite(portable, public_runtime, public_runner, edge_generator) -> None:
    manifest = _manifest("public-suite-manifest.json")
    cases = manifest["cases"]
    public_cases = public_runner.public_cases()
    assert manifest["suite_id"] == SUITE_ID
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["public"].case_count == 108
    assert len(public_cases) == 108
    assert all(case["role"] == "performance" for case in cases)
    assert [case["case_id"] for case in cases] == [c["case_id"] for c in public_cases]
    expected_weights = _expected_weights(public_cases, public_runtime.GROUP_TIER)
    for declared, actual in zip(cases, public_cases):
        for field in ("family", "group", "h_q", "s_q", "s_kv", "topk", "sm_scale", "seed"):
            assert declared[field] == actual[field], (declared["case_id"], field)
        assert declared["seed"] < 2**32
        assert declared["tier"] == public_runtime.GROUP_TIER[actual["group"]]
        assert declared["measurement"] == ("stream" if actual["group"] in ("trace", "replay") else "trials")
        assert _weight(declared) == expected_weights[actual["case_id"]], declared["case_id"]
    assert sum(_weight(case) for case in cases) == 1
    case_suite = manifest["case_suite"]
    assert case_suite == {
        "generator": "scripts/generate_public_cases.py",
        "path": PUBLIC_CASES,
        "sha256": hashlib.sha256((ROOT / PUBLIC_CASES).read_bytes()).hexdigest(),
        "suite_id": public_runner.PUBLIC_CASE_SUITE_ID,
    }
    source = (ROOT / "images/work/benchmark/public_runner.py").read_text(encoding="utf-8")
    assert 'trials = 3 if args.suite == "full" or args.case else 1' in source
    edge = json.loads((ROOT / PUBLIC_EDGE).read_text(encoding="utf-8"))
    assert manifest["case_counts"] == {
        "total": 108 + len(edge["cases"]),
        "scored_performance": 108,
        "zero_weight_hard_gate": len(edge["cases"]),
    }
    edge_suite = manifest["edge_suite"]
    assert edge_suite["suite_id"] == edge["suite_id"] == public_runner.PUBLIC_EDGE_SUITE_ID == "edge-public-v3"
    assert edge_suite["case_count"] == len(edge["cases"]) == 73
    assert edge_suite["sanitize_case_count"] == sum(case["sanitize"] for case in edge["cases"]) == 7
    assert edge_suite["categories"] == sorted(name for name, _ in edge_generator.PUBLIC_CATEGORY_COUNTS)
    assert edge_suite["path"] == PUBLIC_EDGE
    assert edge_suite["sha256"] == hashlib.sha256((ROOT / PUBLIC_EDGE).read_bytes()).hexdigest()
    semantics = manifest["selection_semantics"]
    assert public_runner.ONE_CASE_ID in semantics and "108 public cases" in semantics
    assert "73 public edge cases" in semantics and "7 public sanitizer cases" in semantics
    quick = [case["case_id"] for case in public_runner.cases_for_suite("quick")]
    selection = _manifest("selection-protocol.json")["public"]
    assert selection["one"]["case_ids"] == [public_runner.ONE_CASE_ID]
    assert selection["quick"]["case_ids"] == quick
    assert selection["quick"]["case_count"] == len(quick) == 8
    assert selection["full"]["case_count"] == 108
    assert selection["edge"]["case_count"] == len(edge["cases"])
    assert selection["edge"]["suite_id"] == selection["sanitize"]["suite_id"] == "edge-public-v3"
    assert selection["sanitize"]["case_count"] == edge_suite["sanitize_case_count"]


def test_interactive_manifest_is_the_complete_frozen_final_suite(portable, verifier) -> None:
    manifest = _manifest("interactive-suite-manifest.json")
    final = _manifest("final-suite-manifest.json")
    assert manifest["role"] == "interactive" and final["role"] == "final"
    assert manifest["suite_id"] == final["suite_id"] == verifier.FINAL_SUITE_ID
    for document in (manifest, final):
        assert document["evaluation_policy"] == verifier.EVALUATION_POLICY == "full-final-shared-v3"
        assert document["selection_protocol_id"] == verifier.SELECTION_PROTOCOL_ID == "sparse-mla-shared-final-v3"
    for field in ("cases", "case_counts", "suite_gate_timeouts", "frozen_instance", "draw_policy", "edge_suite",
                  "workload"):
        assert manifest[field] == final[field], field
    assert len(manifest["cases"]) == portable.evaluation_suites[SUITE_ID].roles["interactive"].case_count
    selection = _manifest("selection-protocol.json")
    assert selection["protocol_id"] == "sparse-mla-shared-final-v3"
    assert selection["evaluation_policy"] == "full-final-shared-v3"
    for key in ("suite_id", "case_count", "edge_case_count", "edge_suite_id", "sanitize_case_count", "trials", "warmups"):
        assert selection["interactive"][key] == selection["final"][key], key
    assert selection["interactive"]["suite_id"] == verifier.FINAL_SUITE_ID
    assert selection["interactive"]["case_count"] == len(manifest["cases"])
    assert selection["interactive"]["edge_suite_id"] == verifier.EDGE_SUITE_ID
    assert selection["interactive"]["candidate_version_override"].startswith("forbidden")


def test_final_manifest_rows_are_the_frozen_instance(runtime, verifier) -> None:
    """Judge-only: every declared row names the frozen case, and its weight is its exponent."""
    manifest = _manifest("final-suite-manifest.json")
    cases = manifest["cases"]
    frozen = _cases("judge/private/final_cases.json")
    assert [c["case_id"] for c in cases] == [c["case_id"] for c in frozen]
    for declared, actual in zip(cases, frozen):
        assert (declared["family"], declared["group"]) == (actual["family"], actual["group"])
    weights = _expected_weights(frozen, runtime.GROUP_TIER)
    for declared in cases:
        assert _weight(declared) == weights[declared["case_id"]], declared["case_id"]
    edge = _cases(PRIVATE_EDGE)
    edge_suite = manifest["edge_suite"]
    assert edge_suite["case_count"] == len(edge) == verifier.EDGE_CASE_COUNT
    assert edge_suite["sanitize_case_count"] == sum(c["sanitize"] for c in edge) == verifier.EDGE_SANITIZE_COUNT
    assert edge_suite["sanitize_only_case_count"] == sum(not c["native"] for c in edge)
    assert edge_suite["native_case_count"] == sum(c["native"] for c in edge)


def test_final_manifest_matches_the_holdout_and_edge_generators(raw_task, portable, runtime, verifier, generator, edge_generator) -> None:
    manifest = _manifest("final-suite-manifest.json")
    cases = manifest["cases"]
    assert manifest["suite_id"] == generator.SUITE_ID == verifier.FINAL_SUITE_ID == "final-holdout-v4"
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["final"].case_count == 997
    _assert_scored_rows(cases, runtime)
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["group"]] = counts.get(case["group"], 0) + 1
    assert counts == verifier.FINAL_GROUP_COUNTS
    assert manifest["case_counts"] == {
        "scored_by_group": dict(sorted(verifier.FINAL_GROUP_COUNTS.items())),
        "total": 997 + verifier.EDGE_CASE_COUNT,
        "scored_performance": 997,
        "zero_weight_hard_gate": verifier.EDGE_CASE_COUNT,
    }
    env = _judge_env()
    for declared in cases:
        stream = declared["group"] in runtime.STREAM_GROUPS
        assert declared["trials"] == (1 if stream else env[f"{ENV_PREFIX}VERIFY_TRIALS"])
    assert env[f"{ENV_PREFIX}VERIFY_TRIALS"] == verifier.EXPECTED_TRIALS == 3
    assert env[f"{ENV_PREFIX}VERIFY_WARMUPS"] == verifier.EXPECTED_WARMUPS == 1
    workload = manifest["workload"]
    assert workload["families"] == {
        family: {
            "d_qk": runtime.FAMILY_GEOMETRY[family][0], "h_q": runtime.FAMILY_GEOMETRY[family][1],
            "max_context": runtime.FAMILY_MAX_CONTEXT[family], "model": runtime.FAMILY_MODEL[family],
            "topk": runtime.FAMILY_TOPK[family],
        }
        for family in runtime.FAMILIES
    }
    assert set(workload["groups"]) == set(runtime.GROUPS)
    assert "s_q 4096" in workload["groups"]["anchor"] and "sm_scale 0.5" in workload["groups"]["anchor"]
    assert manifest["draw_policy"]["seed_range"] == [verifier.FINAL_SEED_MINIMUM, verifier.MAX_SEED]
    assert manifest["draw_policy"]["generator"].startswith("scripts/generate_final_manifest.py --plan ")
    assert "never enters the repository" in manifest["draw_policy"]["plan"]
    assert manifest["frozen_instance"]["path"] == "judge/private/final_cases.json"
    assert manifest["edge_suite"]["frozen_instance"]["path"] == PRIVATE_EDGE
    semantics = manifest["selection_semantics"]
    for fragment in ("997 scored cases", f"median of {verifier.EXPECTED_TRIALS} trials",
                     f"{runtime.STREAM_WARM_CALLS} untimed warm-up calls",
                     f"{int(runtime.STREAM_TIME_LIMIT_SECONDS)} timed seconds", "sqrt(evidence x sweep)"):
        assert fragment in semantics, fragment
    edge_suite = manifest["edge_suite"]
    assert edge_suite["suite_id"] == verifier.EDGE_SUITE_ID == edge_generator.PRIVATE_SUITE_ID
    assert edge_suite["case_count"] == verifier.EDGE_CASE_COUNT == edge_generator.PRIVATE_CASE_COUNT == 470
    assert edge_suite["sanitize_case_count"] == verifier.EDGE_SANITIZE_COUNT == edge_generator.PRIVATE_SANITIZE_COUNT
    assert edge_suite["sanitize_only_case_count"] == verifier.EDGE_SANITIZE_ONLY_COUNT == 1
    assert edge_suite["native_case_count"] == verifier.EDGE_CASE_COUNT - verifier.EDGE_SANITIZE_ONLY_COUNT
    assert edge_suite["categories"] == sorted(name for name, *_ in edge_generator.CATEGORY_COUNTS)
    assert dict(edge_generator.CATEGORY_COUNTS) == verifier.EDGE_CATEGORY_COUNTS
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
    assert "FINAL_CASE_COUNT=997" in lib and f"EDGE_CASE_COUNT={verifier.EDGE_CASE_COUNT}" in lib
    selection = _manifest("selection-protocol.json")
    assert selection["final"]["suite_id"] == verifier.FINAL_SUITE_ID
    assert selection["final"]["edge_suite_id"] == verifier.EDGE_SUITE_ID
    assert selection["final"]["case_count"] == selection["final"]["scored_case_count"] == 997
    assert selection["final"]["trials"] == verifier.EXPECTED_TRIALS and selection["final"]["warmups"] == 1
    assert selection["final"]["edge_case_count"] == selection["final"]["hard_gate_case_count"] == (
        verifier.EDGE_CASE_COUNT
    )
    assert selection["final"]["sanitize_case_count"] == verifier.EDGE_SANITIZE_COUNT


def test_oracle_protocol_matches_runtime_tolerances(runtime, verifier) -> None:
    oracle = _manifest("oracle-protocol.json")
    tolerances = oracle["tolerances"]
    assert float(tolerances["out"]["abs"]) == runtime.OUT_ABS_TOL
    assert float(tolerances["out"]["cos"]) == runtime.OUT_COS_TOL
    assert eval(tolerances["out"]["rel"]) == runtime.OUT_REL_TOL  # noqa: S307 - literal "3.01/128"
    for name in ("max_logits", "lse"):
        assert float(tolerances[name]["abs"]) == runtime.LOGIT_ABS_TOL
        assert float(tolerances[name]["cos"]) == runtime.LOGIT_COS_TOL
        assert eval(tolerances[name]["rel"]) == runtime.LOGIT_REL_TOL  # noqa: S307
    assert oracle["reference"]["provenance"] == runtime.GENERATOR_PROVENANCE
    assert oracle["reference"]["chunk_bytes"] == runtime.REFERENCE_CHUNK_BYTES
    safety = oracle["memory_safety"]
    assert safety["guard"] == {
        "min_bytes": runtime.GUARD_MIN_BYTES, "rows": runtime.GUARD_ROWS, "fill_byte": runtime.GUARD_FILL_BYTE,
    }
    assert safety["soft_landing"] == verifier.SOFT_LANDING
    assert str(verifier.SANITIZER_PADDING_BYTES) in safety["sanitizer"]
    assert "compute_sanitizer_memcheck_subset" in oracle["integrity"]
    assert "l2_flush_before_timed_call" in oracle["integrity"]


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
    assert portable.task_core.contract_id == "sparse-mla-prefill-bf16-v3"
    families = definition["families"]
    assert set(families) == set(runtime.FAMILIES)
    for family in runtime.FAMILIES:
        assert dict(families[family]) == {
            "model": runtime.FAMILY_MODEL[family],
            "d_qk": runtime.FAMILY_GEOMETRY[family][0],
            "h_q": runtime.FAMILY_GEOMETRY[family][1],
            "topk": runtime.FAMILY_TOPK[family],
            "max_context": runtime.FAMILY_MAX_CONTEXT[family],
        }
    assert set(definition["scored_groups"]) == set(runtime.GROUPS)
    assert "s_q 4096" in definition["scored_groups"]["anchor"]
    assert "sm_scale 0.5" in definition["scored_groups"]["anchor"]
    assert definition["d_v"] == runtime.D_V and definition["h_kv"] == runtime.H_KV
    domain = definition["support_domain"]
    assert list(domain["s_q"]) == [1, runtime.MAX_S_Q]
    assert list(domain["s_kv"]) == [1, runtime.MAX_S_KV] == [1, 1 << 20]
    assert [list(item) for item in domain["geometries"]] == [list(runtime.FAMILY_GEOMETRY[f]) for f in runtime.FAMILIES]
    assert f"multiple of {runtime.H_Q_MULTIPLE} from {runtime.MIN_H_Q}" in domain["h_q"]
    assert "stale" in domain["indices"]
    spec = metric.aggregation.spec
    assert {tier: list(groups) for tier, groups in spec["tiers"].items()} == {
        tier: [group for group, member in runtime.GROUP_TIER.items() if member == tier] for tier in runtime.TIERS
    }
    assert f"median of {3} trials" in spec["case_statistic"]
    assert f"{runtime.STREAM_WARM_CALLS} untimed warm-up calls" in spec["stream_protocol"]
    assert "fp8, int8 and fp4 forbidden" in definition["precision"]
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
