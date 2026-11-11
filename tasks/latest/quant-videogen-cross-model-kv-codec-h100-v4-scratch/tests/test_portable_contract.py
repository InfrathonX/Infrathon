"""The portable contract freezes the TaskCore byte-for-byte.

Every digest, path set, case table and count declared by ``portable/*.json``
and the task JSON is recomputed here from the shipped sources (the release,
the paper protocol, the hidden evaluator and its shard planner, the source
policy, the public tests, Dockerfiles, the staging recipe), never trusted from
the declaration.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from conftest import HARDWARE_TYPES, ROOT, quality_file, release_file, task_file
from sforge.harness.portable_spec import parse_portable_spec, verify_sha256_file

PORTABLE = ROOT / "portable"
TASK_PATH = ROOT / "tasks/quant_videogen_cross_model_kv_codec_h100_scratch_v4.json"
PROMPTS_PATH = ROOT / "judge/private/prompts/MovieGenVideoBench_extended.txt"
SUITE_ID = "quant-videogen-kv-codec-v4"
PROFILE_ID = "cuda-sm80-79gib-sharded-judge"
METRIC_ID = "primary"
MODELS = ("longcat", "hy_worldplay_8b")
LONGCAT_RECORDS = (12, 185, 320, 593, 893, 894)
LONGCAT_DRAW_SEED = "qvg-v4-longcat-20261006"
HY_IDS = (*(f"hy-{n:02d}" for n in range(1, 11)), "hy-qvg")
VBENCH = ("aesthetic_quality", "background_consistency", "image_quality", "subject_consistency")
PAPER_POSE = "w-8,s-8,a-8,d-8,up-8,down-8,w-8,s-8,a-7"
V3_LONGCAT_INIT_SHA256 = "7108816ae0e7db920ad7bed25550644c9d3f3e206fcb2d200fe0793189c3dc15"
V3_HY_IMAGE_SHA256 = "c4a7f5418f852ffa7f98ea2a7e18f9123b26b62a495aeb65b75e04bd37ae1c9b"
# v3's hy-qvg BF16 golden; v4 keeps that case, so its golden stays bound to it.
V3_HY_GOLDEN_SHA256 = "a8ee4657df167b536ef6ecc6d6f1e9d03daab6e7108aeadbff7edd7aa80a3398"
# §4.1 of the v4 iteration spec (docs/tasks/...-v4-iteration-prompt.md).
SPEC_JUDGE = {
    "max_gpu_count": 8, "min_gpu_count": 1, "slot_uids": list(range(23001, 23009)), "master_port_base": 29600,
    "threads_per_slot": 4, "start_stagger_seconds": 20, "case_overhead_seconds": 120,
    # 79 GiB (the spec's 99e9 bytes covered a BF16 rollout, which only the frozen goldens need).
    "controller_overhead_seconds": 300, "min_device_memory_bytes": 79 * 1024**3,
}
SPEC_BUDGETS = {
    "longcat": {"fps": 15, "stage_seconds": 4500, "fidelity_metric_seconds": 600, "vbench_metric_seconds": 1200,
                "profile_key": "persistent_kv", "frames_file": "final.frames.pt"},
    "hy_worldplay_8b": {"fps": 24, "stage_seconds": 1500, "fidelity_metric_seconds": 600, "vbench_metric_seconds": 900,
                        "profile_key": "persistent_vision_kv", "frames_file": "gen.frames.pt"},
}
# Delivery inputs the Dockerfiles do not COPY but the build manifest hashes.
BUILD_EXTRAS = {
    ".dockerignore", "images/work/Dockerfile", "images/judge/Dockerfile", "images/work/base/Dockerfile",
    # The host-built holdout layer and the release config it receives through its own context.
    "images/judge/Dockerfile.private", "configs/h100_scratch_v4_release.json",
    "portable/REBUILD_PROMPT.md", "scripts/build_images.sh", "scripts/codex_search.py", "scripts/eval_archive.sh",
    "scripts/host_proxy_bridge.py", "scripts/lib.sh", "scripts/restricted_connect_proxy.py",
    "scripts/runtime_identity.py", "scripts/sforge_runtime_entry.py", "scripts/validate_layout.py",
    "scripts/verify_models.py",
}


def _load(name: str, relative: str):
    path = ROOT / relative
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
def policy():
    return _load("portable_test_policy", "policy/codec_source_policy.py")


@pytest.fixture(scope="module")
def layout():
    return _load("portable_test_validate_layout", "scripts/validate_layout.py")


@pytest.fixture(scope="module")
def refresh():
    return _load("portable_test_refresh", "scripts/refresh_portable_manifests.py")


@pytest.fixture(scope="module")
def shards():
    return _load("portable_test_judge_shards", "judge/judge_shards.py")


@pytest.fixture(scope="module")
def evaluate_hidden():
    return _load("portable_test_evaluate_hidden", "judge/evaluate_hidden.py")


@pytest.fixture(scope="module")
def staging():
    return _load("portable_test_stage_eval_inputs", "scripts/stage_eval_inputs.py")


@pytest.fixture(scope="module")
def release() -> dict:
    return json.loads(release_file().read_text(encoding="utf-8"))


def _manifest(name: str) -> dict:
    return json.loads((PORTABLE / name).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files_under(relative: str) -> set[str]:
    return {f.relative_to(ROOT).as_posix() for f in (ROOT / relative).rglob("*") if f.is_file() and "__pycache__" not in f.parts and f.suffix not in (".pyc", ".pyo")}


def _dockerfile_copies(*dockerfiles: str) -> list[tuple[list[str], str]]:
    copies = []
    for name in dockerfiles:
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            if line.startswith("COPY "):
                parts = [part for part in line.split()[1:] if not part.startswith("--") and part != "\\"]
                copies.append((parts[:-1] if len(parts) > 1 else parts, parts[-1]))
    return copies


def _controller_bound(release: dict, gpu_count: int) -> int:
    """Independent restatement of the spec: list-schedule every case at its full budget."""
    judge = release["judge"]
    units = sorted((release["models"][c["model"]]["stage_seconds"]
                    + max(release["models"][c["model"]]["fidelity_metric_seconds"], release["models"][c["model"]]["vbench_metric_seconds"])
                    + judge["case_overhead_seconds"] for c in release["cases"]), reverse=True)
    free = [i * judge["start_stagger_seconds"] for i in range(gpu_count)]
    for unit in units:
        slot = free.index(min(free))
        free[slot] += unit
    return max(free) + judge["controller_overhead_seconds"]


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
    for filename, binding in (("oracle-protocol.json", suite.oracle), ("selection-protocol.json", suite.selection),
                              ("software-abi.json", profile.software_abi), ("calibration-protocol.json", profile.calibration_protocol)):
        assert _manifest(filename)["protocol_id"] == binding.protocol_id


def test_resource_manifest_hashes_every_real_scientific_resource() -> None:
    manifest = _manifest("resource-manifest.json")
    expected = {
        "images/work/AGENT_TASK.md", "images/work/public_tests/run_public_tests.py", "images/work/public_tests/run_self_forcing_paper.py",
        "assets/public_prompts.txt", "judge/run_judge.py", "judge/evaluate_hidden.py", "judge/judge_shards.py",
        "judge/run_longcat.py", "judge/run_hy_worldplay_8b.py",
        "configs/h100_scratch_v4_release.json", "configs/paper_protocol.json", "metrics/compare_frames.py", "metrics/run_vbench.py",
        "policy/codec_source_policy.py", "locks/model-lock.json", "upstream/experiments/Self-Forcing/pipeline/causal_inference.py",
        *_files_under("runtime/quant_videogen"), *_files_under("upstream/experiments/LongCat"), *_files_under("upstream/experiments/HY-WorldPlay"),
    }
    # The v3 HY image is the hy-qvg case image; it stays hashed while it ships in-tree.
    if (ROOT / "assets/hyworld.png").is_file():
        expected.add("assets/hyworld.png")
    assert {resource["path"] for resource in manifest["resources"]} == expected
    for resource in manifest["resources"]:
        verify_sha256_file(task_file(resource["path"]), resource["sha256"])
    assert not any(p.startswith("oracle/") for p in expected)


def test_build_manifest_covers_exact_portable_delivery_inputs(raw_task) -> None:
    path = PORTABLE / "build-resource-manifest.json"
    declared = raw_task["delivery"]["build_resource_manifest_sha256"]
    assert verify_sha256_file(path, declared) == declared
    manifest = json.loads(path.read_text(encoding="utf-8"))
    declared_paths = {resource["path"] for resource in manifest["resources"]}
    # Exactly every path the three Dockerfiles copy from the task-root context
    # (the base is built from this repository too), plus the delivery inputs
    # that are not copied: Dockerfiles, adapter, launchers, proxy tooling.
    copied: set[str] = set()
    for sources, _ in _dockerfile_copies("images/work/base/Dockerfile", "images/work/Dockerfile", "images/judge/Dockerfile"):
        for source in sources:
            source = source.rstrip("/")
            copied |= _files_under(source) if (ROOT / source).is_dir() else {source}
    assert declared_paths == copied | BUILD_EXTRAS, (sorted(copied | BUILD_EXTRAS - declared_paths), sorted(declared_paths - copied - BUILD_EXTRAS))
    assert not any(p.startswith("oracle/") for p in declared_paths)
    for resource in manifest["resources"]:
        verify_sha256_file(task_file(resource["path"]), resource["sha256"])


def test_host_private_material_cannot_enter_the_work_image() -> None:
    work = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8").casefold()
    assert "portable" not in work and "final-suite-manifest" not in work
    for private in ("/models/hidden", "/eval/hidden", "/opt/qvg-judge", "/opt/qvg-reference"):
        assert f"test ! -e {private}" in work
    judge = (ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    assert judge.startswith("ARG WORK_IMAGE=")
    controller = {source for sources, target in _dockerfile_copies("images/judge/Dockerfile") if target.startswith("/opt/qvg-judge/") for source in sources}
    assert {"judge/evaluate_hidden.py", "judge/run_judge.py", "judge/judge_shards.py"} <= controller
    # Case inputs live in the Judge-only hidden tree, never in an image.
    assert "hyworld.png" not in judge and "assets/" not in judge


def test_public_manifest_lists_the_disclosed_diagnostics_only(portable) -> None:
    manifest = _manifest("public-suite-manifest.json")
    cases = manifest["cases"]
    assert len(cases) == portable.evaluation_suites[SUITE_ID].roles["public"].case_count == 3
    assert all(case["role"] == "diagnostic" for case in cases)
    assert all(Fraction(case["weight"]["numerator"], case["weight"]["denominator"]) == 0 for case in cases)
    prefixes = sorted(case["prefix"] for case in cases if "prefix" in case)
    assert prefixes == [350, 700]
    runner = (ROOT / "images/work/public_tests/run_self_forcing_paper.py").read_text(encoding="utf-8")
    assert 'raise SystemExit("paper public prefixes are exactly 350 and 700")' in runner
    prompts = [line for line in (ROOT / "assets/public_prompts.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    for case in cases:
        if "prompt_count" in case:
            assert case["prompt_count"] == len(prompts) == 2
    assert manifest["public_prompts_sha256"] == hashlib.sha256((ROOT / "assets/public_prompts.txt").read_bytes()).hexdigest()
    assert manifest["case_counts"] == {"total": 3, "scored_performance": 0, "zero_weight_hard_gate": 0, "diagnostic": 3}


def test_release_is_the_upstream_17_case_suite(release) -> None:
    assert release["protocol_schema_version"] == 2 and release["status"] in ("calibrating", "calibrated")
    assert release["judge"] == SPEC_JUDGE
    for name, expected in SPEC_BUDGETS.items():
        assert {key: release["models"][name][key] for key in expected} == expected
    prompts = PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
    assert release["prompt_file"] == "/eval/hidden/prompts/MovieGenVideoBench_extended.txt"
    assert release["prompt_file_sha256"] == _sha(PROMPTS_PATH) and len(prompts) == 1003
    ids = [case["case_id"] for case in release["cases"]]
    assert ids == [f"lc-{n:03d}" for n in LONGCAT_RECORDS] + list(HY_IDS)
    longcat = [case for case in release["cases"] if case["model"] == "longcat"]
    hy = [case for case in release["cases"] if case["model"] == "hy_worldplay_8b"]
    # Record 12 is v3's case; the other five are the documented seeded draw
    # from every record except the two public ones (1, 6) and 12.
    pool = [index for index in range(1, len(prompts) + 1) if index not in (1, 6, 12)]
    assert sorted(random.Random(LONGCAT_DRAW_SEED).sample(pool, 5)) == list(LONGCAT_RECORDS[1:])
    for case in longcat:
        index = case["prompt_index"]
        assert case["prompt"] == prompts[index - 1]
        assert case["segments"] == 10 and case["frame_count"] == 273 and case["fidelity_frames"] == [73, 93]
        assert case["init_video"] == f"/eval/hidden/longcat/{index:03d}/init.mp4"
        assert case["bf16_frames"] == f"/eval/hidden/longcat/{index:03d}/bf16-10seg.frames.pt"
    # lc-012 continues from v3's frozen init; the other inits are pinned once generated.
    assert longcat[0]["init_video_sha256"] == V3_LONGCAT_INIT_SHA256
    lock = json.loads((ROOT / "locks/model-lock.json").read_text(encoding="utf-8"))
    locked = {entry["path"]: entry["sha256"] for entry in lock["git_bundles"]["hy_worldplay_official_assets"]["files"]}
    sources = [f"assets/img/{n}.{'jpeg' if n in (5, 6) else 'png'}" for n in range(1, 11)] + ["assets/img/test.png"]
    for case, source in zip(hy, sources):
        name = case["case_id"].removeprefix("hy-")
        assert case["image_source"] == f"HY-WorldPlay {source} @1588e133"
        assert case["image"] == f"/eval/hidden/hy/{name}/image.{source.rsplit('.', 1)[1]}"
        assert case["image_sha256"] == locked[source]
        assert case["video_length"] == case["frame_count"] == 285 and case["fidelity_frames"] == [29, 45]
        assert case["pose"] == PAPER_POSE and case["bf16_frames"] == f"/eval/hidden/hy/{name}/bf16-285.frames.pt"
    assert len({case["prompt"] for case in hy}) == len(hy)
    assert hy[-1]["image_sha256"] == V3_HY_IMAGE_SHA256
    # hy-qvg carries v3's golden: a v4 regeneration must reproduce it bit for bit.
    assert hy[-1]["case_id"] == "hy-qvg" and hy[-1]["bf16_frames_sha256"] == V3_HY_GOLDEN_SHA256
    if (ROOT / "assets/hyworld.png").is_file():
        assert _sha(ROOT / "assets/hyworld.png") == V3_HY_IMAGE_SHA256


def test_release_validator_accepts_the_release_and_rejects_drift(release, layout) -> None:
    assert layout.release_errors(release, ROOT) == []

    def errors(mutate) -> list[str]:
        changed = copy.deepcopy(release)
        mutate(changed)
        return layout.release_errors(changed, ROOT)

    if release["status"] == "calibrating":
        # Null byte counts and placeholder gates (and, before the goldens are
        # bound, null digests and zero BF16 VBench) are admissible only while
        # calibrating.
        found = errors(lambda r: r.update(status="calibrated"))
        assert any("bf16_kv_bytes is null in a calibrated release" in e for e in found)
        assert any("placeholder gates" in e for e in found)
        unbound = errors(lambda r: [r.update(status="calibrated")] + [
            case.update(bf16_frames_sha256=None, bf16_vbench={k: 0 for k in case["bf16_vbench"]})
            for case in r["cases"]])
        assert any("bf16_frames_sha256 is null in a calibrated release" in e for e in unbound)
        assert any("bf16_vbench still holds placeholders" in e for e in unbound)
    assert errors(lambda r: r["cases"][1].update(prompt="x"))
    assert errors(lambda r: r["cases"][0].update(prompt_index=1, case_id="lc-001"))
    assert errors(lambda r: r["cases"][-1].update(image_sha256="0" * 64))
    assert errors(lambda r: r["cases"][-1].update(pose="w-8"))
    assert errors(lambda r: r["cases"][3].update(bf16_frames=r["cases"][2]["bf16_frames"]))
    assert errors(lambda r: r["cases"][0].update(bf16_frames_sha256="not-a-digest"))
    assert errors(lambda r: r["cases"].pop())
    assert errors(lambda r: r["judge"].update(slot_uids=[23001] * 8))
    assert errors(lambda r: r["models"]["longcat"]["gates"]["case_floor"].update(psnr_min=50))
    assert errors(lambda r: r.update(protocol_schema_version=1))


def test_interactive_and_final_manifests_mirror_the_release(portable, release, raw_task) -> None:
    interactive = _manifest("interactive-suite-manifest.json")
    final = _manifest("final-suite-manifest.json")
    bound = _controller_bound(release, release["judge"]["max_gpu_count"])
    assert bound == 8720  # 6 LongCat + 11 HY cases on 8 devices
    assert raw_task["judge"]["eval_timeout"] >= bound
    for role, manifest in (("interactive", interactive), ("final", final)):
        cases = manifest["cases"]
        assert len(cases) == portable.evaluation_suites[SUITE_ID].roles[role].case_count == 17
        assert manifest["suite_id"] == "quant-videogen-hidden-v4" and manifest["role"] == role
        assert manifest["case_counts"] == {"scored_performance": 17, "total": 17, "zero_weight_hard_gate": 0}
        assert manifest["counts_per_workload"] == {"longcat": 6, "hy_worldplay_8b": 11}
        assert manifest["release_sha256"] == _sha(release_file()) and manifest["release_status"] == release["status"]
        assert [case["case_id"] for case in cases] == [case["case_id"] for case in release["cases"]]
        for name in MODELS:
            weights = [Fraction(c["weight"]["numerator"], c["weight"]["denominator"]) for c in cases if c["workload"] == name]
            assert sum(weights) == 1
            assert Fraction(manifest["workload_exponents"][name]["numerator"], manifest["workload_exponents"][name]["denominator"]) == Fraction(1, 2)
            gates = manifest["workload_gates"][name]
            for kind in ("mean", "case_floor"):
                declared, frozen = gates[kind], release["models"][name]["gates"][kind]
                assert {k: float(declared[k]) for k in ("psnr_min", "ssim_min", "lpips_max")} == {k: frozen[k] for k in ("psnr_min", "ssim_min", "lpips_max")}
                assert {d: float(v) for d, v in declared["vbench_allowed_drop"].items()} == frozen["vbench_allowed_drop"]
            assert gates["wall_seconds_max"] == release["models"][name]["stage_seconds"]
        for case, frozen in zip(cases, release["cases"]):
            assert case["workload"] == frozen["model"] and case["role"] == "performance" and case["seed"] == frozen["seed"]
            assert case["goldens"] == {"bf16_frames_sha256": frozen["bf16_frames_sha256"], "bf16_kv_bytes": frozen["bf16_kv_bytes"], "bf16_vbench": frozen["bf16_vbench"]}
            assert case["fidelity_frames"] == frozen["fidelity_frames"] and case["frame_count"] == frozen["frame_count"]
            if frozen["model"] == "longcat":
                assert (case["prompt_index"], case["segments"], case["init_video_sha256"]) == (frozen["prompt_index"], frozen["segments"], frozen["init_video_sha256"])
            else:
                assert (case["image_sha256"], case["video_length"], case["pose"]) == (frozen["image_sha256"], frozen["video_length"], frozen["pose"])
        timeouts = manifest["suite_gate_timeouts"]
        assert timeouts["controller_seconds"] == bound and timeouts["max_gpu_count"] == release["judge"]["max_gpu_count"]
        for name in MODELS:
            assert timeouts["workloads"][name] == {k: release["models"][name][k] for k in ("stage_seconds", "fidelity_metric_seconds", "vbench_metric_seconds")}
        assert {"release_calibrated", "own_device_only", "frozen_bf16_kv_bytes", "case_wall_time", "hardware_envelope"} <= set(manifest["run_gates"])
    assert final["orchestration_timeouts"]["standalone_final_evaluation_seconds"] == raw_task["judge"]["eval_timeout"]
    # The Judge has one hidden suite: no round-id routing exists.
    for name in ("judge/run_judge.py", "judge/evaluate_hidden.py"):
        assert "round_id" not in (ROOT / name).read_text(encoding="utf-8")


def test_controller_bound_is_the_shard_planner_makespan(release, refresh, shards, evaluate_hidden) -> None:
    judge = release["judge"]
    ids = [case["case_id"] for case in release["cases"]]
    costs = {case["case_id"]: release["models"][case["model"]]["stage_seconds"]
             + max(release["models"][case["model"]]["fidelity_metric_seconds"], release["models"][case["model"]]["vbench_metric_seconds"])
             + judge["case_overhead_seconds"] for case in release["cases"]}
    for gpus in range(judge["min_gpu_count"], judge["max_gpu_count"] + 1):
        initial = [index * judge["start_stagger_seconds"] for index in range(gpus)]
        plan = shards.plan_shards(ids, costs, gpus, initial)
        planned = shards.plan_makespan(plan, costs, initial) + judge["controller_overhead_seconds"]
        assert refresh.controller_bound_seconds(release, gpus) == planned == _controller_bound(release, gpus)
        # The manifests' controller_seconds is the controller's own budget.
        assert evaluate_hidden.controller_budget_seconds(release, gpus) == planned


def test_oracle_and_calibration_match_the_release(release) -> None:
    oracle = _manifest("oracle-protocol.json")
    policy = oracle["policy"]
    assert policy["paper_protocol_sha256"] == release["paper_protocol_sha256"] == _sha(ROOT / "configs/paper_protocol.json")
    assert policy["metric_implementation_sha256"] == release["metric_implementation_sha256"]
    assert _sha(ROOT / "metrics/compare_frames.py") == release["metric_implementation_sha256"]["fidelity"]
    assert _sha(ROOT / "metrics/run_vbench.py") == release["metric_implementation_sha256"]["vbench"]
    assert policy["prompt_file_sha256"] == release["prompt_file_sha256"] and policy["release_status"] == release["status"]
    for name, model in release["models"].items():
        for kind, frozen in model["gates"].items():
            declared = policy["gates"][name][kind]
            assert float(declared["psnr_min"]) == frozen["psnr_min"] and float(declared["lpips_max"]) == frozen["lpips_max"]
    assert policy["execution_budget"]["controller_seconds"] == _controller_bound(release, release["judge"]["max_gpu_count"])
    assert policy["judge"]["slot_uids"] == release["judge"]["slot_uids"]
    assert "wall_seconds <= the workload's stage_seconds" in oracle["reference"]["latency"]
    assert "summed over every call" in oracle["reference"]["compression"]
    calibration = _manifest("calibration-protocol.json")
    assert "no timing or throughput term" in calibration["score_measurement"]
    assert {"mean_gates", "case_floor_gates", "controls", "runs", "uncalibrated_release"} <= set(calibration["gate_calibration"])
    paper = json.loads((ROOT / "configs/paper_protocol.json").read_text(encoding="utf-8"))["mlsysforge"]
    assert paper["fidelity_windows"]["longcat"] == [73, 93] and paper["fidelity_windows"]["hy_worldplay_8b"] == [29, 45]
    assert paper["hidden_suite"]["longcat"]["prompt_records_one_based"] == list(LONGCAT_RECORDS)
    assert paper["hidden_suite"]["hy_worldplay_8b"]["cases"] == len(HY_IDS) and paper["hidden_suite"]["hy_worldplay_8b"]["pose"] == PAPER_POSE
    assert "accumulated over every compression call" in paper["kv_byte_accounting"]["scope"]
    selection = _manifest("selection-protocol.json")
    assert selection["final"]["case_count"] == selection["interactive"]["case_count"] == 17
    assert selection["cases_per_workload"] == {"longcat": 6, "hy_worldplay_8b": 11}


def test_the_declared_quality_term_is_the_controllers(portable, evaluate_hidden) -> None:
    # Portable specs hold decimal strings; each must parse to the controller's exact float.
    quality = portable.metrics[METRIC_ID].aggregation.spec["quality"]
    ceiling = float(quality["psnr_ceiling_db"])
    assert ceiling == evaluate_hidden.PSNR_CEILING_DB
    assert float(quality["fidelity_weight"]) == evaluate_hidden.FIDELITY_WEIGHT
    assert float(quality["video_weight"]) == evaluate_hidden.VIDEO_WEIGHT
    assert evaluate_hidden.FIDELITY_WEIGHT + evaluate_hidden.VIDEO_WEIGHT == 1.0
    # The zero point is device-measured per class, never a portable constant or release data.
    assert "hw/<type>/judge/h100_scratch_v4_quality.json" not in quality["reference"]  # no hardware names in task text
    assert "each supported device class" in quality["reference"]
    assert not hasattr(evaluate_hidden, "QUALITY_REFERENCE")


def test_the_shared_release_carries_no_quality_reference() -> None:
    release = json.loads(release_file().read_text(encoding="utf-8"))
    assert release["status"] == "calibrated"
    assert all("quality_reference" not in model for model in release["models"].values())
    assert "per_device_class" in release["oracle_calibration"]


@pytest.mark.parametrize("hardware", HARDWARE_TYPES)
def test_each_device_class_freezes_its_quality_reference(evaluate_hidden, hardware: str) -> None:
    path = quality_file(hardware)
    quality = json.loads(path.read_text(encoding="utf-8"))
    assert quality["hardware"] == hardware
    release = evaluate_hidden.load_release(release_file(), quality_path=path)
    evaluate_hidden.require_quality(release)
    for name in MODELS:
        reference = release["models"][name]["quality_reference"]
        assert set(reference) == {"psnr", "ssim", "lpips", "vbench"} and set(reference["vbench"]) == set(VBENCH)
    calibration = quality["oracle_calibration"]
    assert "Quant-VideoGen @ 0601468f" in calibration["oracle"] and calibration["quality_reference_source"]
    low, high = calibration["score_band"]
    assert low < calibration["raw_score"] < high
    # Compression is deterministic: the same on every device class.
    assert calibration["compression_ratio"] == {"longcat": 6.879606055368714, "hy_worldplay_8b": 6.754934557313714}


def test_software_abi_matches_the_images_and_policy(policy) -> None:
    abi = _manifest("software-abi.json")
    assert abi["allowed_import_roots"] == sorted(policy.ALLOWED_IMPORT_ROOTS)
    assert abi["forbidden_call_names"] == sorted(policy.FORBIDDEN_CALL_NAMES)
    assert abi["max_total_bytes"] == policy.MAX_TOTAL_BYTES and abi["policy_schema_version"] == policy.POLICY_SCHEMA_VERSION
    work = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    assert "ARG BASE_WORK_IMAGE=quant_videogen_h100_scratch_v4.base.quant_videogen_cross_model_kv_codec_h100_scratch_v4:local" in work
    assert abi["model_binding"]["adapter"] == "scripts/sforge_runtime_entry.py"


def test_work_base_is_buildable_from_this_repository() -> None:
    """The Work base must not depend on an out-of-tree intermediate image."""
    base = (ROOT / "images/work/base/Dockerfile").read_text(encoding="utf-8")
    pins = [line for line in base.splitlines() if line.startswith("ARG BASE_IMAGE=")]
    assert len(pins) == 1 and "@sha256:" in pins[0], pins
    assert "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:" in pins[0]
    assert "FROM ${BASE_IMAGE}" in base
    # Every dependency is pinned; nothing resolves at build time.
    lock = (ROOT / "images/work/base/requirements.lock").read_text(encoding="utf-8")
    entries = [line.strip() for line in lock.splitlines() if line.strip() and not line.startswith("#")]
    assert entries
    for entry in entries:
        assert ("==" in entry) or (" @ " in entry and "#sha256=" in entry), entry
    assert "--no-deps" in base and "requirements.lock" in base
    # The base is agent-free and installs no reference codec or starter tree.
    instructions = [line for line in base.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert any("! command -v codex" in line for line in instructions)
    for forbidden in ("nodejs.org", "@openai/codex", "npm install", "/workspace/starter"):
        assert not any(forbidden in line for line in instructions), forbidden
    assert not any(
        line.startswith("COPY ") and ("qvg-reference" in line or "upstream/quant_videogen" in line)
        for line in instructions
    )
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert build.index("images/work/base/Dockerfile") < build.index("images/work/Dockerfile") < build.index("images/judge/Dockerfile")


def test_portable_layers_preserve_metric_and_use_capabilities(raw_task, portable, release) -> None:
    metric = portable.metrics[METRIC_ID]
    assert metric.direction == "maximize" == raw_task["judge"]["score_direction"]
    assert metric.unit == "dimensionless"
    assert metric.comparability == "portable"
    assert metric.baseline_policy == "none"
    assert list(metric.aggregation.spec["workloads"]) == list(MODELS)
    assert "sum of stored_kv_bytes" in metric.aggregation.spec["formula"]
    assert "(1 + longcat_q)" in metric.aggregation.spec["formula"]
    profile = portable.execution_profiles[PROFILE_ID]
    assert profile.vendor == "nvidia" and profile.count == 1
    assert "sharded-judge-1-to-8-gpus" in profile.features
    assert profile.min_memory_bytes == release["judge"]["min_device_memory_bytes"]
    assert profile.min_compute_capability == "8.0"
    definition = portable.task_core.definition
    assert list(definition["workloads"]) == list(MODELS) and definition["hidden_workload_count"] == 2
    assert definition["hidden_case_count"] == len(release["cases"]) == 17
    assert definition["hidden_cases_per_workload"] == {"longcat": 6, "hy_worldplay_8b": 11}
    assert portable.defaults.evaluation_suite_id == SUITE_ID and portable.defaults.execution_profile_id == PROFILE_ID and portable.defaults.metric_id == METRIC_ID
    portable_text = json.dumps(raw_task["portable"], sort_keys=True)
    assert "h100" not in portable_text.casefold() and "gpu-" not in portable_text.casefold()
    assert "h100" not in raw_task["name"].casefold() and raw_task["name"].endswith(" v4")
    assert raw_task["cwd"] == "/workspace" and raw_task["submit_paths"] == ["submission"]
    # Separate Work and Judge allocations (owner decision 2026-10-06).
    assert (raw_task["work"]["cpu_limit"], raw_task["work"]["mem_limit"]) == (8, "64g")
    assert (raw_task["judge"]["cpu_limit"], raw_task["judge"]["mem_limit"]) == (32, "448g")
    query = raw_task["work"]["agent_query"].casefold()
    assert "share the single" not in query and "prepare_submit" not in query


def test_staging_plan_builds_every_hidden_input_on_its_own_device(release, staging, tmp_path) -> None:
    hidden = tmp_path / "hidden"
    # Plan as before the goldens were bound: generated digests cleared, the
    # v3-carried lc-012 init and hy-qvg golden kept release-bound.
    release = copy.deepcopy(release)
    for case in release["cases"]:
        if case["case_id"] != "hy-qvg":
            case["bf16_frames_sha256"] = None
        if case["model"] == "longcat" and case["case_id"] != "lc-012":
            case["init_video_sha256"] = None
    # lc-012's init is release-bound: never regenerated, only reused.
    with pytest.raises(SystemExit, match="release-bound"):
        staging.plan_jobs(release, hidden, steps={"inits", "goldens"}, cases=None)
    jobs = staging.plan_jobs(release, hidden, steps={"inits", "goldens"}, cases=None,
                             reused=frozenset({"longcat/012/init.mp4"}))
    by_case = {case["case_id"]: case for case in release["cases"]}
    assert sorted(job.case_id for job in jobs) == sorted(by_case)
    assert [job.model for job in jobs[:6]] == ["longcat"] * 6  # heaviest first
    runs = [run for job in jobs for run in job.runs]
    assert len(runs) == 17 + 5
    ports = [run.master_port for run in runs]
    assert len(set(ports)) == len(ports) and min(ports) == release["judge"]["master_port_base"]
    for job in jobs:
        case = by_case[job.case_id]
        for run in job.runs:
            command = run.command
            assert command[command.index("--master-port") + 1] == str(run.master_port)
            assert command[command.index("--device-index") + 1] == "0"
            assert command[command.index("--mode") + 1] == "bf16"
            assert run.env["OMP_NUM_THREADS"] == str(release["judge"]["threads_per_slot"])
            if case["model"] == "longcat":
                assert run.env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
                assert command[command.index("--prompt-index") + 1] == str(case["prompt_index"])
            else:
                assert "PYTORCH_CUDA_ALLOC_CONF" not in run.env
                assert command[command.index("--pose") + 1] == case["pose"]
                assert command[command.index("--image-path") + 1] == case["image"]
                assert command[command.index("--prompt") + 1] == case["prompt"]
        golden = job.runs[-1]
        assert "/eval/hidden/" + golden.output.target == case["bf16_frames"]
        assert golden.output.pin == case["bf16_frames_sha256"]
        if case["model"] == "longcat":
            assert golden.command[golden.command.index("--segments") + 1] == str(case["segments"])
            assert golden.command[golden.command.index("--init-video") + 1] == case["init_video"]
    assert staging.LOCK_PREFIXES == ("sforge", "mlsysforge")
    # Release-bound files are never overwritten and must match their pin.
    source = tmp_path / "frames.pt"
    source.write_bytes(b"frames")
    digest = hashlib.sha256(b"frames").hexdigest()
    with pytest.raises(staging.StageError, match="pins"):
        staging.install(source, hidden, "hy/01/bf16-285.frames.pt", "0" * 64)
    assert staging.install(source, hidden, "hy/01/bf16-285.frames.pt", digest) == digest
    with pytest.raises(staging.StageError, match="refusing to overwrite"):
        staging.install(source, hidden, "hy/01/bf16-285.frames.pt", None)
    # A run killed between the chmod and the rename left a read-only .partial;
    # the next run resumes over it instead of failing on the stale file.
    stale = hidden / "hy/02/bf16-285.frames.pt.partial"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"half")
    stale.chmod(0o444)
    assert staging.install(source, hidden, "hy/02/bf16-285.frames.pt", None) == digest
    assert not stale.exists() and (hidden / "hy/02/bf16-285.frames.pt").read_bytes() == b"frames"
    # A pin hy-qvg's regeneration must reproduce.
    assert next(run.output.pin for job in staging.plan_jobs(release, tmp_path / "empty", steps={"goldens"},
                                                            cases={"hy-qvg"}) for run in job.runs) == V3_HY_GOLDEN_SHA256
    pinned = copy.deepcopy(release)
    pinned["cases"][6]["bf16_frames_sha256"] = "1" * 64
    assert staging.audit_existing(pinned, hidden)
    assert not staging.plan_jobs(release, hidden, steps={"goldens"}, cases={"hy-01"})
    # The dry run touches no GPU and no Docker.
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/stage_eval_inputs.py"), "--root", str(tmp_path / "root"),
                             "--dry-run", *[arg for index in ("012", "185", "320", "593", "893", "894")
                                            for arg in ("--reuse", f"longcat/{index}/init.mp4={source}")]],
                            capture_output=True, text=True, check=False,
                            env={"PATH": "/nonexistent", **{key: os.environ[key] for key in ("HOME", "XDG_CACHE_HOME", "INFRATHON_ASSETS_ROOT") if key in os.environ}})
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["jobs"]) == 17


def test_layout_validator_passes() -> None:
    release_file()
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/validate_layout.py")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_refresh_tool_reports_no_drift() -> None:
    release_file()
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/refresh_portable_manifests.py"), "--check"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
