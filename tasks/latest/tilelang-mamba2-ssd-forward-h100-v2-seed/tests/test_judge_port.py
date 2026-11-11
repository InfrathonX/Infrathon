from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import pytest


TASK_ROOT = Path(__file__).resolve().parents[1]
JUDGE_ROOT = TASK_ROOT / "images/judge"
sys.path.insert(0, str(JUDGE_ROOT))
SPEC = importlib.util.spec_from_file_location("tilelang_ssd_verify_test", JUDGE_ROOT / "verify_ssd.py")
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
RUNTIME_ROOT = TASK_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))
import mamba2_ssd_runtime as runtime  # noqa: E402

_WORK_SPEC = importlib.util.spec_from_file_location(
    "tilelang_ssd_work_runtime_test", TASK_ROOT / "images/work/benchmark/mamba2_ssd_runtime.py"
)
assert _WORK_SPEC is not None and _WORK_SPEC.loader is not None
work = importlib.util.module_from_spec(_WORK_SPEC)
sys.modules["tilelang_ssd_work_runtime_test"] = work
_WORK_SPEC.loader.exec_module(work)


def test_judge_frozen_contract_matches_trusted_runtime() -> None:
    assert verifier.GEOMETRIES == runtime.GEOMETRIES
    assert verifier.SCORE_FAMILIES == runtime.SCORE_FAMILIES
    assert verifier.TIERS == runtime.TIERS
    assert verifier.FAMILY_WEIGHTS == runtime.FAMILY_WEIGHTS
    assert verifier.FINAL_FAMILY_COUNTS == runtime.FINAL_FAMILY_COUNTS
    assert verifier.EXPECTED_CASE_COUNT == runtime.FINAL_CASE_COUNT
    assert verifier.REAL_TRIALS == runtime.REAL_TRIALS
    assert verifier.MIN_LONG_LENGTH == runtime.MIN_LONG_LENGTH
    assert verifier.MAX_BATCH == runtime.MAX_BATCH
    assert verifier.MAX_TENSOR_ELEMENTS == runtime.MAX_TENSOR_ELEMENTS
    assert verifier.CASE_KEYS == set(runtime.CASE_KEYS)
    assert verifier.OUT_REL_L2_TOL == runtime.OUT_REL_L2_TOL
    assert verifier.OUT_SCALED_ABS_TOL == runtime.OUT_SCALED_ABS_TOL
    assert verifier.OUT_REL_TOL == runtime.OUT_REL_TOL
    assert verifier.REAL_LAYER_REL_L2_TOL == runtime.REAL_LAYER_REL_L2_TOL
    assert verifier.WARMUP_WALL_CAP_SECONDS == runtime.WARMUP_WALL_CAP_SECONDS
    assert verifier.WORKSPACE_MULTIPLE == runtime.WORKSPACE_MULTIPLE
    assert verifier.WORKSPACE_CAP_BYTES == runtime.WORKSPACE_CAP_BYTES
    assert verifier.FINAL_SUITE_ID == runtime.FINAL_SUITE_ID
    assert verifier.MANIFEST_SCHEMA_VERSION == runtime.MANIFEST_SCHEMA_VERSION
    assert verifier.EDGE_CASE_IDS == tuple(case["case_id"] for case in runtime.EDGE_FIXTURES)
    assert verifier.ALLOWED_TORCH_OPS == runtime.ALLOWED_TORCH_OPS
    assert verifier.PUBLIC_SEEDS == frozenset(case["seed"] for case in work.public_suite())


# A synthetic suite that satisfies every family rule: one geometry per family.
SHAPES = {
    "batch": ("mamba2_2p7b", 2, 1000),
    "serving": ("codestral_7b", 1, 2000),
    "long": ("falcon_h1_34b", 1, 60000),
    "real": ("granite_h_micro", 1, 16381),
}


def final_manifest() -> dict[str, Any]:
    cases = []
    index = 0
    for family, count in verifier.FINAL_FAMILY_COUNTS.items():
        geometry, batch, seqlen = SHAPES[family]
        for ordinal in range(count):
            cases.append(
                {
                    "case_id": f"final-{index:03d}-{family}",
                    "geometry": geometry,
                    "family": family,
                    "tier": "real" if family == "real" else ("recipe" if ordinal == 0 else "sweep"),
                    "batch": batch,
                    "seqlen": seqlen if family == "real" else seqlen + ordinal,
                    "regime": "mixed",
                    "seed": 0x1234_5678_9000 + index,
                    "check_trial": 0 if family == "real" else ordinal % 3,
                    "layer": f"granite-h-micro-layer{ordinal:02d}" if family == "real" else None,
                }
            )
            index += 1
    return {"schema_version": 3, "suite_id": verifier.FINAL_SUITE_ID, "cases": cases}


def test_final_manifest_is_accepted() -> None:
    cases = verifier.validate_final_manifest(final_manifest())
    assert len(cases) == verifier.EXPECTED_CASE_COUNT == 138
    runtime.validate_manifest_cases(final_manifest()["cases"])


@pytest.mark.parametrize(
    "mutation",
    ("public_seed", "duplicate_seed", "duplicate_shape", "duplicate_layer", "batch_on_serving",
     "short_long", "missing_case", "extra_field", "wrong_suite", "old_schema", "seed_too_wide",
     "slow_regime", "check_trial_out_of_range", "real_without_layer", "layer_on_synthetic",
     "unknown_geometry", "outside_domain"),
)
def test_final_manifest_mutations_are_rejected(mutation: str) -> None:
    payload = final_manifest()
    cases = payload["cases"]
    serving = next(c for c in cases if c["family"] == "serving")
    long_case = next(c for c in cases if c["family"] == "long")
    real = [c for c in cases if c["family"] == "real"]
    if mutation == "public_seed":
        cases[0]["seed"] = work.public_suite()[0]["seed"]
    elif mutation == "duplicate_seed":
        cases[1]["seed"] = cases[0]["seed"]
    elif mutation == "duplicate_shape":
        cases[1]["seqlen"] = cases[0]["seqlen"]
    elif mutation == "duplicate_layer":
        real[1]["layer"] = real[0]["layer"]
    elif mutation == "batch_on_serving":
        serving["batch"] = 2
    elif mutation == "short_long":
        long_case["seqlen"] = 32768
    elif mutation == "missing_case":
        cases.pop()
    elif mutation == "extra_field":
        cases[0]["slot"] = 1
    elif mutation == "wrong_suite":
        payload["suite_id"] = "tilelang-ssd-final-v2"
    elif mutation == "old_schema":
        payload["schema_version"] = 2
    elif mutation == "seed_too_wide":
        cases[0]["seed"] = 2**63
    elif mutation == "slow_regime":
        cases[0]["regime"] = "slow_decay"
    elif mutation == "check_trial_out_of_range":
        cases[0]["check_trial"] = 3
    elif mutation == "real_without_layer":
        real[0]["layer"] = None
    elif mutation == "layer_on_synthetic":
        cases[0]["layer"] = "granite-h-micro-layer99"
    elif mutation == "unknown_geometry":
        cases[0]["geometry"] = "unknown_model"
    else:
        cases[0]["batch"] = 16
        cases[0]["seqlen"] = 2**31 // (16 * 80 * 64) + 1
    with pytest.raises(RuntimeError):
        verifier.validate_final_manifest(payload)


def clean_worker_result(trials: int = 3) -> dict[str, Any]:
    cases = final_manifest()["cases"]
    rows = []
    family_values: dict[str, list[float]] = {family: [] for family in verifier.SCORE_FAMILIES}
    for index, case in enumerate(cases):
        case_trials = verifier.REAL_TRIALS if case["family"] == "real" else trials
        base = 0.001 + index * 0.00001
        trial_seconds = [base + ordinal * 0.000001 for ordinal in range(case_trials)]
        host_walls = list(trial_seconds)
        event_seconds = [seconds * 0.9 for seconds in trial_seconds]
        median = sorted(trial_seconds)[len(trial_seconds) // 2]
        geometry = verifier.GEOMETRIES[case["geometry"]]
        flops = verifier._effective_flops(case)
        tflops = flops / median / 1.0e12
        family_values[case["family"]].append(tflops)
        rows.append(
            {
                **{key: case[key] for key in ("case_id", "family", "geometry", "tier", "batch", "seqlen")},
                "checked_trial": case["check_trial"],
                "scored": True,
                "heads": geometry["h"],
                "headdim": geometry["p"],
                "dstate": geometry["n"],
                "groups": geometry["g"],
                "regime": "mixed",
                "trials": case_trials,
                "trial_seconds": trial_seconds,
                "median_seconds": median,
                "min_seconds": min(trial_seconds),
                "max_seconds": max(trial_seconds),
                "host_wall_seconds": host_walls,
                "event_seconds": event_seconds,
                "warmup_wall_seconds": 1.0,
                "workspace_bytes": 0,
                "fresh_input_ok": True,
                "fresh_output_ok": True,
                "tilelang_only_ok": True,
                "torch_ops": ["aten::empty.memory_format"],
                "effective_flops": flops,
                "tflops": tflops,
                "rel_l2": verifier.OUT_REL_L2_TOL / 2,
                "max_abs": 1.0,
                "max_abs_scaled": verifier.OUT_SCALED_ABS_TOL / 2,
                "max_rel": 12.0,
                **(
                    {
                        "real_rel_l2": verifier.REAL_LAYER_REL_L2_TOL / 2,
                        "real_max_abs": 1.0,
                        "real_max_abs_scaled": 0.05,
                        "real_max_rel": 30.0,
                        "real_elementwise_violations": 7.0,
                    }
                    if case["family"] == "real"
                    else {}
                ),
            }
        )
    family_tflops = {
        family: verifier.geometric_mean(values) for family, values in family_values.items()
    }
    edge_rows = [
        {
            "case_id": case_id,
            "family": "edge",
            "geometry": "mamba2_2p7b",
            "scored": False,
            "tilelang_only_ok": True,
            "torch_ops": ["aten::empty.memory_format", "aten::view"],
            "rel_l2": verifier.OUT_REL_L2_TOL / 2,
            "max_abs": 1.0,
            "max_abs_scaled": verifier.OUT_SCALED_ABS_TOL / 2,
            "max_rel": 12.0,
        }
        for case_id in verifier.EDGE_CASE_IDS
    ]
    return {
        "suite_id": verifier.FINAL_SUITE_ID,
        "submission_sha256": "a" * 64,
        "entrypoint": verifier.SSD_ENTRYPOINT,
        "trials": trials,
        "real_trials": verifier.REAL_TRIALS,
        "warmups": 1,
        "hardware": {
            "device_name": "NVIDIA H100",
            "capability": "sm_90",
            "total_memory_bytes": 100_000_000_000,
            "tilelang_version": "0.1.13",
        },
        "tolerances": {
            "rel_l2": verifier.OUT_REL_L2_TOL,
            "scaled_abs": verifier.OUT_SCALED_ABS_TOL,
            "rel": verifier.OUT_REL_TOL,
            "real_layer_rel_l2": verifier.REAL_LAYER_REL_L2_TOL,
        },
        "allowed_torch_ops": sorted(verifier.ALLOWED_TORCH_OPS),
        "edge_fixtures": edge_rows,
        "cases": rows,
        "family_tflops": family_tflops,
        "score_tflops": verifier.weighted_family_score(family_tflops),
        "scored_cases": len(rows),
    }


def validate(result: dict[str, Any], trials: int = 3) -> None:
    verifier.validate_runtime_result(
        result,
        final_manifest()["cases"],
        expected_trials=trials,
        expected_warmups=1,
        expected_suite_id=verifier.FINAL_SUITE_ID,
    )


def test_clean_worker_result_passes() -> None:
    result = clean_worker_result()
    score, families, geometries = verifier.validate_runtime_result(
        result,
        final_manifest()["cases"],
        expected_trials=3,
        expected_warmups=1,
        expected_suite_id=verifier.FINAL_SUITE_ID,
    )
    assert score > 0.0
    assert set(families) == set(verifier.SCORE_FAMILIES)
    assert set(geometries) == {geometry for geometry, _b, _l in SHAPES.values()}


@pytest.mark.parametrize(
    "mutation",
    ("tflops", "trials", "real_trials", "checked_trial", "warmup_wall_seconds", "workspace_bytes",
     "workspace_cap", "suite_id", "host_torch_op", "tilelang_only_flag", "edge_host_torch_op",
     "edge_family", "allowlist", "tier", "real_rel_l2", "real_metric_missing"),
)
def test_tampered_worker_result_is_rejected(mutation: str) -> None:
    result = copy.deepcopy(clean_worker_result())
    row = result["cases"][0]
    if mutation == "tflops":
        row["tflops"] *= 2.0
    elif mutation == "trials":
        row["trials"] -= 1
    elif mutation == "real_trials":
        result["real_trials"] = 3
    elif mutation == "checked_trial":
        row["checked_trial"] = (row["checked_trial"] + 1) % 3
    elif mutation == "warmup_wall_seconds":
        row["warmup_wall_seconds"] = verifier.WARMUP_WALL_CAP_SECONDS + 0.001
    elif mutation == "workspace_bytes":
        row["workspace_bytes"] = math.floor(
            verifier.WORKSPACE_MULTIPLE * verifier._io_bytes(final_manifest()["cases"][0])
        ) + 1
    elif mutation == "workspace_cap":
        long_row = next(r for r in result["cases"] if r["family"] == "long")
        long_row["workspace_bytes"] = verifier.WORKSPACE_CAP_BYTES + 1
    elif mutation == "suite_id":
        result["suite_id"] = "tilelang-ssd-final-v2"
    elif mutation == "host_torch_op":
        row["torch_ops"] = ["aten::empty.memory_format", "aten::mul.Tensor"]
    elif mutation == "tilelang_only_flag":
        row["tilelang_only_ok"] = False
    elif mutation == "edge_host_torch_op":
        result["edge_fixtures"][0]["torch_ops"] = ["aten::mm"]
    elif mutation == "edge_family":
        result["edge_fixtures"][0]["family"] = "batch"
    elif mutation == "tier":
        row["tier"] = "real"
    elif mutation == "real_rel_l2":
        next(r for r in result["cases"] if r["family"] == "real")["real_rel_l2"] = (
            verifier.REAL_LAYER_REL_L2_TOL * 1.01
        )
    elif mutation == "real_metric_missing":
        next(r for r in result["cases"] if r["family"] == "real").pop("real_rel_l2")
    else:
        result["allowed_torch_ops"] = sorted(verifier.ALLOWED_TORCH_OPS | {"aten::mm"})
    with pytest.raises(RuntimeError):
        validate(result)


def test_candidate_environment_removes_compiler_ingress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in (
        "TILELANG_DEFAULT_TARGET",
        "TL_ENABLE_FAST_MATH",
        "TVM_LIBRARY_PATH",
        "SKIP_LOADING_TILELANG_SO",
        "CXX",
        "CC",
    ):
        monkeypatch.setenv(key, "attacker-controlled")
    env = verifier._candidate_environment(tmp_path)
    assert all(
        key in {
            "TILELANG_CACHE_DIR",
            "TILELANG_TMP_DIR",
            "TILELANG_DISABLE_CACHE",
            "TILELANG_AUTO_TUNING_DISABLE_CACHE",
            "TILELANG_KERNEL_CACHE_USE_LIB_STAMP",
            "TILELANG_CLEANUP_TEMP_FILES",
        }
        for key in env
        if key.startswith("TILELANG_")
    )
    assert not any(key.startswith(("TL_", "TVM_")) for key in env)
    assert "SKIP_LOADING_TILELANG_SO" not in env
    assert "CXX" not in env and "CC" not in env


def test_adapter_counts_are_pinned_to_the_trusted_runtime() -> None:
    """The adapter's manifest-identity constants must track the runtime.

    These were the one place where a stale number turned a perfect Judge run
    into ten ERROR gates: the verifier reported 14 edge fixtures while the
    adapter still expected 13, so `manifest_identity_ok` was false and every
    gate was rendered as untrusted rather than as passed.
    """
    spec = importlib.util.spec_from_file_location(
        "tilelang_ssd_run_judge_test", JUDGE_ROOT / "run_judge.py"
    )
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    assert adapter.EXPECTED_EDGE_CASE_COUNT == float(len(runtime.EDGE_FIXTURES))
    assert adapter.EXPECTED_CASE_COUNT == float(runtime.FINAL_CASE_COUNT)
    assert adapter.FAMILY_KEYS == tuple(f"{name}_tflops" for name in runtime.SCORE_FAMILIES)
    assert adapter.FAMILY_WEIGHTS == tuple(runtime.FAMILY_WEIGHTS[name] for name in runtime.SCORE_FAMILIES)
    assert adapter.PROFILE_IDENTITIES[2.0] == ("final", runtime.FINAL_SUITE_ID)
    assert "tilelang_only_ok" in {name for name, _label in adapter.GATES}
