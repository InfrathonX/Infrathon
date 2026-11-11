"""The frozen contract: geometry, ABI, tolerances, budgets.

These are the numbers a released task may not change silently.  Each assertion
corresponds to a claim made in docs/SCORING_CONTRACT.md or AGENT_TASK.md.
"""

from __future__ import annotations

import pytest

from conftest import TASK_ROOT, _load

runtime = _load("mamba2_ssd_runtime", "runtime/mamba2_ssd_runtime.py")

AGENT_TASK = (TASK_ROOT / "images/work/AGENT_TASK.md").read_text()
# docs/SCORING_CONTRACT.md is an internal document (the public tree has none).
SCORING_PATH = TASK_ROOT / "docs/SCORING_CONTRACT.md"
SCORING = SCORING_PATH.read_text() if SCORING_PATH.is_file() else None


def test_entry_point_name() -> None:
    assert runtime.ENTRYPOINT == "ssd_forward"
    assert "def ssd_forward(x, dt, A, B, C, D, out) -> None" in AGENT_TASK


def test_geometries_are_deployed_layouts_inside_the_domain() -> None:
    """Each geometry is a published model config or its per-rank TP layout (checked 2026-11-07)."""
    assert runtime.GEOMETRIES["mamba2_2p7b"] == {"h": 80, "p": 64, "n": 128, "g": 1}
    assert runtime.GEOMETRIES["codestral_7b"] == {"h": 128, "p": 64, "n": 128, "g": 8}
    assert runtime.GEOMETRIES["falcon_h1_34b"] == {"h": 32, "p": 128, "n": 256, "g": 2}
    assert runtime.GEOMETRIES["nemotron_h_47b"] == {"h": 256, "p": 64, "n": 256, "g": 8}
    assert len(runtime.GEOMETRIES) == 15
    for name, geometry in runtime.GEOMETRIES.items():
        assert geometry["h"] % geometry["g"] == 0, name
        assert 1 <= geometry["h"] <= 256 and geometry["p"] % 16 == 0, name
        assert geometry["n"] in (16, 32, 64, 128, 256), name
    for name in ("nemotron_h_47b_tp2", "nemotron_h_47b_tp4", "nemotron_h_47b_tp8"):
        tp = int(name[-1])
        parent = runtime.GEOMETRIES["nemotron_h_47b"]
        assert runtime.GEOMETRIES[name] == {
            "h": parent["h"] // tp, "p": parent["p"], "n": parent["n"], "g": max(1, parent["g"] // tp)
        }


def test_workload_families_and_final_counts() -> None:
    assert runtime.SCORE_FAMILIES == ("batch", "serving", "long", "real")
    assert runtime.FINAL_FAMILY_COUNTS == {"batch": 42, "serving": 54, "long": 18, "real": 24}
    assert runtime.FINAL_CASE_COUNT == 138
    assert runtime.REAL_TRIALS == 1


def test_family_rules_are_enforced() -> None:
    base = {"case_id": "x", "geometry": "granite_h_micro", "tier": "sweep", "regime": "mixed",
            "seed": 1, "check_trial": 0, "layer": None}
    runtime.validate_case({**base, "family": "serving", "batch": 1, "seqlen": 6909}, scored=True)
    for bad in (
        {"family": "serving", "batch": 2, "seqlen": 6909},
        {"family": "batch", "batch": 1, "seqlen": 6909},
        {"family": "long", "batch": 1, "seqlen": 32768},
        {"family": "real", "batch": 1, "seqlen": 16381},
        {"family": "serving", "batch": 1, "seqlen": 2**31 // (64 * 64) + 1},
        {"family": "batch", "batch": 17, "seqlen": 1024},
    ):
        with pytest.raises(ValueError):
            runtime.validate_case({**base, **bad}, scored=True)
    runtime.validate_case(
        {**base, "family": "real", "tier": "real", "batch": 1, "seqlen": 16381,
         "layer": "granite-h-micro-layer00"},
        scored=True,
    )


def test_family_weights_are_equal() -> None:
    assert set(runtime.FAMILY_WEIGHTS) == set(runtime.SCORE_FAMILIES)
    assert all(weight == pytest.approx(0.25) for weight in runtime.FAMILY_WEIGHTS.values())
    assert sum(runtime.FAMILY_WEIGHTS.values()) == pytest.approx(1.0)


def test_workspace_limit_is_the_smaller_of_the_multiple_and_the_cap() -> None:
    assert runtime.workspace_limit_bytes(2**30) == 4 * 2**30
    assert runtime.workspace_limit_bytes(9 * 2**30) == 24 * 2**30


def test_effective_flops_is_chunk_size_independent() -> None:
    assert runtime.effective_flops(8, 8192, 128, 64, 128) == 4 * 8 * 8192 * 128 * 64 * 128
    assert runtime.effective_flops(8, 4096, 32, 128, 256) == 4 * 8 * 4096 * 32 * 128 * 256
    assert runtime.effective_flops(2, 2048, 80, 64, 128) == 2 * runtime.effective_flops(
        2, 1024, 80, 64, 128
    )


def test_effective_tflops_scales_inversely_with_time() -> None:
    fast = runtime.effective_tflops(8, 8192, 128, 64, 128, 1.0e-3)
    slow = runtime.effective_tflops(8, 8192, 128, 64, 128, 2.0e-3)
    assert fast == pytest.approx(2.0 * slow)


def test_tolerances_match_the_published_contract() -> None:
    assert runtime.OUT_REL_L2_TOL == 6.0e-3
    assert runtime.OUT_SCALED_ABS_TOL == 1.5e-2
    assert runtime.OUT_REL_TOL == pytest.approx(3.01 / 128.0)
    if SCORING is not None:
        assert "6e-3" in SCORING and "1.5e-2" in SCORING
    assert "6e-3" in AGENT_TASK and "1.5e-2" in AGENT_TASK


def test_tolerance_separates_honest_from_buggy() -> None:
    """The qualification claim, asserted as a number rather than a sentence.

    Worst measured honest relative L2 was 2.37e-3; the nearest planted bug --
    dropping the inter-chunk state -- was 0.21.
    """
    assert runtime.OUT_REL_L2_TOL > 2.0 * 2.37e-3
    assert runtime.OUT_REL_L2_TOL < 0.21 / 20.0


def test_budget_gates() -> None:
    assert runtime.WARMUP_WALL_CAP_SECONDS == 300.0
    assert runtime.WORKSPACE_MULTIPLE == 4.0


def test_regimes_cover_both_decay_extremes() -> None:
    assert set(runtime.REGIMES) == {"mixed", "slow_decay", "fast_decay", "unit_dt"}
    slow_a, slow_dt = runtime._regime_ranges("slow_decay")
    fast_a, fast_dt = runtime._regime_ranges("fast_decay")
    assert slow_a[1] + slow_dt[1] < fast_a[0] + fast_dt[0]


def test_agent_task_forbids_every_escape_route() -> None:
    for token in (
        "T.import_source", "T.CUDASourceCodeKernel", "T.call_extern", "T.call_packed",
        "T.call_kernel", "compile_flags", "execution_backend", "register_*_postproc",
        "JITKernel.from_database", "tilelang.contrib", "T.ptx_*", "tilelang.autotune",
        "torch.matmul", "scaled_dot_product_attention", "x.__matmul__",
        "operator.attrgetter", "torch.linalg", "tilelang_only_ok",
    ):
        assert token in AGENT_TASK, token


def test_reference_blocking_is_bounded() -> None:
    assert runtime.REF_CHUNK in (64, 128, 256)
    assert runtime.REF_HEAD_BLOCK >= 1
    assert runtime.REF_CHUNK_BLOCK >= 1
