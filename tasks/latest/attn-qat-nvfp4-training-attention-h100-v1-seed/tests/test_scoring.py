from __future__ import annotations

import math

import pytest
from conftest import load_module

runtime = load_module("attn_qat_runtime_scoring", "runtime/attn_qat_runtime.py")


def test_forward_flops_counts_two_qk_and_two_pv_passes() -> None:
    # Statistics pre-pass QK, main-pass QK, `Pq Vf` and `P~ Vf`, each 2 FLOPs per MAC.
    assert runtime.forward_flops(1, 1, 4, 4, 8) == 8 * 4 * 4 * 8
    assert runtime.forward_flops(2, 3, 5, 7, 64) == 8 * 2 * 3 * 5 * 7 * 64


def test_backward_flops_counts_the_five_matrix_products() -> None:
    # QK recompute, dV, dP, dK, dQ.  The seed executes seven; the denominator counts the
    # algorithmic minimum, so removing the duplicated recomputation is worth score.
    assert runtime.backward_flops(1, 1, 4, 4, 8) == 10 * 4 * 4 * 8
    assert runtime.backward_flops(2, 3, 5, 7, 64) == 10 * 2 * 3 * 5 * 7 * 64


def test_effective_flops_is_the_production_weighting() -> None:
    # Full activation checkpointing runs the forward twice per backward.
    for geometry in ((1, 1, 4, 4, 8), (1, 3, 31200, 31200, 128), (1, 32, 47616, 47616, 64)):
        b, h, s_q, s_kv, d = geometry
        assert runtime.effective_flops(*geometry) == 26 * b * h * s_q * s_kv * d
        assert runtime.effective_flops(*geometry) == (
            2 * runtime.forward_flops(*geometry) + runtime.backward_flops(*geometry)
        )


def test_trial_time_weights_the_forward_twice() -> None:
    assert runtime.trial_seconds(2.0, 3.0) == pytest.approx(7.0)
    with pytest.raises(RuntimeError):
        runtime.trial_seconds(0.0, 1.0)


def test_effective_tflops_divides_flops_by_the_weighted_time() -> None:
    flops = runtime.effective_flops(1, 2, 8, 8, 16)
    seconds = runtime.trial_seconds(1e-6, 2e-6)
    assert runtime.effective_tflops(1, 2, 8, 8, 16, seconds) == pytest.approx(
        flops / seconds / 1e12
    )


@pytest.mark.parametrize("bad", (0, -1, True))
def test_flop_accounting_rejects_non_positive_geometry(bad) -> None:
    with pytest.raises(ValueError):
        runtime.effective_flops(1, bad, 8, 8, 16)


def _metrics(**changes: float) -> dict[str, float]:
    row: dict[str, float] = {}
    for name in runtime.GATE_TENSORS:
        row[f"{name}_rel_l2"] = 1.0e-3
        row[f"{name}_head_rel_l2"] = 2.0e-3
        row[f"{name}_max_abs"] = 1.0
        row[f"{name}_max_abs_ref"] = 1.0
        row[f"{name}_ref_norm"] = 10.0
    row.update(changes)
    return row


def test_shipped_gate_constants_are_the_a2_calibration() -> None:
    # Helper A2, 27 cases x 4 honest implementations x 19 planted contract violations.
    assert runtime.GATE_TENSORS == ("o", "dq", "dk", "dv")
    assert runtime.O_REL == 1.0e-2
    assert runtime.DQ_REL == 8.0e-2
    assert runtime.DK_REL == 3.0e-2
    assert runtime.DV_REL == 1.2e-1
    assert runtime.SCORED_GATE_LIMITS == {
        "o": 1.0e-2,
        "dq": 8.0e-2,
        "dk": 3.0e-2,
        "dv": 1.2e-1,
    }
    assert runtime.ZERO_REF_ABS == 1.0e-4
    assert runtime.GATE_CALIBRATION == "a2-27-cases-4-honest-19-planted-2026-09-06"


def test_every_edge_fixture_has_its_own_limits() -> None:
    # Fixed inputs, so each fixture is calibrated individually; scored cases cannot be,
    # because the holdout draws its seed and its slot from a hidden manifest.
    assert set(runtime.FIXTURE_GATE_LIMITS) == set(runtime.EDGE_CASE_IDS)
    assert runtime.GATE_FLOORS == {"o": 5.0e-3, "dq": 1.5e-2, "dk": 1.0e-2, "dv": 2.0e-2}
    assert runtime.GATE_CAPS == {"o": 1.0e-2, "dq": 1.5e-1, "dk": 3.0e-2, "dv": 3.0e-1}
    for case_id, limits in runtime.FIXTURE_GATE_LIMITS.items():
        assert set(limits) == set(runtime.GATE_TENSORS), case_id
        for name, value in limits.items():
            assert value >= runtime.GATE_FLOORS[name], (case_id, name)
            assert value <= runtime.GATE_CAPS[name], (case_id, name)
    # The two rows the design record names explicitly.
    assert runtime.FIXTURE_GATE_LIMITS["edge-e7-wan-grad-capture"]["dq"] == 1.5e-1
    assert runtime.FIXTURE_GATE_LIMITS["edge-e4-unaligned-cross"]["dv"] == 3.0e-1
    # e1 is the tightest row: at S = 64 no 16-group is anywhere near the e4m3 flush
    # threshold, which is what makes it the fixture that catches a flush-only cheat.
    assert runtime.FIXTURE_GATE_LIMITS["edge-e1-tiny"] == {
        "o": 1.0e-2,
        "dq": 1.5e-2,
        "dk": 1.5e-2,
        "dv": 2.0e-2,
    }


def test_every_limit_clears_the_worst_honest_error_by_at_least_two() -> None:
    # Worst honest error over four honest implementations, from helper A2's calibration.
    # The margin the calibration bought, asserted so a later edit cannot silently spend it.
    scored_worst = {"o": 2.585e-3, "dq": 2.459e-2, "dk": 4.044e-3, "dv": 3.594e-2}
    for name, value in scored_worst.items():
        assert runtime.SCORED_GATE_LIMITS[name] >= 2.0 * value, name
    fixture_worst = {
        "edge-e1-tiny": {"o": 2.410e-3, "dq": 3.078e-3, "dk": 4.967e-3, "dv": 1.769e-3},
        "edge-e2-unaligned": {"o": 1.998e-3, "dq": 1.401e-2, "dk": 4.262e-3, "dv": 1.020e-2},
        "edge-e3-cross-attention": {
            "o": 2.233e-3, "dq": 1.923e-2, "dk": 4.757e-3, "dv": 8.766e-3,
        },
        "edge-e4-unaligned-cross": {
            "o": 1.831e-3, "dq": 3.544e-3, "dk": 2.563e-3, "dv": 1.076e-1,
        },
        "edge-e5-zero-key-zero-v-groups": {"o": 1.595e-3, "dk": 2.325e-3},
        "edge-e6-rtne-ties": {"o": 2.112e-3, "dq": 5.478e-3, "dk": 5.137e-3, "dv": 1.015e-2},
        "edge-e7-wan-grad-capture": {
            "o": 2.809e-3, "dq": 4.308e-2, "dk": 9.507e-3, "dv": 1.279e-2,
        },
        "edge-e8-forced-sink": {"o": 2.535e-3, "dq": 5.154e-3, "dk": 4.116e-3, "dv": 2.651e-3},
        "edge-e9-tiny-do-scale": {
            "o": 1.767e-3, "dq": 6.388e-3, "dk": 3.329e-3, "dv": 2.474e-3,
        },
        "edge-e10-ltx2-minimal": {
            "o": 3.336e-3, "dq": 8.057e-3, "dk": 6.350e-3, "dv": 2.123e-2,
        },
        "edge-e11-head-dim-64": {
            "o": 2.098e-3, "dq": 2.780e-3, "dk": 2.755e-3, "dv": 2.070e-2,
        },
    }
    for case_id, measured in fixture_worst.items():
        for name, value in measured.items():
            assert runtime.FIXTURE_GATE_LIMITS[case_id][name] >= 2.0 * value, (case_id, name)


def test_gate_group_selection() -> None:
    assert runtime.gate_group({"case_id": "anchor-c1", "scored": True}) == "scored"
    assert runtime.gate_group({"case_id": "holdout-c4", "scored": True}) == "scored"
    for case_id in runtime.EDGE_CASE_IDS:
        assert runtime.gate_group({"case_id": case_id, "scored": False}) == case_id


def test_realised_limits_follow_the_group_table() -> None:
    scored = runtime.realised_limits({"case_id": "anchor-c1", "scored": True})
    assert scored["gate_group"] == "scored"
    assert scored["dv_rel_l2_limit"] == pytest.approx(1.2e-1)
    assert scored["dq_rel_l2_limit"] == pytest.approx(8.0e-2)
    fixture = runtime.realised_limits({"case_id": runtime.FIXTURE_CASE_ID, "scored": False})
    assert fixture["dq_rel_l2_limit"] == pytest.approx(1.5e-1)
    assert fixture["dv_rel_l2_limit"] == pytest.approx(5.0e-2)
    tiny = runtime.realised_limits({"case_id": "edge-e1-tiny", "scored": False})
    assert tiny["dv_rel_l2_limit"] == pytest.approx(2.0e-2)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("o_rel_l2", 1.01e-2, "o global"),
        ("dq_rel_l2", 8.01e-2, "dq global"),
        ("dk_rel_l2", 3.01e-2, "dk global"),
        ("dv_rel_l2", 1.21e-1, "dv global"),
    ),
)
def test_each_gate_fails_only_on_its_inequality(field: str, value: float, message: str) -> None:
    scored = {"case_id": "anchor-c1", "scored": True}
    with pytest.raises(RuntimeError, match=message):
        runtime.check_metric_bundle(_metrics(**{field: value}), scored)
    runtime.check_metric_bundle(_metrics(), scored)


def test_a_fixture_row_can_be_tighter_than_the_scored_row() -> None:
    # A candidate comfortably inside the flat scored limits is still rejected by e1, which
    # is what makes a flush-only violation visible.
    candidate = _metrics(dv_rel_l2=6.0e-2)
    runtime.check_metric_bundle(candidate, {"case_id": "anchor-c1", "scored": True})
    with pytest.raises(RuntimeError, match="dv global"):
        runtime.check_metric_bundle(candidate, {"case_id": "edge-e1-tiny"})
    # And looser where the fixture earned it.
    dq_candidate = _metrics(dq_rel_l2=1.0e-1)
    with pytest.raises(RuntimeError, match="dq global"):
        runtime.check_metric_bundle(dq_candidate, {"case_id": "edge-e1-tiny"})
    runtime.check_metric_bundle(dq_candidate, {"case_id": runtime.FIXTURE_CASE_ID})


def test_zero_reference_gate_replaces_the_relative_one() -> None:
    # `dq` is exactly zero when `k = 0`, so the relative statistic is undefined.
    scored = {"case_id": "anchor-c1", "scored": True}
    limits = runtime.realised_limits(scored)
    zero = _metrics(dq_ref_norm=0.0, dq_rel_l2=float("inf"), dq_max_abs=1e-5)
    assert runtime.gate_failures(zero, limits) == ["nonfinite metric dq_rel_l2"]
    passing = _metrics(dq_ref_norm=0.0, dq_rel_l2=0.0, dq_max_abs=1e-5)
    assert runtime.gate_failures(passing, limits) == []
    failing = _metrics(dq_ref_norm=0.0, dq_rel_l2=0.0, dq_max_abs=1e-3)
    assert runtime.gate_failures(failing, limits) == ["dq zero-reference absolute gate failed"]


def test_nonfinite_metric_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="nonfinite"):
        runtime.check_metric_bundle(
            _metrics(o_rel_l2=float("nan")), {"case_id": "anchor-c1", "scored": True}
        )


def test_every_failing_gate_is_named_not_just_the_first() -> None:
    candidate = _metrics(o_rel_l2=0.5, dq_rel_l2=0.5, dk_rel_l2=0.5, dv_rel_l2=0.5)
    scored = {"case_id": "anchor-c1", "scored": True}
    limits = runtime.realised_limits(scored)
    assert runtime.gate_failures(candidate, limits) == [
        "o global rel-L2 gate failed",
        "dq global rel-L2 gate failed",
        "dk global rel-L2 gate failed",
        "dv global rel-L2 gate failed",
    ]
    with pytest.raises(RuntimeError) as failure:
        runtime.check_metric_bundle(candidate, scored)
    for fragment in ("o global", "dq global", "dk global", "dv global"):
        assert fragment in str(failure.value)


def test_gate_failures_is_empty_exactly_when_the_candidate_passes() -> None:
    limits = runtime.realised_limits({"case_id": "anchor-c1", "scored": True})
    assert runtime.gate_failures(_metrics(), limits) == []
    assert runtime.gate_failures(_metrics(dv_rel_l2=1.21e-1), limits) == [
        "dv global rel-L2 gate failed"
    ]


def test_aggregate_weighs_tiers_equally_inside_a_family_and_families_equally() -> None:
    rows = [
        {"scored": True, "family": "wan13b", "tier": "recipe", "tflops": 100.0},
        {"scored": True, "family": "wan13b", "tier": "sweep", "tflops": 400.0},
        {"scored": True, "family": "wan13b", "tier": "sweep", "tflops": 400.0},
        {"scored": True, "family": "wan13b", "tier": "sweep", "tflops": 400.0},
        {"scored": True, "family": "ltx2", "tier": "recipe", "tflops": 50.0},
        {"scored": True, "family": "kandinsky5", "tier": "recipe", "tflops": 25.0},
    ]
    result = runtime.aggregate_case_results(rows)
    assert result["family_tflops"] == pytest.approx(
        {"wan13b": 200.0, "ltx2": 50.0, "kandinsky5": 25.0}
    )
    assert result["primary_tflops"] == pytest.approx((200.0 * 50.0 * 25.0) ** (1.0 / 3.0))


def test_trial_seed_derivation_is_distinct_per_trial_and_stream() -> None:
    base = 17011
    trial = [base ^ (runtime.TRIAL_SEED_BASE + i) for i in range(7)]
    decoy = [base ^ (runtime.DECOY_SEED_BASE + i) for i in range(7)]
    warm = base ^ runtime.WARMUP_SEED_XOR
    generator_warmup = base ^ runtime.GENERATOR_WARMUP_XOR
    everything = trial + decoy + [warm, generator_warmup, base]
    assert len(set(everything)) == len(everything)


def test_saved_cap_is_exactly_upstreams_footprint() -> None:
    # Qf, Kf, Vf and O' in bf16 plus the fp32 row statistic M.
    for b, h, s_q, s_kv, d in ((1, 3, 31200, 31200, 128), (2, 2, 1000, 515, 128)):
        expected = 2 * b * h * d * (2 * s_q + 2 * s_kv) + 4 * b * h * s_q
        assert runtime.saved_bytes_limit(b, h, s_q, s_kv, d) == expected
    # Self attention: exactly four bf16 tensors of q's shape plus one fp32 row vector.
    assert runtime.saved_bytes_limit(1, 3, 100, 100, 128) == 4 * 2 * 3 * 100 * 128 + 4 * 3 * 100


def test_memory_bound_covers_the_largest_scored_sets() -> None:
    generator = load_module("attn_qat_scoring_generator", "scripts/generate_final_manifest.py")
    rows = generator.grid_cases()
    for row in rows:
        args = (1, row["s_q"], row["s_kv"], row["h"], row["d"])
        assert runtime.case_resident_bytes(*args) <= runtime.MAX_RESIDENT_BYTES
        assert runtime.case_input_bytes(*args) <= runtime.MAX_CASE_INPUT_BYTES
    total = sum(
        runtime.case_input_bytes(1, row["s_q"], row["s_kv"], row["h"], row["d"]) for row in rows
    )
    assert total <= runtime.MAX_SUITE_INPUT_BYTES


def test_geometric_mean_rejects_non_positive_values() -> None:
    assert runtime.geometric_mean([4.0, 9.0]) == pytest.approx(6.0)
    with pytest.raises(RuntimeError):
        runtime.geometric_mean([1.0, 0.0])
    with pytest.raises(ValueError):
        runtime.geometric_mean([])
    assert math.isfinite(runtime.geometric_mean([1e-6, 1e6]))
