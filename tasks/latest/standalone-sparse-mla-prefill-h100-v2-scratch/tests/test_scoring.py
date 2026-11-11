"""Judge-side manifest validation and score recomputation, without GPUs.

The manifests and runtime results here are synthetic (``suite_fixtures``): the
final manifest is the final generator's draw from a made-up plan that keeps the
frozen design's group counts and the disclosed anchor shapes, and the edge
manifest is the edge generator's output for a test seed, so these tests do not
depend on the frozen private instance (its own tests validate it through the
same functions).
"""
from __future__ import annotations

import copy
import math
from typing import Any, Callable

import pytest

from conftest import load_module
from suite_fixtures import (  # noqa: F401 - re-exported for the other test modules
    H100,
    TEST_EDGE_SEED,
    accuracy as _accuracy,
    build_edge_manifest,
    build_final_manifest,
    build_runtime_result,
    manifests,
)


verifier = load_module("sparse_mla_v2_private_verifier_scoring", "images/judge/verify_sparse_mla.py")
runtime = load_module("sparse_mla_v2_runtime_scoring", "runtime/sparse_mla_prefill_runtime.py")
final_generator = load_module("sparse_mla_v2_final_generator_scoring", "scripts/generate_final_manifest.py")

TRIALS = verifier.EXPECTED_TRIALS
WARMUPS = verifier.EXPECTED_WARMUPS


# --------------------------------------------------------------------------
# The Judge's duplicated tables equal the trusted runtime's.
# --------------------------------------------------------------------------
def test_judge_tables_match_the_runtime() -> None:
    for name in (
        "FAMILIES", "FAMILY_GEOMETRY", "FAMILY_TOPK", "FAMILY_MODEL", "FAMILY_MAX_CONTEXT",
        "FAMILY_SM_SCALE", "ANCHOR_S_KV", "ANCHOR_S_Q", "ANCHOR_SM_SCALE", "GROUP_TIER", "GROUPS", "TIERS",
        "STREAM_GROUPS", "SCORED_SINK_MODES", "LENGTH_DIVISORS", "MAX_REQUESTS", "MAX_STREAM_CALLS",
        "STREAM_WARM_CALLS", "STREAM_CHECK_EVERY", "STREAM_TIME_LIMIT_SECONDS", "STALE_TAIL_FRACTION",
        "MIN_H_Q", "H_Q_MULTIPLE", "D_V", "MIN_TOPK", "MAX_TOPK", "TOPK_MULTIPLE", "MAX_S_Q", "MAX_S_KV",
        "MAX_Q_ELEMENTS", "MAX_TOPK_POSITIONS", "NATIVE_MAX_TENSOR_BYTES", "LARGE_OFFSET_ELEMENTS",
        "MAX_SEED", "INDEX_PATTERNS", "BIAS_MODES", "INDEX_MODES", "INVALID_ROW_MODES", "SINK_MODES", "LENGTH_MODES",
        "OUT_ABS_TOL", "OUT_REL_TOL", "OUT_COS_TOL", "LOGIT_ABS_TOL", "LOGIT_REL_TOL", "LOGIT_COS_TOL",
        "COS_LIMITS", "GUARD_MIN_BYTES", "GUARD_ROWS", "L2_FLUSH_BYTES", "EDGE_CALLS",
        "GENERATOR_PROVENANCE", "SCHEMA_VERSION", "PREFILL_ENTRYPOINT", "SCORED_KEYS", "SCORED_GROUP_KEYS",
        "EDGE_KEYS", "EDGE_OPTIONAL_KEYS", "MIN_COMPUTE_CAPABILITY", "MIN_DEVICE_MEMORY_BYTES",
        "ACCURACY_FIELDS",
    ):
        assert getattr(verifier, name) == getattr(runtime, name), name
    assert runtime.FAMILIES == ("v32", "glm5", "v4_flash", "v4_pro")
    assert runtime.MAX_S_KV == 1 << 20 and runtime.NATIVE_MAX_TENSOR_BYTES == 3 << 30
    assert "causal" in runtime.INDEX_MODES
    assert verifier.SOFT_LANDING == {"indices": runtime.SOFT_INDEX, "topk_length": runtime.SOFT_LENGTH}
    assert verifier.FINAL_CASE_COUNT == final_generator.EXPECTED_CASE_COUNT == 997
    assert verifier.FINAL_SCHEMA_VERSION == final_generator.SCHEMA_VERSION == 5
    assert verifier.FINAL_SUITE_ID == final_generator.SUITE_ID == "final-holdout-v4"
    assert tuple(verifier.FINAL_GROUP_COUNTS) == runtime.GROUPS == final_generator.GROUPS
    assert verifier.FINAL_GROUP_COUNTS == {
        "anchor": 17, "trace": 80, "replay": 60, "multi": 100, "pattern": 120,
        "sweep": 400, "heads": 120, "length": 100,
    }
    assert sum(verifier.FINAL_GROUP_COUNTS.values()) == verifier.FINAL_CASE_COUNT
    assert verifier.MAX_REQUESTS == 128 and verifier.STREAM_TIME_LIMIT_SECONDS == 8.0
    assert runtime.INDEX_PATTERNS == ("window", "sink_window", "blocks", "sorted")
    assert [group for group, tier in runtime.GROUP_TIER.items() if tier == "evidence"] == [
        "anchor", "trace", "replay", "multi", "pattern",
    ]
    assert [group for group, tier in runtime.GROUP_TIER.items() if tier == "sweep"] == ["sweep", "heads", "length"]
    assert verifier.FINAL_GROUP_COUNTS["anchor"] == len(runtime.anchor_grid())
    assert final_generator.FAMILIES == runtime.FAMILIES
    assert final_generator.FAMILY_HEADS == {
        family: heads for family, (_d_qk, heads) in runtime.FAMILY_GEOMETRY.items()
    }
    assert final_generator.STREAM_GROUPS == runtime.STREAM_GROUPS
    assert set(final_generator.HALF_GATED_GROUPS) == {
        group for group, tier in runtime.GROUP_TIER.items() if tier == "sweep"
    }
    assert final_generator.SEED_RANGE[0] == verifier.FINAL_SEED_MINIMUM
    assert final_generator.SEED_RANGE[1] == verifier.MAX_SEED + 1
    assert verifier.EXPECTED_TRIALS == 3 and verifier.EXPECTED_WARMUPS == 1
    assert verifier.SANITIZE_STATUSES == runtime.SANITIZE_STATUSES
    assert verifier.expected_protocol(1000.0) == runtime.protocol_record(1000.0)
    assert verifier.RUNTIME_GATES == tuple(sorted(runtime.GATE_ORDER, key=verifier.RUNTIME_GATES.index))
    assert set(verifier.RUNTIME_GATES) == set(runtime.GATE_ORDER)


def test_plausibility_bounds_match_the_runtime() -> None:
    assert verifier.BF16_DENSE_PEAK_BOUND_TFLOPS == runtime.BF16_DENSE_PEAK_BOUND_TFLOPS
    assert verifier.plausibility_bound([9, 0]) == runtime.plausibility_bound([9, 0]) == 1000.0
    assert verifier.plausibility_bound([7, 5]) is None


def test_geometric_mean() -> None:
    assert math.isclose(verifier.geometric_mean([1.0, 4.0, 16.0]), 4.0)
    assert math.isclose(runtime.geometric_mean([1.0, 4.0, 16.0]), 4.0)
    with pytest.raises(RuntimeError):
        verifier.geometric_mean([1.0, 0.0])


def test_sanitize_order_matches_the_runtime() -> None:
    _cases, edges = manifests()
    expected = [case["case_id"] for case in runtime.sanitize_order(edges)]
    assert [case["case_id"] for case in verifier.sanitize_order(edges)] == expected
    flagged = [case for case in edges if case["sanitize"]]
    assert len(expected) == verifier.EDGE_SANITIZE_COUNT == len(flagged) == 17
    # The sanitizer-only case runs first.
    first = next(case for case in edges if case["case_id"] == expected[0])
    assert first["native"] is False and first["s_q"] == 70000
    assert sum(not case["native"] for case in edges) == verifier.EDGE_SANITIZE_ONLY_COUNT == 1


def test_synthetic_manifests_satisfy_both_the_judge_and_the_runtime() -> None:
    cases, edges = manifests()
    assert len(cases) == verifier.FINAL_CASE_COUNT == 997 and len(edges) == verifier.EDGE_CASE_COUNT == 470
    for case in cases:
        scored = runtime.validate_scored_case(case)
        assert scored["tier"] == runtime.GROUP_TIER[case["group"]]
        assert scored["stream"] is (case["group"] in runtime.STREAM_GROUPS)
    for case in edges:
        runtime.validate_edge_case(case)
    # Every group holds every family; every anchor shape appears once.
    assert {(case["group"], case["family"]) for case in cases} == {
        (group, family) for group in runtime.GROUPS for family in runtime.FAMILIES
    }
    assert sorted((case["family"], case["s_kv"]) for case in cases if case["group"] == "anchor") == sorted(
        runtime.anchor_grid()
    )


def _first(payload: dict[str, Any], group: str) -> dict[str, Any]:
    return next(case for case in payload["cases"] if case["group"] == group)


def _final_mutation(name: str) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        cases = payload["cases"]
        anchors = [case for case in cases if case["group"] == "anchor"]
        if name == "missing_case":
            cases.pop()
        elif name == "duplicate_anchor":
            same = next(case for case in anchors[1:] if case["family"] == anchors[0]["family"])
            same["s_kv"] = anchors[0]["s_kv"]
        elif name == "off_grid_anchor":
            anchors[0]["s_kv"] = 16384
        elif name == "anchor_s_q":
            anchors[0]["s_q"] = 2048
        elif name == "anchor_sink":
            anchors[0]["sink_mode"] = "none"
        elif name == "upstream_sink_off_anchor":
            _first(payload, "sweep")["sink_mode"] = "upstream"
        elif name == "low_seed":
            cases[0]["seed"] = 12345
        elif name == "huge_seed":
            cases[0]["seed"] = 1 << 63
        elif name == "repeated_seed":
            cases[1]["seed"] = cases[0]["seed"]
        elif name == "unknown_family":
            cases[0]["family"] = "m1_64"
        elif name == "unknown_group":
            _first(payload, "sweep")["group"] = "bonus"
        elif name == "extra_field":
            cases[0]["b"] = 256
        elif name == "derived_field":
            # The runtime derives topk from the family; the frozen manifest never carries it.
            cases[0]["topk"] = 2048
        elif name == "missing_group_field":
            del _first(payload, "length")["length_divisor"]
        elif name == "foreign_group_field":
            _first(payload, "sweep")["length_divisor"] = 2
        elif name == "group_counts":
            moved = _first(payload, "sweep")
            moved["group"] = "length"
            moved["length_divisor"] = 2
            moved["ragged"] = False
        elif name == "heads_at_full_width":
            case = _first(payload, "heads")
            case["h_q"] = runtime.FAMILY_GEOMETRY[case["family"]][1]
        elif name == "sweep_changes_heads":
            case = _first(payload, "sweep")
            case["h_q"] = runtime.FAMILY_GEOMETRY[case["family"]][1] // 2
        elif name == "heads_not_multiple_of_16":
            _first(payload, "heads")["h_q"] = 24
        elif name == "acausal_sweep":
            case = _first(payload, "sweep")
            case["s_q"] = case["s_kv"] + 1
        elif name == "beyond_context":
            case = _first(payload, "sweep")
            case["s_kv"] = runtime.FAMILY_MAX_CONTEXT[case["family"]] + 1
        elif name == "length_divisor":
            _first(payload, "length")["length_divisor"] = 3
        elif name == "missing_ragged":
            del _first(payload, "length")["ragged"]
        elif name == "ragged_not_bool":
            _first(payload, "length")["ragged"] = 1
        elif name == "foreign_ragged":
            _first(payload, "sweep")["ragged"] = False
        elif name == "missing_index_pattern":
            del _first(payload, "pattern")["index_pattern"]
        elif name == "unknown_index_pattern":
            _first(payload, "pattern")["index_pattern"] = "uniform"
        elif name == "foreign_index_pattern":
            _first(payload, "sweep")["index_pattern"] = "window"
        elif name == "ungated_pattern":
            _first(payload, "pattern")["gated"] = False
        elif name == "acausal_pattern":
            case = _first(payload, "pattern")
            case["s_q"] = case["s_kv"] + 1
        elif name == "multi_lengths_sum":
            _first(payload, "multi")["request_lengths"][0] += 1
        elif name == "multi_one_request":
            case = _first(payload, "multi")
            case["request_lengths"] = [case["s_kv"]]
        elif name == "ungated_multi":
            _first(payload, "multi")["gated"] = False
        elif name == "ungated_stream":
            _first(payload, "trace")["gated"] = False
        elif name == "stream_extent":
            _first(payload, "replay")["s_kv"] += 1
        elif name == "stream_acausal_call":
            case = _first(payload, "trace")
            case["calls"][0] = [case["calls"][0][1] + 1, case["calls"][0][1]]
        elif name == "stream_too_long":
            case = _first(payload, "trace")
            case["calls"] = case["calls"] * (runtime.MAX_STREAM_CALLS // len(case["calls"]) + 1)
        elif name == "native_tensor_cap":
            # v32: 2 * 30000 * 128 * 576 bytes of q is above 3 GiB.
            case = next(case for case in payload["cases"] if case["group"] == "sweep" and case["family"] == "v32")
            case.update(s_q=30000, s_kv=60000)
        elif name == "schema_version":
            payload["schema_version"] = 4
        elif name == "suite_id":
            payload["suite_id"] = "final-holdout-v3"
        elif name == "bad_case_id":
            cases[0]["case_id"] = "Final V32"
        else:  # pragma: no cover
            raise AssertionError(name)

    return mutate


FINAL_MUTATIONS = (
    "missing_case", "duplicate_anchor", "off_grid_anchor", "anchor_s_q", "anchor_sink",
    "upstream_sink_off_anchor", "low_seed", "huge_seed", "repeated_seed", "unknown_family", "unknown_group",
    "extra_field", "derived_field", "missing_group_field", "foreign_group_field", "group_counts",
    "heads_at_full_width",
    "sweep_changes_heads", "heads_not_multiple_of_16", "acausal_sweep", "beyond_context", "length_divisor",
    "missing_ragged", "ragged_not_bool", "foreign_ragged", "missing_index_pattern", "unknown_index_pattern",
    "foreign_index_pattern", "ungated_pattern", "acausal_pattern",
    "multi_lengths_sum", "multi_one_request", "ungated_multi", "ungated_stream", "stream_extent",
    "stream_acausal_call", "stream_too_long", "native_tensor_cap", "schema_version", "suite_id", "bad_case_id",
)
# Single-case mutations that the runtime's own case validation rejects as well.
RUNTIME_CASE_MUTATIONS = {
    "off_grid_anchor", "anchor_s_q", "anchor_sink", "upstream_sink_off_anchor", "huge_seed", "unknown_family",
    "unknown_group", "extra_field", "missing_group_field", "foreign_group_field", "heads_at_full_width",
    "sweep_changes_heads", "heads_not_multiple_of_16", "acausal_sweep", "beyond_context", "length_divisor",
    "missing_ragged", "ragged_not_bool", "foreign_ragged", "missing_index_pattern", "unknown_index_pattern",
    "foreign_index_pattern", "ungated_pattern", "acausal_pattern",
    "multi_lengths_sum", "multi_one_request", "ungated_multi", "ungated_stream", "stream_extent",
    "stream_acausal_call", "stream_too_long", "native_tensor_cap",
}


@pytest.mark.parametrize("mutation", FINAL_MUTATIONS)
def test_final_manifest_is_exactly_the_frozen_design(mutation: str) -> None:
    payload = build_final_manifest()
    verifier.validate_final_manifest(copy.deepcopy(payload))
    original = copy.deepcopy(payload["cases"])
    _final_mutation(mutation)(payload)
    with pytest.raises(RuntimeError):
        verifier.validate_final_manifest(payload)
    if mutation in RUNTIME_CASE_MUTATIONS:
        changed = [case for case, before in zip(payload["cases"], original) if case != before]
        assert changed
        for case in changed:
            with pytest.raises(ValueError):
                runtime.validate_scored_case(case)


def test_final_generator_draws_seeds_sinks_and_the_checked_half() -> None:
    payload = build_final_manifest()
    cases = payload["cases"]
    assert payload["schema_version"] == 5 and payload["suite_id"] == verifier.FINAL_SUITE_ID
    assert len({case["seed"] for case in cases}) == len(cases)
    assert all(verifier.FINAL_SEED_MINIMUM <= case["seed"] <= verifier.MAX_SEED for case in cases)
    for case in cases:
        if case["group"] == "anchor":
            assert case["sink_mode"] == "upstream" and case["gated"] is True
        else:
            assert case["sink_mode"] in ("none", "finite")
        if case["group"] in ("multi", "pattern", *runtime.STREAM_GROUPS):
            assert case["gated"] is True
    for group in final_generator.HALF_GATED_GROUPS:
        members = [case for case in cases if case["group"] == group]
        assert sum(case["gated"] for case in members) == (len(members) + 1) // 2
    # A plan of the wrong size is refused.
    from suite_fixtures import fake_plan

    with pytest.raises(RuntimeError):
        final_generator.generate_manifest(fake_plan()[:-1])


EDGE_FIELD_MUTATIONS: tuple[tuple[str, Any], ...] = (
    ("family", "m1_64"),
    ("s_q", 0),
    ("s_q", 70001),
    ("s_kv", 0),
    ("s_kv", (1 << 20) + 1),
    ("topk", 64),
    ("topk", 200),
    ("topk", 4096),
    ("h_q", 0),
    ("h_q", 24),
    ("h_q", 4096),
    ("h_q", True),
    ("sm_scale", 0.0),
    ("sm_scale", float("inf")),
    ("sm_scale", True),
    ("bias", "random"),
    ("index_mode", "clamp"),
    ("invalid_rows", "most"),
    ("sink_mode", "nan"),
    ("length_mode", "full"),
    ("duplicates", 1),
    ("sanitize", "yes"),
    ("seed", -1),
    ("seed", 1 << 63),
)


@pytest.mark.parametrize("field,value", EDGE_FIELD_MUTATIONS)
def test_edge_domain_matches_the_runtime(field: str, value: Any) -> None:
    _cases, edges = manifests()
    case = dict(next(edge for edge in edges if edge["native"] and edge["s_q"] == 62))
    runtime.validate_edge_case(case)
    case[field] = value
    with pytest.raises(RuntimeError):
        verifier.validate_edge_case(case)
    with pytest.raises(ValueError):
        runtime.validate_edge_case(case)


def test_edge_domain_admits_tensor_parallel_heads_long_contexts_and_causal_indices() -> None:
    _cases, edges = manifests()
    base = dict(next(edge for edge in edges if edge["native"] and edge["s_q"] == 62))
    heads = runtime.FAMILY_GEOMETRY[base["family"]][1]
    for case in (
        dict(base, h_q=16),
        dict(base, h_q=heads),
        dict(base, s_kv=1 << 20),
        dict(base, index_mode="causal"),
    ):
        runtime.validate_edge_case(case)
        verifier.validate_edge_case(case)


def test_contract_caps_are_enforced_by_both_sides() -> None:
    _cases, edges = manifests()
    base = dict(next(edge for edge in edges if edge["family"] == "v32" and edge["native"]))
    # v32 at topk 2048: s_q * topk <= 2**25 allows s_q <= 16384.
    case = dict(base, s_q=16385, topk=2048)
    with pytest.raises(ValueError):
        runtime.validate_edge_case(case)
    with pytest.raises(RuntimeError):
        verifier.validate_edge_case(case)
    # Inside the caps but not above 2**31 q elements: not a legal sanitizer-only case.
    case = dict(base, s_q=16384, topk=2048, native=False, sanitize=True)
    with pytest.raises(ValueError):
        runtime.validate_edge_case(case)
    with pytest.raises(RuntimeError):
        verifier.validate_edge_case(case)
    # ... and legal natively: q (2.25 GiB) and out (2 GiB) are inside the 3 GiB cap.
    runtime.validate_edge_case(dict(base, s_q=16384, topk=2048, native=True))
    verifier.validate_edge_case(dict(base, s_q=16384, topk=2048, native=True))
    # Above the 3 GiB native cap: 2 * 30000 * 128 * 576 bytes of q.
    case = dict(base, s_q=30000, topk=128, native=True)
    with pytest.raises(ValueError):
        runtime.validate_edge_case(case)
    with pytest.raises(RuntimeError):
        verifier.validate_edge_case(case)
    # The upstream s_q = 70000 corner is legal only as a sanitizer-only case.
    large = next(edge for edge in edges if not edge["native"])
    runtime.validate_edge_case(large)
    verifier.validate_edge_case(large)
    with pytest.raises(ValueError):
        runtime.validate_edge_case(dict(large, native=True))
    with pytest.raises(RuntimeError):
        verifier.validate_edge_case(dict(large, native=True))
    with pytest.raises(ValueError):
        runtime.validate_edge_case(dict(large, sanitize=False))


def _edge_mutation(name: str) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        cases = payload["cases"]
        if name == "missing_case":
            cases.pop(0)
        elif name == "duplicate_id":
            cases[1]["case_id"] = cases[0]["case_id"]
        elif name == "drop_sanitize_flag":
            next(case for case in cases if case["sanitize"] and case["native"])["sanitize"] = False
        elif name == "drop_large_s_q":
            large = next(case for case in cases if not case["native"])
            large.update(s_q=2048, native=True)
        elif name == "category_count":
            cases[0]["category"] = "odd_sizes"
        elif name == "drop_zero_bias":
            for case in cases:
                if case["category"] == "scored_precision" and case["s_kv"] == 49152:
                    case["bias"] = "upstream"
        elif name == "drop_scored_semantics":
            for case in cases:
                if case["category"] == "scored_semantics" and case["family"] == "v4_pro":
                    case["index_mode"] = "upstream"
        elif name == "drop_glm5_causal":
            for case in cases:
                if case["index_mode"] == "causal" and case["family"] == "glm5":
                    case["index_mode"] = "upstream"
        elif name == "drop_glm5_long_context":
            for case in cases:
                if case["family"] == "glm5" and case["s_kv"] >= runtime.FAMILY_MAX_CONTEXT["glm5"] - 1:
                    case["s_kv"] = 4097
        elif name == "drop_tp_heads":
            for case in cases:
                case.pop("h_q", None)
        elif name == "schema_version":
            payload["schema_version"] = 4
        elif name == "suite_id":
            payload["suite_id"] = "edge-public-v3"
        else:  # pragma: no cover
            raise AssertionError(name)

    return mutate


@pytest.mark.parametrize(
    "mutation",
    ["missing_case", "duplicate_id", "drop_sanitize_flag", "drop_large_s_q", "category_count",
     "drop_zero_bias", "drop_scored_semantics", "drop_glm5_causal", "drop_glm5_long_context",
     "drop_tp_heads", "schema_version", "suite_id"],
)
def test_edge_manifest_counts_flags_and_coverage_fail_closed(mutation: str) -> None:
    payload = build_edge_manifest()
    _edge_mutation(mutation)(payload)
    with pytest.raises(RuntimeError):
        verifier.validate_edge_manifest(payload)


def test_runtime_result_recomputes_groups_tiers_families_and_primary() -> None:
    result, cases, edges = build_runtime_result()
    primary, families, breakdown = verifier.validate_runtime_result(
        result, cases, edges, expected_trials=TRIALS, expected_warmups=WARMUPS
    )
    assert set(families) == set(runtime.FAMILIES)
    assert math.isclose(primary, result["primary_tflops"], rel_tol=1e-12)
    assert math.isclose(
        primary, math.exp(sum(math.log(v) for v in families.values()) / len(runtime.FAMILIES)), rel_tol=1e-12
    )
    assert set(breakdown) == {"tiers", "groups"}
    assert set(breakdown["tiers"]) == {f"{f}:{t}" for f in runtime.FAMILIES for t in runtime.TIERS}
    assert set(breakdown["groups"]) == {f"{f}:{g}" for f in runtime.FAMILIES for g in runtime.GROUPS}
    for family in runtime.FAMILIES:
        tiers = breakdown["tiers"]
        assert math.isclose(
            families[family], math.sqrt(tiers[f"{family}:evidence"] * tiers[f"{family}:sweep"]), rel_tol=1e-12
        )
        for tier in runtime.TIERS:
            members = [breakdown["groups"][f"{family}:{g}"] for g, t in runtime.GROUP_TIER.items() if t == tier]
            assert math.isclose(
                tiers[f"{family}:{tier}"], math.exp(sum(map(math.log, members)) / len(members)), rel_tol=1e-12
            )
    assert families == result["family_tflops"]


def test_a_stream_that_stopped_at_its_time_limit_scores_the_work_it_finished() -> None:
    result, cases, edges = build_runtime_result()
    row = next(row for row in result["cases"] if row["stream"] and row["calls_total"] > 1)
    run = row["calls_total"] - 1
    expected = next(case for case in cases if case["case_id"] == row["case_id"])
    positions = sum(s_q for s_q, _ in expected["calls"][:run]) * row["topk"] // 2
    seconds = runtime.STREAM_TIME_LIMIT_SECONDS + 0.5
    row.update(
        calls_run=run, truncated=True, stream_seconds=seconds, stream_positions=positions,
        checked_calls=verifier._expected_stream_checks(row["calls_total"], run),
        tflops=runtime.effective_flops(positions, row["h_q"], row["d_qk"]) / seconds / 1.0e12,
    )
    result["sentinel_checks"] -= 1
    result.update(runtime.aggregate_case_results(result["cases"]))
    verifier.validate_runtime_result(result, cases, edges, expected_trials=TRIALS, expected_warmups=WARMUPS)


def test_promoted_sanitizer_only_cases_must_be_run_and_reported() -> None:
    _cases, edges = manifests()
    large = next(edge["case_id"] for edge in edges if not edge["native"])
    result, cases, edges = build_runtime_result(promoted=(large,))
    verifier.validate_runtime_result(
        result, cases, edges, expected_trials=TRIALS, expected_warmups=WARMUPS, promoted=[large]
    )
    # A result that skipped the promoted case is rejected.
    with pytest.raises(RuntimeError):
        verifier.validate_runtime_result(
            build_runtime_result()[0], cases, edges,
            expected_trials=TRIALS, expected_warmups=WARMUPS, promoted=[large],
        )
    # And one that ran it while nothing was promoted is rejected too.
    with pytest.raises(RuntimeError):
        verifier.validate_runtime_result(
            result, cases, edges, expected_trials=TRIALS, expected_warmups=WARMUPS
        )


def _tamper(name: str) -> Callable[[dict[str, Any]], None]:
    def mutate(result: dict[str, Any]) -> None:
        first = result["cases"][0]
        assert first["group"] == "anchor"
        stream = next(row for row in result["cases"] if row["stream"])
        ungated = next(row for row in result["cases"] if not row["stream"] and not row["gated"])
        if name == "primary":
            result["primary_tflops"] *= 1.01
        elif name == "family":
            result["family_tflops"]["glm5"] *= 1.01
        elif name == "tier":
            result["tier_tflops"]["v32:sweep"] *= 1.01
        elif name == "group":
            result["group_tflops"]["v4_pro:replay"] *= 1.01
        elif name == "missing_group":
            del result["group_tflops"]["v4_flash:length"]
        elif name == "trial_tflops":
            first["trial_tflops"][0] *= 2.0
        elif name == "trial_clock":
            first["trial_us"][0] *= 0.5
        elif name == "positions":
            first["trial_positions"][0] -= 1
        elif name == "sm_scale":
            first["sm_scale"] = 0.25
        elif name == "geometry":
            first["h_q"] = 64
        elif name == "tier_label":
            first["tier"] = "sweep"
        elif name == "decoys":
            first["decoy_precalls"] = 0
        elif name == "checked_calls":
            ungated["checked_calls"] = 1
        elif name == "cos_diff":
            first["out_cos_diff"] = 8.0e-6
        elif name == "missing_metrics":
            del first["lse_cos_diff"]
        elif name == "stream_kind":
            stream["stream"] = False
        elif name == "stream_truncated_early":
            stream["calls_run"] -= 1
            stream["truncated"] = True
        elif name == "stream_flag":
            stream["truncated"] = True
        elif name == "stream_checks":
            stream["checked_calls"] -= 1
        elif name == "stream_warmups":
            stream["warmup_calls"] = 0
        elif name == "stream_positions":
            stream["stream_positions"] *= 4
        elif name == "stream_tflops":
            stream["tflops"] *= 1.01
        elif name == "stream_metrics":
            stream["out_cos_diff"] = 8.0e-6
        elif name == "lse_cos_diff":
            result["edge"]["results"][3]["lse_cos_diff"] = 2.0e-7
        elif name == "edge_failed":
            result["edge"]["cases_failed"] = 1
        elif name == "edge_order":
            rows = result["edge"]["results"]
            rows[0], rows[1] = rows[1], rows[0]
        elif name == "protocol":
            result["protocol"]["l2_flush_bytes"] = 0
        elif name == "stream_protocol":
            result["protocol"]["stream_warm_calls"] = 0
        elif name == "provenance":
            result["generator_provenance"] = "flashmla-15f13e5-derived-sparse-prefill-v1-judge-port"
        elif name == "sentinel":
            result["sentinel_checks"] = 3
        elif name == "gate":
            result["gates"]["memory_safety_ok"] = None
        elif name == "hardware":
            result["hardware"]["total_memory_bytes"] = 16 * 1024**3
        elif name == "missing_case":
            result["cases"].pop()
        elif name == "trials":
            result["trials_per_case"] = 7
        else:  # pragma: no cover
            raise AssertionError(name)

    return mutate


@pytest.mark.parametrize(
    "mutation",
    ["primary", "family", "tier", "group", "missing_group", "trial_tflops", "trial_clock", "positions",
     "sm_scale", "geometry", "tier_label", "decoys", "checked_calls", "cos_diff", "missing_metrics",
     "stream_kind", "stream_truncated_early", "stream_flag", "stream_checks", "stream_warmups",
     "stream_positions", "stream_tflops", "stream_metrics", "lse_cos_diff", "edge_failed", "edge_order",
     "protocol", "stream_protocol", "provenance", "sentinel", "gate", "hardware", "missing_case", "trials"],
)
def test_runtime_result_tampering_fails_closed(mutation: str) -> None:
    result, cases, edges = build_runtime_result()
    verifier.validate_runtime_result(result, cases, edges, expected_trials=TRIALS, expected_warmups=WARMUPS)
    tampered = copy.deepcopy(result)
    _tamper(mutation)(tampered)
    with pytest.raises(RuntimeError):
        verifier.validate_runtime_result(
            tampered, cases, edges, expected_trials=TRIALS, expected_warmups=WARMUPS
        )


def test_case_above_the_physical_bound_is_rejected() -> None:
    result, cases, edges = build_runtime_result()
    first = result["cases"][0]
    scale = 2000.0 / first["tflops"]
    for key in ("trial_us", "trial_host_us", "trial_event_us"):
        first[key] = [value / scale for value in first[key]]
    first["trial_tflops"] = [value * scale for value in first["trial_tflops"]]
    first["median_us"] /= scale
    first["tflops"] *= scale
    result.update(runtime.aggregate_case_results(result["cases"]))
    with pytest.raises(RuntimeError, match="physical bound"):
        verifier.validate_runtime_result(
            result, cases, edges, expected_trials=TRIALS, expected_warmups=WARMUPS
        )


def test_stream_above_the_physical_bound_is_rejected() -> None:
    result, cases, edges = build_runtime_result()
    stream = next(row for row in result["cases"] if row["stream"])
    scale = 2000.0 / stream["tflops"]
    stream["stream_seconds"] /= scale
    stream["tflops"] *= scale
    result.update(runtime.aggregate_case_results(result["cases"]))
    with pytest.raises(RuntimeError, match="physical bound"):
        verifier.validate_runtime_result(
            result, cases, edges, expected_trials=TRIALS, expected_warmups=WARMUPS
        )


def test_effective_flops_is_the_upstream_formula() -> None:
    # 2 * s_q * topk * h_q * (d_qk + d_v): v32 at s_q 4096 is 2.34e12 FLOPs per call.
    assert runtime.effective_flops(4096 * 2048, 128, 576) == 2.0 * 4096 * 2048 * 128 * (576 + 512)
    assert math.isclose(runtime.effective_flops(4096 * 2048, 128, 576), 2.336e12, rel_tol=1e-3)
    assert math.isclose(runtime.effective_flops(4096 * 2048, 64, 576), 1.168e12, rel_tol=1e-3)
    assert math.isclose(runtime.effective_flops(4096 * 512, 64, 512), 2.75e11, rel_tol=1e-2)
    assert math.isclose(runtime.effective_flops(4096 * 1024, 128, 512), 1.10e12, rel_tol=1e-2)
    assert math.isclose(runtime.effective_tflops(4096 * 2048, 128, 576, 0.01), 233.6, rel_tol=1e-3)


def test_logit_abs_limit_is_upstream_at_half_and_scales_above() -> None:
    assert runtime.logit_abs_tol(0.5) == runtime.LOGIT_ABS_TOL
    assert runtime.logit_abs_tol(576 ** -0.55) == runtime.LOGIT_ABS_TOL
    assert runtime.logit_abs_tol(4.0) == pytest.approx(8 * runtime.LOGIT_ABS_TOL)
