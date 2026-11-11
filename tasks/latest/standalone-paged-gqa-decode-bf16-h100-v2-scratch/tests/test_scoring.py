"""Judge-side manifest validation and score recomputation, without GPUs.

The manifests and runtime results here are synthetic, built from the disclosed
grid and the runtime's own helpers, so these tests do not depend on the frozen
private instance (its own tests validate it through the same functions).
"""
from __future__ import annotations

import copy
import json
import math
from typing import Any, Callable

import pytest

from conftest import load_module


verifier = load_module("paged_gqa_v2_private_verifier_scoring", "images/judge/verify_paged_gqa.py")
runtime = load_module("paged_gqa_v2_runtime_scoring", "runtime/paged_gqa_decode_runtime.py")

TRIALS = 2
WARMUPS = 1
H100 = {
    "name": "NVIDIA H100 NVL",
    "count": 1,
    "capability": [9, 0],
    "uuid": "GPU-00000000-0000-0000-0000-000000000000",
    "total_memory_bytes": 95_830 * 1024**2,
}

# Valid native edge templates, one per hazard the verifier requires the
# private instance to keep exercising (verify_paged_gqa.EDGE_REQUIRED_COVERAGE).
EDGE_TEMPLATES: tuple[dict[str, Any], ...] = (
    {"category": "batch_one", "s_q": 1, "b": 1, "l_max": 37, "h_q": 32, "h_kv": 8, "page_size": 16},
    {"category": "fa3_layout", "s_q": 3, "b": 3, "l_max": 129, "h_q": 6, "h_kv": 3, "page_size": 4,
     "width_factor": 3},
    {"category": "group_seven", "s_q": 16, "b": 2, "l_max": 500, "h_q": 28, "h_kv": 4, "page_size": 16},
    {"category": "vllm_pool", "s_q": 1, "b": 5, "l_max": 97, "h_q": 16, "h_kv": 1, "page_size": 1,
     "table_mode": "random", "pool_pages": 2048},
    {"category": "page_256", "s_q": 2, "b": 2, "l_max": 523, "h_q": 8, "h_kv": 2, "page_size": 256},
    {"category": "shared_prefix", "s_q": 1, "b": 4, "l_max": 339, "h_q": 32, "h_kv": 8, "page_size": 16,
     "table_mode": "shared"},
    {"category": "group_twelve", "s_q": 5, "b": 3, "l_max": 100, "h_q": 96, "h_kv": 8, "page_size": 16},
    {"category": "nan_slots", "s_q": 4, "b": 3, "l_max": 127, "h_q": 64, "h_kv": 8, "page_size": 64,
     "fill": "nan"},
    {"category": "negative_logits", "s_q": 1, "b": 2, "l_max": 54, "h_q": 12, "h_kv": 2, "page_size": 8,
     "values": "negative_logits"},
    {"category": "long_context", "s_q": 1, "b": 1, "l_max": 131073, "h_q": 8, "h_kv": 1, "page_size": 256},
)
LARGE_OFFSET = {
    "category": "large_offset", "s_q": 1, "b": 2, "l_max": 64, "h_q": 8, "h_kv": 8, "page_size": 16,
    "table_mode": "tail_perm", "pool_pages": 135_000,
}


def edge_case(template: dict[str, Any], case_id: str, seed: int, *, sanitize: bool, native: bool = True) -> dict[str, Any]:
    spec = dict(template)
    s_q, b, l_max, page_size = spec["s_q"], spec["b"], spec["l_max"], spec["page_size"]
    width = -(-l_max // page_size) * spec.pop("width_factor", 1)
    return {
        "case_id": case_id,
        "category": spec["category"],
        "s_q": s_q,
        "b": b,
        "l_max": l_max,
        "h_q": spec["h_q"],
        "h_kv": spec["h_kv"],
        "page_size": page_size,
        "lengths": [max(s_q, l_max - 3 * row) for row in range(b)],
        "table_mode": spec.get("table_mode", "perm"),
        "pool_pages": spec.get("pool_pages", b * width),
        "table_width": width,
        "fill": spec.get("fill", "randn"),
        "values": spec.get("values", "randn"),
        "seed": seed,
        "sanitize": sanitize,
        "native": native,
    }


def build_edge_manifest() -> dict[str, Any]:
    cases = [
        edge_case(
            EDGE_TEMPLATES[index % len(EDGE_TEMPLATES)],
            f"edge-{index:03d}-{EDGE_TEMPLATES[index % len(EDGE_TEMPLATES)]['category']}",
            (1 << 40) + index,
            sanitize=index < verifier.EDGE_SANITIZE_COUNT - verifier.EDGE_SANITIZE_ONLY_COUNT,
        )
        for index in range(verifier.EDGE_CASE_COUNT - verifier.EDGE_SANITIZE_ONLY_COUNT)
    ]
    cases.append(edge_case(LARGE_OFFSET, "edge-large-offset", 7, sanitize=True, native=False))
    return {"schema_version": verifier.SCHEMA_VERSION, "suite_id": verifier.EDGE_SUITE_ID, "cases": cases}


STREAM_EVERY = 37  # every 37th synthetic case is a sequence


def call_case(index: int) -> dict[str, Any]:
    family = ("decode", "spec", "long")[index % 3]
    s_q = 1 if family != "spec" else 2 + index % 5
    page_size = (16, 1, 64, 2, 256)[index % 5]
    b = 1 + index % 4
    if family == "long":
        lengths = [40_000 + 97 * index] + [s_q + 50 * row for row in range(1, b)]
    else:
        lengths = [max(s_q, 300 + (37 * index + 211 * row) % 1500) for row in range(b)]
    # Requests 0 and 1 share their first block when they are long enough; every
    # sixth case shares at page granularity (the PagedAttention sharing workloads).
    block_tokens = max(16, page_size) if index % 6 == 1 else 512
    blocks = [[] for _ in range(b)]
    shared = [row for row in range(min(b, 2)) if lengths[row] >= block_tokens]
    for row in shared:
        blocks[row] = [index % 7]
    needed = -(-max(lengths) // page_size)
    case = {
        "case_id": f"final-{index:03d}-{family}",
        "family": family,
        "s_q": s_q,
        "h_q": (32, 28, 8, 12)[index % 4],
        "h_kv": (8, 4, 1, 2)[index % 4],
        "page_size": page_size,
        "block_tokens": block_tokens,
        "lengths": lengths,
        "blocks": blocks,
        "table_width": needed + index % 3,
        "pool_pages": 0,
        "fill": "nan" if index % 4 == 3 else "randn",
        "values": "outlier" if index % 5 == 2 else "randn",
        "seed": (1 << 62) + 1009 * index,
    }
    case["pool_pages"] = runtime.call_page_demand(lengths, blocks, page_size, block_tokens) + index % 5
    return case


def stream_case(index: int) -> dict[str, Any]:
    page_size = (16, 1, 32)[index % 3]
    requests = [
        [0, 6, 600 + index, [100 + index]],
        [1, 4, 520, [100 + index]],
        [2, 6, 40 + index % 9, []],
        [0, 2, 700, []],
        [4, 6, 3000, [200 + index, 201 + index]],
    ]
    case = {
        "case_id": f"final-{index:03d}-stream",
        "family": "stream",
        "s_q": 1,
        "h_q": 16,
        "h_kv": 2,
        "page_size": page_size,
        "block_tokens": 512,
        "values": "outlier" if index % 2 else "randn",
        "steps": 6,
        "requests": requests,
        "table_width": 0,
        "pool_pages": 0,
        "seed": (1 << 62) + 1009 * index,
    }
    plan = runtime.stream_plan(case)
    case["table_width"] = -(-max(plan.finals) // page_size)
    case["pool_pages"] = plan.peak_pages + 3
    return case


def build_final_manifest() -> dict[str, Any]:
    cases = [
        stream_case(index) if index % STREAM_EVERY == STREAM_EVERY - 1 else call_case(index)
        for index in range(verifier.FINAL_CASE_COUNT)
    ]
    return {"schema_version": verifier.SCHEMA_VERSION, "suite_id": verifier.FINAL_SUITE_ID, "cases": cases}


def manifests() -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    return (
        verifier.validate_final_manifest(build_final_manifest()),
        verifier.validate_edge_manifest(build_edge_manifest()),
    )


def _accuracy() -> dict[str, float]:
    return {
        "out_max_abs": 3.9e-3,
        "out_max_abs_limit": 7.8e-3,
        "out_mean_abs": 1.2e-4,
        "out_mean_abs_limit": 2.0e-4,
        "out_norm_l2": 1.1e-3,
    }


def _common(scored: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": scored["case_id"],
        "family": scored["family"],
        "scored": True,
        "entrypoint": runtime.DECODE_ENTRYPOINT,
        **{name: scored[name] for name in ("s_q", "b", "l_max", "h_q", "h_kv", "page_size",
                                           "table_width", "pool_pages")},
        "softmax_scale": runtime.SOFTMAX_SCALE,
    }


def build_runtime_result(trials: int = TRIALS, warmups: int = WARMUPS) -> tuple[dict, tuple, tuple]:
    """A passing schema-4 runtime result, shaped and aggregated as the runtime does."""
    cases, edges = manifests()
    rows = []
    checks = 0
    for index, case in enumerate(cases):
        scored = runtime.validate_scored_case({key: case[key] for key in (
            verifier.STREAM_KEYS if case["family"] == "stream" else verifier.CALL_KEYS)})
        pairs = scored["total_pairs"]
        if scored["family"] == "stream":
            steps = scored["steps"]
            step_us = [20.0 + 3.0 * ((index + step) % 5) for step in range(steps)]
            seconds = runtime._fsum(step_us) * 1.0e-6
            rows.append({
                **_common(scored),
                "steps": steps,
                "step_us": step_us,
                "total_pairs": pairs,
                "seconds": seconds,
                "tflops": runtime.effective_tflops(pairs, scored["h_q"], seconds),
                "max_step_us": max(step_us),
                "median_step_us": runtime._median(step_us),
                "warmup_calls": runtime.STREAM_WARMUPS,
                "golden_checks": sum(1 for step in range(steps)
                                     if step % runtime.STREAM_CHECK_EVERY == 0 or step == steps - 1),
                **_accuracy(),
            })
            checks += runtime.STREAM_WARMUPS + steps
            continue
        trial_us, host_us, event_us, trial_tflops = [], [], [], []
        for ordinal in range(trials):
            seconds = 1.0e-4 * (index % 11 + 1) * (ordinal + 1)
            trial_us.append(seconds * 1.0e6)
            host_us.append(seconds * 1.0e6)
            event_us.append(seconds * 0.9e6)
            trial_tflops.append(runtime.effective_tflops(pairs, scored["h_q"], seconds))
        median_us = runtime._median(trial_us)
        rows.append({
            **_common(scored),
            "trial_us": trial_us,
            "trial_host_us": host_us,
            "trial_event_us": event_us,
            "trial_tflops": trial_tflops,
            "trial_pairs": [pairs] * trials,
            "median_us": median_us,
            "tflops": runtime._median(trial_tflops),
            "kv_gbps": 4 * scored["h_kv"] * 128 * sum(scored["lengths"]) / (median_us * 1.0e-6) / 1.0e9,
            "decoy_precalls": trials,
            **_accuracy(),
        })
        checks += warmups + 2 * trials
    native = [edge for edge in edges if edge["native"]]
    result = {
        "schema_version": runtime.SCHEMA_VERSION,
        "generator_provenance": runtime.GENERATOR_PROVENANCE,
        "entrypoints": list(runtime.ENTRYPOINTS),
        "hardware": dict(H100),
        "trials_per_case": trials,
        "warmups": warmups,
        "protocol": {
            "decoy_precall": True,
            "stream_warmups": runtime.STREAM_WARMUPS,
            "stream_check_every": runtime.STREAM_CHECK_EVERY,
            "stream_time_limit_seconds": runtime.STREAM_TIME_LIMIT_SECONDS,
            "edge_calls": 2,
            "guard_bytes": runtime.GUARD_BYTES,
            "soft_landing": {"page_table": runtime.SOFT_PAGE_ID, "cache_seqlens": "s_q"},
            "max_abs_mult": runtime.MAX_ABS_MULT,
            "mean_abs_mult": runtime.MEAN_ABS_MULT,
            "max_abs_floor": runtime.MAX_ABS_FLOOR,
            "norm_l2_tol": runtime.NORM_L2_TOL,
            "plausibility_bound_tflops": runtime.plausibility_bound(H100["capability"]),
        },
        "edge": {
            "cases_total": len(native),
            "cases_skipped_sanitize_only": len(edges) - len(native),
            "cases_passed": len(native),
            "cases_failed": 0,
            "results": [
                {"case_id": edge["case_id"], "category": edge["category"], "ok": True, **_accuracy()}
                for edge in native
            ],
        },
        "cases": rows,
        "candidate_sha256": "c" * 64,
        "status": "passed",
        "failed_gate": None,
        "private_error": "",
        "gates": {name: True for name in runtime.GATE_ORDER},
        "sentinel_checks": 1 + 2 * len(native) + checks,
        **runtime.aggregate_case_results(rows),
    }
    # The Judge only ever sees the worker's JSON file.
    return json.loads(json.dumps(result, allow_nan=False)), cases, edges


# --------------------------------------------------------------------------
# The Judge's duplicated tables equal the trusted runtime's.
# --------------------------------------------------------------------------
def test_judge_tables_match_the_runtime() -> None:
    for name in (
        "HEAD_DIM", "SOFTMAX_SCALE", "FAMILIES", "CALL_FAMILIES", "BLOCK_SIZES", "SCORED_FILLS",
        "SCORED_VALUES", "OUTLIER_PROBABILITY", "OUTLIER_SCALE",
        "MAX_STREAM_STEPS", "MAX_STREAM_REQUESTS", "STREAM_WARMUPS", "STREAM_CHECK_EVERY",
        "STREAM_TIME_LIMIT_SECONDS", "MAX_SCORED_POOL_PAGES", "MAX_SCORED_KV_BYTES",
        "MAX_STREAM_TABLE_ELEMENTS", "MAX_SCORED_WIDTH_FACTOR", "MAX_B",
        "MAX_S_Q", "MAX_H_Q", "MAX_L_MAX", "PAGE_SIZES", "TABLE_MODES", "FILLS", "VALUES",
        "MAX_POOL_PAGES", "MAX_WIDTH_FACTOR", "MAX_OUTPUT_ELEMENTS", "MAX_CASE_KV_BYTES",
        "MAX_SANITIZE_ONLY_KV_BYTES", "MAX_SEED", "MAX_ABS_MULT", "MEAN_ABS_MULT", "MAX_ABS_FLOOR",
        "NORM_L2_TOL", "GUARD_BYTES", "SCHEMA_VERSION", "GENERATOR_PROVENANCE", "DECODE_ENTRYPOINT",
        "CALL_KEYS", "STREAM_KEYS", "EDGE_KEYS", "MIN_COMPUTE_CAPABILITY", "MIN_DEVICE_MEMORY_BYTES",
        "BF16_DENSE_PEAK_BOUND_TFLOPS",
    ):
        assert getattr(verifier, name) == getattr(runtime, name), name
    assert set(verifier.RUNTIME_GATES) == set(runtime.GATE_ORDER)
    assert verifier.SOFT_LANDING["page_table"] == runtime.SOFT_PAGE_ID


def test_judge_recomputes_stream_work_and_pages_like_the_runtime() -> None:
    for index in range(STREAM_EVERY - 1, verifier.FINAL_CASE_COUNT, STREAM_EVERY):
        case = stream_case(index)
        plan = runtime.stream_plan(case)
        judged = verifier._validate_stream(case, case["case_id"])
        assert judged["total_pairs"] == plan.total_pairs
        assert judged["b"] == max(len(now) for now in plan.active)
        tight = dict(case, pool_pages=plan.peak_pages)
        verifier._validate_stream(tight, case["case_id"])
        with pytest.raises(RuntimeError):
            verifier._validate_stream(dict(case, pool_pages=plan.peak_pages - 1), case["case_id"])
        with pytest.raises(ValueError):
            runtime.validate_scored_case(dict(case, pool_pages=plan.peak_pages - 1))


def test_plausibility_bounds_match_the_runtime() -> None:
    assert verifier.BF16_DENSE_PEAK_BOUND_TFLOPS == runtime.BF16_DENSE_PEAK_BOUND_TFLOPS
    assert verifier.plausibility_bound([9, 0]) == runtime.plausibility_bound([9, 0]) == 1200.0
    assert verifier.plausibility_bound([8, 0]) == 400.0
    assert verifier.plausibility_bound([11, 7]) is None
    assert verifier.plausibility_bound(None) is None


def test_geometric_mean() -> None:
    assert math.isclose(verifier.geometric_mean([4.0, 9.0]), 6.0)
    with pytest.raises(RuntimeError):
        verifier.geometric_mean([4.0, 0.0])


# --------------------------------------------------------------------------
# Manifests.
# --------------------------------------------------------------------------
def test_synthetic_manifests_satisfy_both_the_judge_and_the_runtime() -> None:
    cases, edges = manifests()
    assert len(cases) == verifier.FINAL_CASE_COUNT == 1114
    assert {case["family"] for case in cases} == set(verifier.FAMILIES)
    for case in cases:
        keys = verifier.STREAM_KEYS if case["family"] == "stream" else verifier.CALL_KEYS
        raw = {key: case[key] for key in keys}
        checked = runtime.validate_scored_case(raw)
        assert checked["total_pairs"] == case["total_pairs"]
        assert checked["b"] == case["b"] and checked["l_max"] == case["l_max"]
    assert len(edges) == verifier.EDGE_CASE_COUNT
    assert sum(edge["sanitize"] for edge in edges) == verifier.EDGE_SANITIZE_COUNT
    assert [edge["case_id"] for edge in edges if not edge["native"]] == ["edge-large-offset"]
    for edge in edges:
        runtime.validate_edge_case(edge)


def _first(payload: dict, family: str) -> dict:
    return next(case for case in payload["cases"] if case["family"] == family)


def _final_mutations() -> list[Callable[[dict], None]]:
    def schema(payload):
        payload["schema_version"] = 3

    def suite(payload):
        payload["suite_id"] = "final-holdout-v2"

    def extra_field(payload):
        payload["generator"] = "x"

    def short(payload):
        payload["cases"].pop()

    def v2_grid_case(payload):
        payload["cases"][0] = {"case_id": "final-m1-q1", "family": "q1", "cell": "M1", "l_max": 2048,
                               "seed": 1 << 62}

    def extra_case_field(payload):
        payload["cases"][0]["cell"] = "M1"

    def public_seed(payload):
        payload["cases"][3]["seed"] = 99013

    def seed_too_large(payload):
        payload["cases"][3]["seed"] = 1 << 63

    def repeated_seed(payload):
        payload["cases"][4]["seed"] = payload["cases"][5]["seed"]

    def repeated_case_id(payload):
        payload["cases"][4]["case_id"] = payload["cases"][5]["case_id"]

    def bad_case_id(payload):
        payload["cases"][0]["case_id"] = "Final C1"

    def unknown_family(payload):
        payload["cases"][0]["family"] = "prefill"

    def missing_family(payload):
        for case in payload["cases"]:
            if case["family"] == "long":
                case["family"] = "decode"

    def spec_at_one_query(payload):
        _first(payload, "spec")["s_q"] = 1

    def decode_at_two_queries(payload):
        case = _first(payload, "decode")
        case["s_q"] = 2

    def length_above_domain(payload):
        case = _first(payload, "long")
        case["lengths"][0] = (1 << 20) + 1

    def bool_length(payload):
        _first(payload, "decode")["lengths"][0] = True

    def blocks_longer_than_request(payload):
        case = _first(payload, "decode")
        case["blocks"][0] = list(range(100))

    def repeated_block(payload):
        case = next(case for case in payload["cases"] if case["family"] != "stream" and case["blocks"][0])
        case["blocks"][0] = case["blocks"][0] * 2

    def pool_below_demand(payload):
        case = _first(payload, "decode")
        case["pool_pages"] = runtime.call_page_demand(case["lengths"], case["blocks"], case["page_size"],
                                                      case["block_tokens"]) - 1

    def pool_far_above_demand(payload):
        case = _first(payload, "decode")
        case["pool_pages"] = 3 * case["pool_pages"] + 10_000

    def narrow_table(payload):
        case = _first(payload, "decode")
        case["table_width"] = -(-max(case["lengths"]) // case["page_size"]) - 1

    def stream_step_without_requests(payload):
        case = _first(payload, "stream")
        case["requests"] = [request for request in case["requests"] if request[0] != 0 or request[1] > 3]
        case["requests"] = [[3, 6, 50, []]]

    def stream_window_past_end(payload):
        _first(payload, "stream")["requests"][0][1] = 7

    def stream_pool_below_peak(payload):
        case = _first(payload, "stream")
        case["pool_pages"] = runtime.stream_plan(case).peak_pages - 1

    def stream_unknown_field(payload):
        _first(payload, "stream")["lengths"] = [1]

    def block_size_not_a_power(payload):
        _first(payload, "decode")["block_tokens"] = 48

    def block_smaller_than_page(payload):
        case = next(case for case in payload["cases"] if case["family"] != "stream" and case["page_size"] == 256)
        case["block_tokens"] = 128

    def unknown_values(payload):
        _first(payload, "spec")["values"] = "uniform"

    def missing_values(payload):
        _first(payload, "stream").pop("values")

    return [schema, suite, extra_field, short, v2_grid_case, extra_case_field, public_seed,
            seed_too_large, repeated_seed, repeated_case_id, bad_case_id, unknown_family,
            missing_family, spec_at_one_query, decode_at_two_queries, length_above_domain,
            bool_length, blocks_longer_than_request, repeated_block, pool_below_demand,
            pool_far_above_demand, narrow_table, stream_step_without_requests,
            stream_window_past_end, stream_pool_below_peak, stream_unknown_field,
            block_size_not_a_power, block_smaller_than_page, unknown_values, missing_values]


@pytest.mark.parametrize("mutation", _final_mutations(), ids=lambda fn: fn.__name__)
def test_final_manifest_fails_closed(mutation) -> None:
    payload = build_final_manifest()
    verifier.validate_final_manifest(payload)
    mutation(payload)
    with pytest.raises(RuntimeError):
        verifier.validate_final_manifest(payload)


# Single-case mutations outside the runtime's domain: the Judge and the
# runtime must both refuse them.
EDGE_DOMAIN_MUTATIONS: dict[str, Callable[[dict], None]] = {
    "extra_field": lambda case: case.update(scale=1.0),
    "missing_field": lambda case: case.pop("fill"),
    "s_q_17": lambda case: case.update(s_q=17),
    "b_257": lambda case: case.update(b=257, lengths=case["lengths"] * 257),
    "h_q_129": lambda case: case.update(h_q=129),
    "indivisible_heads": lambda case: case.update(h_q=12, h_kv=8),
    "page_size_3": lambda case: case.update(page_size=3),
    "l_max_above_domain": lambda case: case.update(l_max=(1 << 20) + 1),
    "wrong_length_count": lambda case: case["lengths"].pop(),
    "length_below_s_q": lambda case: case.update(s_q=2, lengths=[1] * case["b"]),
    "length_above_l_max": lambda case: case["lengths"].__setitem__(0, case["l_max"] + 1),
    "bool_length": lambda case: case["lengths"].__setitem__(0, True),
    "narrow_table": lambda case: case.update(table_width=-(-case["l_max"] // case["page_size"]) - 1),
    "table_wider_than_4x": lambda case: case.update(
        table_width=4 * -(-case["l_max"] // case["page_size"]) + 1,
        pool_pages=case["b"] * (4 * -(-case["l_max"] // case["page_size"]) + 1),
    ),
    "pool_below_table": lambda case: case.update(pool_pages=case["b"] * case["table_width"] - 1),
    "pool_above_cap": lambda case: case.update(pool_pages=(1 << 18) + 1),
    "unknown_mode": lambda case: case.update(table_mode="strided"),
    "unknown_fill": lambda case: case.update(fill="zeros"),
    "unknown_values": lambda case: case.update(values="fp8"),
    "negative_seed": lambda case: case.update(seed=-1),
    "seed_above_63_bits": lambda case: case.update(seed=1 << 63),
    "string_flag": lambda case: case.update(sanitize="true"),
    "never_runs": lambda case: case.update(native=False, sanitize=False),
    "native_kv_above_cap": lambda case: case.update(
        h_q=8, h_kv=8, page_size=256, l_max=256, lengths=[256] * case["b"], table_width=1,
        table_mode="random", pool_pages=5000,
    ),
}


@pytest.mark.parametrize("name", sorted(EDGE_DOMAIN_MUTATIONS))
def test_edge_domain_matches_the_runtime(name: str) -> None:
    case = copy.deepcopy(build_edge_manifest()["cases"][1])
    verifier.validate_edge_case(case)
    runtime.validate_edge_case(case)
    EDGE_DOMAIN_MUTATIONS[name](case)
    with pytest.raises(RuntimeError):
        verifier.validate_edge_case(case)
    with pytest.raises(ValueError):
        runtime.validate_edge_case(case)


def _edge_manifest_mutations() -> list[Callable[[dict], None]]:
    def suite(payload):
        payload["suite_id"] = "edge-public-v4"

    def v2_identity(payload):
        payload["suite_id"] = "edge-holdout-v2"
        payload["schema_version"] = 3

    def no_long_context(payload):
        for case in payload["cases"]:
            if case["l_max"] > (1 << 17):
                case.update(l_max=4096, lengths=[4096] * case["b"],
                            table_width=16, pool_pages=case["b"] * 16)

    def short(payload):
        payload["cases"].pop(5)

    def repeated_id(payload):
        payload["cases"][3]["case_id"] = payload["cases"][4]["case_id"]

    def bad_category(payload):
        payload["cases"][3]["category"] = "Large Offset"

    def extra_sanitize(payload):
        next(case for case in payload["cases"] if not case["sanitize"])["sanitize"] = True

    def second_sanitize_only(payload):
        case = next(case for case in payload["cases"] if case["sanitize"] and case["native"])
        case.update(edge_case(LARGE_OFFSET, case["case_id"], 9, sanitize=True, native=False))

    def small_sanitize_only(payload):
        payload["cases"][-1].update(pool_pages=1000)

    def native_large_offset(payload):
        payload["cases"][-1]["native"] = True

    def no_nan_slots(payload):
        for case in payload["cases"]:
            if case["fill"] == "nan":
                case["fill"] = "inf"

    def no_vllm_pool(payload):
        for case in payload["cases"]:
            if case["table_mode"] == "random":
                case["pool_pages"] = case["b"] * case["table_width"]

    def only_power_of_two_groups(payload):
        for case in payload["cases"]:
            if not verifier._is_power_of_two(case["h_q"] // case["h_kv"]):
                case["h_q"] = case["h_kv"]

    return [suite, v2_identity, no_long_context, short, repeated_id, bad_category, extra_sanitize,
            second_sanitize_only, small_sanitize_only, native_large_offset, no_nan_slots, no_vllm_pool,
            only_power_of_two_groups]


@pytest.mark.parametrize("mutation", _edge_manifest_mutations(), ids=lambda fn: fn.__name__)
def test_edge_manifest_counts_flags_and_coverage_fail_closed(mutation) -> None:
    payload = build_edge_manifest()
    verifier.validate_edge_manifest(payload)
    mutation(payload)
    with pytest.raises(RuntimeError):
        verifier.validate_edge_manifest(payload)


# --------------------------------------------------------------------------
# Runtime result recomputation.
# --------------------------------------------------------------------------
def test_runtime_result_recomputes_trials_families_and_primary() -> None:
    result, cases, edges = build_runtime_result()
    score, families = verifier.validate_runtime_result(
        result, cases, edges, expected_trials=TRIALS, expected_warmups=WARMUPS
    )
    assert math.isclose(score, result["primary_tflops"], rel_tol=1.0e-12)
    assert set(families) == set(verifier.FAMILIES)
    expected = math.exp(sum(math.log(families[name]) for name in verifier.FAMILIES) / 4)
    assert math.isclose(score, expected, rel_tol=1.0e-12)
    for name, value in families.items():
        assert math.isclose(value, result["family_tflops"][name], rel_tol=1.0e-12)


def test_a_zero_mean_limit_is_an_exact_requirement_not_an_error() -> None:
    result, cases, edges = build_runtime_result()
    result["cases"][4].update(out_mean_abs=0.0, out_mean_abs_limit=0.0)
    result["edge"]["results"][2].update(out_mean_abs=0.0, out_mean_abs_limit=0.0)
    verifier.validate_runtime_result(result, cases, edges, expected_trials=TRIALS, expected_warmups=WARMUPS)
    result["edge"]["results"][2]["out_mean_abs"] = 1.0e-9
    with pytest.raises(RuntimeError, match="mean_abs"):
        verifier.validate_runtime_result(result, cases, edges, expected_trials=TRIALS, expected_warmups=WARMUPS)


def _set(path: tuple[Any, ...], value: Any) -> Callable[[dict], None]:
    def mutate(result: dict) -> None:
        target = result
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value(target[path[-1]]) if callable(value) else value

    mutate.__name__ = "/".join(str(key) for key in path)
    return mutate


def _row(result: dict, family: str, ordinal: int = 0) -> dict:
    return [row for row in result["cases"] if row["family"] == family][ordinal]


def _result_mutations() -> list[Callable[[dict], None]]:
    def trial_clock(result):
        _row(result, "decode")["trial_us"][0] *= 0.5

    def event_slower_than_reported(result):
        row = _row(result, "spec")
        row["trial_event_us"][1] = row["trial_us"][1] * 2.0

    def pairs_differ_from_manifest(result):
        _row(result, "long")["trial_pairs"][0] += 1

    def pairs_list_short(result):
        _row(result, "decode", 1)["trial_pairs"].pop()

    def inflated_case(result):
        _row(result, "decode", 2)["tflops"] *= 2.0

    def median_time(result):
        _row(result, "spec", 1)["median_us"] += 1.0

    def bool_page_size(result):
        next(row for row in result["cases"] if row["page_size"] == 1)["page_size"] = True

    def table_width(result):
        _row(result, "long", 1)["table_width"] += 1

    def pool_pages(result):
        _row(result, "decode", 3)["pool_pages"] += 1

    def stream_step_time(result):
        _row(result, "stream")["step_us"][0] *= 0.5

    def stream_missing_step(result):
        _row(result, "stream")["step_us"].pop()

    def stream_seconds(result):
        _row(result, "stream")["seconds"] *= 0.9

    def stream_pairs(result):
        _row(result, "stream")["total_pairs"] += 1

    def stream_tflops(result):
        _row(result, "stream", 1)["tflops"] *= 1.5

    def stream_max_step(result):
        _row(result, "stream")["max_step_us"] *= 0.5

    def stream_warmups(result):
        _row(result, "stream")["warmup_calls"] = 0

    def stream_golden_checks(result):
        _row(result, "stream")["golden_checks"] = 0

    def stream_over_time_limit(result):
        row = _row(result, "stream")
        row["step_us"][0] = (runtime.STREAM_TIME_LIMIT_SECONDS + 1.0) * 1.0e6
        row["seconds"] = math.fsum(row["step_us"]) * 1.0e-6
        row["max_step_us"] = max(row["step_us"])
        row["median_step_us"] = runtime._median(row["step_us"])
        row["tflops"] = runtime.effective_tflops(row["total_pairs"], row["h_q"], row["seconds"])

    def missing_case(result):
        result["cases"].pop()

    def extra_case(result):
        result["cases"].append(dict(result["cases"][0], case_id="final-extra"))

    def edge_reordered(result):
        rows = result["edge"]["results"]
        rows[0], rows[1] = rows[1], rows[0]

    def edge_missing(result):
        result["edge"]["results"].pop()

    def max_above_limit(result):
        row = _row(result, "decode", 4)
        row["out_max_abs"] = row["out_max_abs_limit"] * 1.01

    def limit_without_floor(result):
        _row(result, "decode", 4).update(out_max_abs=0.0, out_max_abs_limit=1.0e-6)

    def negative_error(result):
        result["edge"]["results"][7]["out_max_abs"] = -1.0

    return [
        trial_clock, event_slower_than_reported, pairs_differ_from_manifest, pairs_list_short,
        inflated_case, median_time, bool_page_size, table_width, pool_pages, stream_step_time,
        stream_missing_step, stream_seconds, stream_pairs, stream_tflops, stream_max_step,
        stream_warmups, stream_golden_checks, stream_over_time_limit, missing_case, extra_case,
        edge_reordered, edge_missing, max_above_limit, limit_without_floor, negative_error,
        _set(("schema_version",), 3),
        _set(("generator_provenance",), "flash-attention-0251105a-derived-paged-gqa-decode-port"),
        _set(("status",), "failed"),
        _set(("gates", "memory_safety_ok"), False),
        _set(("gates", "edge_correctness_ok"), None),
        _set(("trials_per_case",), TRIALS + 1),
        _set(("warmups",), WARMUPS + 1),
        _set(("entrypoints",), ["paged_gqa_decode_bf16", "helper"]),
        _set(("hardware", "capability"), [7, 5]),
        _set(("hardware", "count"), 2),
        _set(("hardware", "total_memory_bytes"), 16 * 1024**3),
        _set(("protocol", "decoy_precall"), False),
        _set(("protocol", "stream_warmups"), 1),
        _set(("protocol", "stream_time_limit_seconds"), 1.0e9),
        _set(("protocol", "edge_calls"), 1),
        _set(("protocol", "guard_bytes"), 0),
        _set(("protocol", "soft_landing"), {"page_table": 0xA5A5A5A5, "cache_seqlens": "s_q"}),
        _set(("protocol", "norm_l2_tol"), 1.0e-2),
        _set(("protocol", "plausibility_bound_tflops"), None),
        _set(("sentinel_checks",), lambda value: value - 1),
        _set(("sentinel_checks",), True),
        _set(("edge", "cases_failed"), 1),
        _set(("edge", "cases_total"), lambda value: value + 1),
        _set(("edge", "cases_skipped_sanitize_only"), 0),
        _set(("edge", "results", 3, "ok"), False),
        _set(("edge", "results", 4, "category"), "other"),
        _set(("edge", "results", 5, "out_norm_l2"), 6.0e-3),
        _set(("cases", 0, "decoy_precalls"), 0),
        _set(("cases", 1, "softmax_scale"), 0.1),
        _set(("cases", 3, "h_q"), 64),
        _set(("cases", 4, "b"), 99),
        _set(("cases", 5, "scored"), False),
        _set(("cases", 7, "out_norm_l2"), 5.1e-3),
        _set(("family_tflops", "decode"), lambda value: value * 1.01),
        _set(("family_tflops", "stream"), lambda value: value * 1.01),
        _set(("primary_tflops",), lambda value: value * 1.01),
    ]


@pytest.mark.parametrize("mutation", _result_mutations(), ids=lambda fn: fn.__name__)
def test_runtime_result_tampering_fails_closed(mutation) -> None:
    result, cases, edges = build_runtime_result()
    mutated = copy.deepcopy(result)
    mutation(mutated)
    with pytest.raises(RuntimeError):
        verifier.validate_runtime_result(mutated, cases, edges, expected_trials=TRIALS, expected_warmups=WARMUPS)


def test_case_above_the_physical_bound_is_rejected() -> None:
    result, cases, _edges = build_runtime_result()
    row = result["cases"][0]
    assert verifier._validate_case_result(row, cases[0], TRIALS, None) == row["tflops"]
    with pytest.raises(RuntimeError, match="physical bound"):
        verifier._validate_case_result(row, cases[0], TRIALS, row["tflops"] * 0.5)


def test_effective_flops_count_causal_pairs_per_query_head() -> None:
    # S_q query rows of a request with cache length L see S_q*L - S_q*(S_q-1)/2 keys.
    result, cases, _edges = build_runtime_result()
    by_id = {case["case_id"]: case for case in cases}
    row = next(row for row in result["cases"] if row["family"] == "spec")
    case = by_id[row["case_id"]]
    s_q = case["s_q"]
    pairs = row["trial_pairs"][0]
    assert pairs == sum(s_q * length - s_q * (s_q - 1) // 2 for length in case["lengths"])
    expected = 4.0 * 128 * row["h_q"] * pairs / (row["trial_us"][0] * 1.0e-6) / 1.0e12
    assert math.isclose(row["trial_tflops"][0], expected, rel_tol=1.0e-12)
