"""Synthetic scored and edge suites and a matching runtime result, without GPUs.

The final manifest is drawn by ``scripts/generate_final_manifest.py`` from a
plan written here: the anchors are the disclosed upstream performance shapes
(``runtime.anchor_grid()``), and every other row is a small made-up shape, so
nothing here depends on (or discloses) the frozen Judge-only plan, its shapes
or its seeds.  The edge manifest is the edge generator's output for a test
seed.  The runtime result is shaped and aggregated the way the runtime builds
one (``_case_record``, ``aggregate_case_results``), so the Judge's recomputation
is exercised end to end.
"""
from __future__ import annotations

import json
import random
from typing import Any, Sequence

from conftest import load_module


verifier = load_module("suite_fixtures_verifier", "images/judge/verify_sparse_mla.py")
runtime = load_module("suite_fixtures_runtime", "runtime/sparse_mla_prefill_runtime.py")
final_generator = load_module("suite_fixtures_final_generator", "scripts/generate_final_manifest.py")
edge_generator = load_module("suite_fixtures_edge_generator", "scripts/generate_edge_manifest.py")

TEST_FINAL_SEED = 20261108
TEST_EDGE_SEED = 20261006
H100 = {
    "name": "NVIDIA H100 NVL",
    "count": 1,
    "capability": [9, 0],
    "uuid": "GPU-00000000-0000-0000-0000-000000000000",
    "total_memory_bytes": 95_830 * 1024**2,
}


def fake_plan() -> list[dict[str, Any]]:
    """A plan with the frozen design's group counts, every family in every group.

    Non-anchor sizes are synthetic and small (they only have to satisfy the
    contract); families rotate so each group has cases of all four.
    """
    rows: list[dict[str, Any]] = []
    for family, s_kv in runtime.anchor_grid():
        rows.append({"group": "anchor", "family": family, "h_q": runtime.FAMILY_GEOMETRY[family][1],
                     "s_q": runtime.ANCHOR_S_Q, "s_kv": s_kv})
    families = runtime.FAMILIES
    for group in runtime.GROUPS:
        if group == "anchor":
            continue
        for number in range(verifier.FINAL_GROUP_COUNTS[group]):
            family = families[number % len(families)]
            heads = runtime.FAMILY_GEOMETRY[family][1]
            row: dict[str, Any] = {"group": group, "family": family, "h_q": heads}
            if group == "sweep":
                row.update(s_q=128 * (1 + number % 4), s_kv=4096 + 1024 * (number // 4))
            elif group == "heads":
                row.update(h_q=heads // 2 if number % 2 else 16, s_q=256, s_kv=2048 + 512 * number)
            elif group == "length":
                row.update(s_q=512, s_kv=8192 + 64 * number, length=f"topk/{(2, 4, 8)[number % 3]}",
                           ragged=bool(number // 3 % 2))
            elif group == "pattern":
                row.update(s_q=256 * (1 + number % 2), s_kv=4096 + 256 * number,
                           index_pattern=runtime.INDEX_PATTERNS[number // 4 % len(runtime.INDEX_PATTERNS)])
            elif group == "multi":
                row.update(s_q=64, s_kv=0, request_lengths=[1000 + number, 2000, 37])
            elif group == "trace":
                row.update(s_q=0, s_kv=0, call_list=[[64, 1024 + number], [128, 2048 + number],
                                                     [256, 4096 + number]])
            elif group == "replay":  # one context in equal chunks: one context in equal chunks
                row.update(s_q=0, s_kv=0, call_list=[[512, 512 * (k + 1)] for k in range(3 + number % 3)])
            else:
                raise AssertionError(f"fake_plan has no rows for group {group!r}")
            rows.append(row)
    return rows


def build_final_manifest() -> dict[str, Any]:
    payload = final_generator.generate_manifest(fake_plan(), random.Random(TEST_FINAL_SEED))
    return json.loads(json.dumps(payload))


def build_edge_manifest() -> dict[str, Any]:
    payload = edge_generator.generate(verifier.EDGE_SUITE_ID, public=False, seed=TEST_EDGE_SEED)
    return json.loads(json.dumps(payload))


def manifests() -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    return (
        verifier.validate_final_manifest(build_final_manifest()),
        verifier.validate_edge_manifest(build_edge_manifest()),
    )


def accuracy() -> dict[str, float]:
    return {
        "out_max_abs": 3.9e-4,
        "out_max_rel": 0.4,
        "out_cos_diff": 1.5e-6,
        "max_logits_max_abs": 2.0e-7,
        "max_logits_max_rel": 1.0e-5,
        "max_logits_cos_diff": -1.0e-12,
        "lse_max_abs": 3.0e-7,
        "lse_max_rel": 2.0e-5,
        "lse_cos_diff": 2.0e-11,
    }


def single_row(case: dict[str, Any], index: int, trials: int) -> dict[str, Any]:
    scored = runtime.validate_scored_case(case)
    ceiling = scored["s_q"] * scored["topk"]
    if scored["group"] == "anchor":
        positions = ceiling
    elif scored["group"] == "length":
        positions = ceiling // scored["length_divisor"]
    else:
        positions = ceiling // 2 + index
    trial_us, host_us, event_us, trial_tflops, trial_positions = [], [], [], [], []
    for ordinal in range(trials):
        seconds = 1.0e-2 * (1 + index % 7) * (ordinal + 1)
        trial_us.append(seconds * 1.0e6)
        host_us.append(seconds * 1.0e6)
        event_us.append(seconds * 0.9e6)
        trial_tflops.append(runtime.effective_tflops(positions, scored["h_q"], scored["d_qk"], seconds))
        trial_positions.append(positions)
    return {
        **runtime._case_record(scored),
        "trial_us": trial_us,
        "trial_host_us": host_us,
        "trial_event_us": event_us,
        "trial_tflops": trial_tflops,
        "trial_positions": trial_positions,
        "median_us": runtime._median(trial_us),
        "tflops": runtime._median(trial_tflops),
        "decoy_precalls": trials,
        "checked_calls": 1 if scored["gated"] else 0,
        **(accuracy() if scored["gated"] else {}),
    }


def stream_row(case: dict[str, Any], index: int) -> dict[str, Any]:
    scored = runtime.validate_scored_case(case)
    calls = scored["calls"]
    positions = sum(s_q for s_q, _s_kv in calls) * scored["topk"] // 2
    seconds = 1.0e-2 * (1 + index % 5) * len(calls)
    work = runtime.effective_flops(positions, scored["h_q"], scored["d_qk"])
    checked = sum(
        1 for number in range(len(calls))
        if number % runtime.STREAM_CHECK_EVERY == 0 or number == len(calls) - 1
    )
    return {
        **runtime._case_record(scored),
        "calls_total": len(calls),
        "calls_run": len(calls),
        "truncated": False,
        "warmup_calls": runtime.STREAM_WARM_CALLS,
        "stream_seconds": seconds,
        "stream_positions": positions,
        "tflops": work / seconds / 1.0e12,
        "checked_calls": checked,
        **accuracy(),
    }


def build_runtime_result(
    trials: int = verifier.EXPECTED_TRIALS,
    warmups: int = verifier.EXPECTED_WARMUPS,
    promoted: Sequence[str] = (),
    manifests_: tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """A passing schema-3 runtime result for the synthetic suites."""
    cases, edges = manifests_ if manifests_ is not None else manifests()
    rows = [
        stream_row(case, index) if case["group"] in runtime.STREAM_GROUPS else single_row(case, index, trials)
        for index, case in enumerate(cases)
    ]
    native = [edge for edge in edges if edge["native"] or edge["case_id"] in promoted]
    singles = sum(1 for row in rows if not row["stream"])
    stream_calls = sum(runtime.STREAM_WARM_CALLS + row["calls_run"] for row in rows if row["stream"])
    result = {
        "schema_version": runtime.SCHEMA_VERSION,
        "generator_provenance": runtime.GENERATOR_PROVENANCE,
        "entrypoints": list(runtime.ENTRYPOINTS),
        "hardware": dict(H100),
        "trials_per_case": trials,
        "warmups": warmups,
        "protocol": runtime.protocol_record(runtime.plausibility_bound(H100["capability"])),
        "edge": {
            "cases_total": len(native),
            "cases_promoted": sorted(promoted),
            "cases_skipped_sanitize_only": len(edges) - len(native),
            "cases_passed": len(native),
            "cases_failed": 0,
            "results": [
                {"case_id": edge["case_id"], "category": edge["category"], "ok": True, **accuracy()}
                for edge in native
            ],
        },
        "cases": rows,
        "candidate_sha256": "c" * 64,
        "status": "passed",
        "failed_gate": None,
        "private_error": "",
        "gates": {name: True for name in runtime.GATE_ORDER},
        "sentinel_checks": 1 + runtime.EDGE_CALLS * len(native) + singles * (warmups + 2 * trials)
        + stream_calls,
        **runtime.aggregate_case_results(rows),
    }
    # The Judge only ever sees the worker's JSON file.
    return json.loads(json.dumps(result, allow_nan=False)), cases, edges
