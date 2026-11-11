from __future__ import annotations

import copy
import json
import math
import statistics

import pytest

from conftest import load_module, judge_only_file


verifier = load_module("standalone_v3_private_verifier", "images/judge/verify_wgrad.py")


def _manifests():
    final = verifier.validate_final_manifest(
        json.loads(judge_only_file("judge/private/final_cases.json").read_text(encoding="utf-8"))
    )
    edges = verifier.validate_edge_manifest(
        json.loads(judge_only_file("judge/private/edge_cases.json").read_text(encoding="utf-8"))
    )
    return final, edges


def _single_row(case: dict, case_index: int, trials: int) -> dict:
    trial_us, host_us, event_us, trial_tflops, trial_ks, trial_total_k = [], [], [], [], [], []
    for ordinal in range(trials):
        ks = verifier.draw_ks(case, verifier.derive_seed(case["seed"], "ks", ordinal))
        flops = 2.0 * sum(ks) * case["m"] * case["n"]
        # About 100 TFLOPS, varied per case and trial, always below the physical bound.
        measured_us = flops / 1.0e8 * (1.0 + 0.001 * case_index) * (ordinal + 1)
        trial_us.append(measured_us)
        host_us.append(measured_us)
        event_us.append(measured_us * 0.9)
        trial_tflops.append(flops / (measured_us * 1.0e-6) / 1.0e12)
        trial_ks.append(ks)
        trial_total_k.append(sum(ks))
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "model": case["model"],
        "entrypoint": verifier.WGRAD_ENTRYPOINT,
        "g": case["g"],
        "m": case["m"],
        "n": case["n"],
        "total_k": case["total_k"],
        "dist": case["dist"],
        "expected_k_hint": verifier.case_hint(case),
        "trial_us": trial_us,
        "trial_host_us": host_us,
        "trial_event_us": event_us,
        "trial_tflops": trial_tflops,
        "trial_ks": trial_ks,
        "trial_total_k": trial_total_k,
        "median_us": float(statistics.median(trial_us)),
        "tflops": float(statistics.median(trial_tflops)),
        "max_element_ratio": 2.8e-3,
        "max_exact_error": 7.4e-9,
        "max_source_error": 5.9e-4,
        "decoy_precalls": trials,
    }


def _sequence_row(case: dict, case_index: int, rounds: int) -> dict:
    round_us, host_us, event_us, round_flops, round_tflops, checked = [], [], [], [], [], []
    shapes = ((case["m"], case["n"]), (case["m2"], case["n2"]))
    for ordinal in range(rounds):
        calls = verifier.sequence_schedule(case, verifier.derive_seed(case["seed"], "sequence-round", ordinal))
        flops = math.fsum(2.0 * sum(ks) * shapes[linear][0] * shapes[linear][1] for linear, _l, ks, _o in calls)
        measured_us = flops / 1.0e8 * (1.0 + 0.001 * case_index) * (ordinal + 1)
        round_us.append(measured_us)
        host_us.append(measured_us)
        event_us.append(measured_us * 0.95)
        round_flops.append(flops)
        round_tflops.append(flops / (measured_us * 1.0e-6) / 1.0e12)
        checked.append(len({(linear, out) for linear, _l, _ks, out in calls}))
    return {
        "case_id": case["case_id"],
        "family": "sequence",
        "model": case["model"],
        "entrypoint": verifier.WGRAD_ENTRYPOINT,
        "g": case["g"],
        "m": case["m"],
        "n": case["n"],
        "m2": case["m2"],
        "n2": case["n2"],
        "total_k": case["total_k"],
        "dist": case["dist"],
        "expected_k_hint": verifier.case_hint(case),
        "calls_per_round": verifier.SEQUENCE_LAYERS * 2,
        "round_us": round_us,
        "round_host_us": host_us,
        "round_event_us": event_us,
        "round_flops": round_flops,
        "round_tflops": round_tflops,
        "checked_calls": checked,
        "median_us": float(statistics.median(round_us)),
        "tflops": float(statistics.median(round_tflops)),
        "max_element_ratio": 0.0,
        "max_exact_error": 7.4e-9,
        "max_source_error": 0.0,
    }


_CACHE: dict = {}


def _runtime_result(trials: int = 2, rounds: int = 1) -> tuple[dict, tuple, tuple]:
    key = (trials, rounds)
    if key in _CACHE:
        return copy.deepcopy(_CACHE[key])
    cases, edges = _manifests()
    rows = []
    values: dict[str, dict[str, list[float]]] = {family: {} for family in verifier.FAMILIES}
    for case_index, case in enumerate(cases):
        row = (
            _sequence_row(case, case_index, rounds)
            if case["family"] == "sequence"
            else _single_row(case, case_index, trials)
        )
        values[case["family"]].setdefault(case["model"], []).append(row["tflops"])
        rows.append(row)
    primary, families = verifier.aggregate_families(values)
    result = {
        "schema_version": 4,
        "status": "passed",
        "gates": {name: True for name in verifier.RUNTIME_GATES},
        "hardware": {"name": "NVIDIA H100", "count": 1, "capability": [9, 0], "total_memory_bytes": 85899345920},
        "entrypoints": list(verifier.ENTRYPOINTS.values()),
        "trials_per_case": trials,
        "warmups": 1,
        "rounds_per_sequence": rounds,
        "protocol": {"decoy_precall": True},
        "edge": {
            "cases_total": len(edges),
            "cases_passed": len(edges),
            "cases_failed": 0,
            "results": [
                {"case_id": edge["case_id"], "ok": True, "max_element_ratio": 1.0e-3, "max_exact_error": 1.0e-8,
                 "max_source_error": 0.0}
                for edge in edges
            ],
        },
        "family_tflops": families,
        "primary_tflops": primary,
        "cases": rows,
    }
    _CACHE[key] = (result, cases, edges)
    return copy.deepcopy(_CACHE[key])


def _first(cases, family: str) -> int:
    return next(index for index, case in enumerate(cases) if case["family"] == family)


def test_geometric_mean() -> None:
    assert math.isclose(verifier.geometric_mean([4.0, 9.0]), 6.0)


def test_runtime_result_recomputes_trials_rounds_families_and_primary() -> None:
    result, cases, edges = _runtime_result()
    score, families = verifier.validate_runtime_result(
        result, cases, edges, expected_trials=2, expected_rounds=1, expected_warmups=1
    )
    assert math.isclose(score, result["primary_tflops"])
    assert families == result["family_tflops"]
    assert set(families) == set(verifier.FAMILIES)


def test_family_scores_weight_every_model_equally() -> None:
    values = {
        "large": {"a": [100.0, 100.0, 100.0, 100.0], "b": [400.0]},
        "many_groups": {"a": [50.0]},
        "small_batch": {"a": [10.0]},
        "sequence": {"a": [80.0]},
    }
    primary, families = verifier.aggregate_families(values)
    assert math.isclose(families["large"], 200.0)
    assert math.isclose(primary, verifier.geometric_mean([200.0, 50.0, 10.0, 80.0]))
    runtime = load_module("scoring_aggregate_runtime", "runtime/k_grouped_wgrad_runtime.py")
    rows = [{"family": family, "model": model, "tflops": value}
            for family, models in values.items() for model, cases in models.items() for value in cases]
    aggregated = runtime.aggregate_case_results(rows)
    assert math.isclose(aggregated["primary_tflops"], primary)
    assert aggregated["family_tflops"] == pytest.approx(families)


def _mutations():
    def trial_clock(result):
        result["cases"][0]["trial_us"][0] *= 0.5

    def edge_failed(result):
        result["edge"]["cases_failed"] = 1

    def edge_missing(result):
        result["edge"]["results"].pop()

    def edge_error(result):
        result["edge"]["results"][3]["max_exact_error"] = 1.0e-3

    def edge_element(result):
        result["edge"]["results"][5]["max_element_ratio"] = 0.2

    def exact_error(result):
        result["cases"][1]["max_exact_error"] = 3.0e-7

    def element_ratio(result):
        result["cases"][3]["max_element_ratio"] = 0.06

    def source_error(result):
        result["cases"][2]["max_source_error"] = 1.0e-3

    def no_decoy(result):
        result["cases"][0]["decoy_precalls"] = 0

    def gate_false(result):
        result["gates"]["memory_safety_ok"] = False

    def failed_status(result):
        result["status"] = "failed"

    def redrawn_ks(result):
        row = result["cases"][0]
        row["trial_ks"][0][0] += 128
        row["trial_ks"][0][1] -= 128

    def total_k_mismatch(result):
        result["cases"][1]["trial_total_k"][0] += 128

    def hint_changed(result):
        result["cases"][2]["expected_k_hint"] += 1

    def model_changed(result):
        result["cases"][2]["model"] = "other"

    def sequence_work(result):
        row = next(item for item in result["cases"] if item["family"] == "sequence")
        row["round_flops"][0] *= 1.01

    def sequence_checks(result):
        row = next(item for item in result["cases"] if item["family"] == "sequence")
        row["checked_calls"][0] -= 1

    def rounds(result):
        result["rounds_per_sequence"] = 2

    def family_aggregate(result):
        result["family_tflops"]["small_batch"] *= 1.01

    return [trial_clock, edge_failed, edge_missing, edge_error, edge_element, exact_error, element_ratio,
            source_error, no_decoy, gate_false, failed_status, redrawn_ks, total_k_mismatch, hint_changed,
            model_changed, sequence_work, sequence_checks, rounds, family_aggregate]


@pytest.mark.parametrize("mutation", _mutations(), ids=lambda fn: fn.__name__)
def test_runtime_result_tampering_fails_closed(mutation) -> None:
    result, cases, edges = _runtime_result()
    mutated = copy.deepcopy(result)
    mutation(mutated)
    with pytest.raises(RuntimeError):
        verifier.validate_runtime_result(
            mutated, cases, edges, expected_trials=2, expected_rounds=1, expected_warmups=1
        )


def test_plausibility_bounds_match_the_runtime() -> None:
    runtime = load_module("scoring_test_runtime", "runtime/k_grouped_wgrad_runtime.py")
    assert runtime.FP8_DENSE_PEAK_BOUND_TFLOPS == verifier.FP8_DENSE_PEAK_BOUND_TFLOPS
    assert runtime.EXACT_ERROR_LIMIT == verifier.EXACT_ERROR_LIMIT == 2.0e-7
    assert runtime.ELEMENT_RATIO_LIMIT == verifier.ELEMENT_RATIO_LIMIT == 0.05
    assert runtime.EDGE_MAX_OUTPUT_ELEMENTS == verifier.EDGE_MAX_OUTPUT_ELEMENTS
    assert runtime.EDGE_MAX_INPUT_BYTES == verifier.EDGE_MAX_INPUT_BYTES
    assert runtime.MAX_TOTAL_K == verifier.MAX_TOTAL_K == 262_144
    assert runtime.SOURCE_ERROR_LIMIT == verifier.SOURCE_ERROR_LIMIT == 1.0e-3
    assert verifier.plausibility_bound([9, 0]) == 2500.0
    assert verifier.plausibility_bound([11, 7]) is None


def test_case_above_the_physical_bound_is_rejected() -> None:
    result, cases, _edges = _runtime_result()
    for family in ("large", "sequence"):
        index = _first(cases, family)
        row = result["cases"][index]
        assert verifier._validate_case_result(row, cases[index], 2, 1, None) == row["tflops"]
        with pytest.raises(RuntimeError, match="physical bound"):
            verifier._validate_case_result(row, cases[index], 2, 1, row["tflops"] * 0.5)


def test_verifier_and_runtime_draw_identical_group_sizes_and_schedules() -> None:
    runtime = load_module("scoring_draw_runtime", "runtime/k_grouped_wgrad_runtime.py")
    cases, _edges = _manifests()
    for case in cases[::37]:
        for seed in (case["seed"], verifier.derive_seed(case["seed"], "ks", 4), 12345):
            assert verifier.draw_ks(case, seed) == runtime.draw_ks(case, seed)
        assert verifier.case_hint(case) == runtime.case_hint(case)
        if case["family"] == "sequence":
            assert verifier.sequence_schedule(case, 7) == runtime.sequence_schedule(case, 7)
    for name in ("SEQUENCE_LAYERS", "SEQUENCE_OUT_POOL", "MAX_HINT", "MAX_TOTAL_K", "MAX_OUTPUT_ELEMENTS",
                 "MIN_DEVICE_MEMORY_BYTES", "FAMILIES", "DISTRIBUTIONS"):
        assert getattr(verifier, name) == getattr(runtime, name), name
    for role in ("ks", "decoy-ks", "sequence-round"):
        assert verifier.derive_seed(5, role, 2) == runtime.derive_seed(5, role, 2)
