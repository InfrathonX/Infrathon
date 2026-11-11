from __future__ import annotations

import copy
import json
import math
import statistics

import pytest

from conftest import TASK_ROOT, load_module, judge_only_file


verifier = load_module("standalone_v4_private_verifier", "images/judge/verify_grouped.py")


def _manifests():
    final = verifier.validate_final_manifest(
        json.loads(judge_only_file("judge/private/final_cases.json").read_text(encoding="utf-8"))
    )
    edges = verifier.validate_edge_manifest(
        json.loads(judge_only_file("judge/private/edge_cases.json").read_text(encoding="utf-8"))
    )
    return final, edges


def _single_row(case: dict, case_index: int, trials: int) -> dict:
    trial_us, host_us, event_us, trial_tflops, trial_m_sizes, trial_valid_m = [], [], [], [], [], []
    for ordinal in range(trials):
        m_sizes = verifier.draw_m_sizes(case, verifier.trial_seed(case["seed"], ordinal))
        # About 100 TFLOPS, varied per case and trial, always below the physical bound.
        measured_us = 2.0 * sum(m_sizes) * case["n"] * case["k"] / 1.0e8 * (1.0 + 0.001 * case_index) * (ordinal + 1)
        value = 2.0 * sum(m_sizes) * case["n"] * case["k"] / (measured_us * 1.0e-6) / 1.0e12
        trial_us.append(measured_us)
        host_us.append(measured_us)
        event_us.append(measured_us * 0.9)
        trial_tflops.append(value)
        trial_m_sizes.append(m_sizes)
        trial_valid_m.append(sum(m_sizes))
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "layout": case["layout"],
        "entrypoint": verifier.ENTRYPOINTS[case["layout"]],
        "g": case["g"],
        "m_max": case["m_max"],
        "expected_m": case["expected_m"],
        "expected_m_hint": verifier.case_hint(case),
        "n": case["n"],
        "k": case["k"],
        "dist": case["dist"],
        "trial_us": trial_us,
        "trial_host_us": host_us,
        "trial_event_us": event_us,
        "trial_tflops": trial_tflops,
        "trial_m_sizes": trial_m_sizes,
        "trial_valid_m": trial_valid_m,
        "median_us": float(statistics.median(trial_us)),
        "tflops": float(statistics.median(trial_tflops)),
        "max_exact_error": 1.4e-6,
        "max_source_error": 6.9e-4,
        "decoy_precalls": trials,
    }


def _sequence_row(case: dict, case_index: int, rounds: int) -> dict:
    round_us, host_us, event_us, round_flops, round_tflops, round_valid_m, checked = [], [], [], [], [], [], []
    shapes = ((case["n"], case["k"]), (case["n2"], case["k2"]))
    for ordinal in range(rounds):
        calls = verifier.sequence_schedule(case, verifier.trial_seed(case["seed"], ordinal))
        flops = math.fsum(2.0 * sum(sizes) * shapes[linear][0] * shapes[linear][1] for linear, _l, sizes, _a, _o in calls)
        measured_us = flops / 1.0e8 * (1.0 + 0.001 * case_index) * (ordinal + 1)
        round_us.append(measured_us)
        host_us.append(measured_us)
        event_us.append(measured_us * 0.95)
        round_flops.append(flops)
        round_tflops.append(flops / (measured_us * 1.0e-6) / 1.0e12)
        round_valid_m.append([sum(sizes) for _linear, _l, sizes, _a, _o in calls])
        checked.append(len({(linear, out) for linear, _l, _s, _a, out in calls}))
    return {
        "case_id": case["case_id"],
        "family": "sequence",
        "layout": case["layout"],
        "entrypoint": verifier.ENTRYPOINTS[case["layout"]],
        "g": case["g"],
        "m_max": case["m_max"],
        "expected_m": case["expected_m"],
        "n": case["n"],
        "k": case["k"],
        "n2": case["n2"],
        "k2": case["k2"],
        "dist": case["dist"],
        "calls_per_round": verifier.SEQUENCE_LAYERS * 2,
        "hints": verifier.sequence_hints(case),
        "round_us": round_us,
        "round_host_us": host_us,
        "round_event_us": event_us,
        "round_flops": round_flops,
        "round_tflops": round_tflops,
        "round_valid_m": round_valid_m,
        "checked_calls": checked,
        "median_us": float(statistics.median(round_us)),
        "tflops": float(statistics.median(round_tflops)),
        "max_exact_error": 1.4e-6,
        "max_source_error": 0.0,
    }


_CACHE: dict = {}


def _runtime_result(trials: int = 2, rounds: int = 1) -> tuple[dict, tuple, tuple]:
    key = (trials, rounds)
    if key in _CACHE:
        return copy.deepcopy(_CACHE[key])
    cases, edges = _manifests()
    rows = []
    family_values: dict[str, list[float]] = {family: [] for family in verifier.FAMILIES}
    for case_index, case in enumerate(cases):
        row = (
            _sequence_row(case, case_index, rounds)
            if case["family"] == "sequence"
            else _single_row(case, case_index, trials)
        )
        family_values[case["family"]].append(row["tflops"])
        rows.append(row)
    families = {family: verifier.geometric_mean(values) for family, values in family_values.items()}
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
                {"case_id": edge["case_id"], "ok": True, "max_exact_error": 1.0e-6, "max_source_error": 0.0}
                for edge in edges
            ],
        },
        "family_tflops": families,
        "primary_tflops": verifier.geometric_mean(list(families.values())),
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


def _mutations():
    def trial_clock(result):
        result["cases"][0]["trial_us"][0] *= 0.5

    def edge_failed(result):
        result["edge"]["cases_failed"] = 1

    def edge_missing(result):
        result["edge"]["results"].pop()

    def edge_error(result):
        result["edge"]["results"][3]["max_exact_error"] = 1.0e-3

    def exact_error(result):
        result["cases"][1]["max_exact_error"] = 6.0e-5

    def source_error(result):
        result["cases"][2]["max_source_error"] = 1.0e-3

    def no_decoy(result):
        result["cases"][0]["decoy_precalls"] = 0

    def gate_false(result):
        result["gates"]["memory_safety_ok"] = False

    def failed_status(result):
        result["status"] = "failed"

    def redrawn_rows(result):
        row = result["cases"][0]
        row["trial_m_sizes"][0][0] += 1
        row["trial_valid_m"][0] += 1

    def sequence_rows(result):
        row = next(item for item in result["cases"] if item["family"] == "sequence")
        row["round_valid_m"][0][3] += 1

    def sequence_work(result):
        row = next(item for item in result["cases"] if item["family"] == "sequence")
        row["round_flops"][0] *= 1.01

    def sequence_checks(result):
        row = next(item for item in result["cases"] if item["family"] == "sequence")
        row["checked_calls"][0] -= 1

    def sequence_hints(result):
        row = next(item for item in result["cases"] if item["family"] == "sequence")
        row["hints"][0] += 1

    def wrong_hint(result):
        result["cases"][0]["expected_m_hint"] += 1

    def rounds(result):
        result["rounds_per_sequence"] = 2

    def family_aggregate(result):
        result["family_tflops"]["latency"] *= 1.01

    return [trial_clock, edge_failed, edge_missing, edge_error, exact_error, source_error,
            no_decoy, gate_false, failed_status, redrawn_rows, sequence_rows, sequence_work,
            sequence_checks, sequence_hints, wrong_hint, rounds, family_aggregate]


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
    runtime = load_module("scoring_test_runtime", "runtime/grouped_fp8_runtime.py")
    assert runtime.FP8_DENSE_PEAK_BOUND_TFLOPS == verifier.FP8_DENSE_PEAK_BOUND_TFLOPS
    assert runtime.EXACT_ERROR_LIMIT == verifier.EXACT_ERROR_LIMIT == 5.0e-5
    assert runtime.SOURCE_ERROR_LIMIT == verifier.SOURCE_ERROR_LIMIT == 1.0e-3
    assert verifier.plausibility_bound([9, 0]) == 2500.0
    assert verifier.plausibility_bound([11, 7]) is None


def test_case_above_the_physical_bound_is_rejected() -> None:
    result, cases, _edges = _runtime_result()
    for family in ("decode", "sequence"):
        index = _first(cases, family)
        row = result["cases"][index]
        assert verifier._validate_case_result(row, cases[index], 2, 1, None) == row["tflops"]
        with pytest.raises(RuntimeError, match="physical bound"):
            verifier._validate_case_result(row, cases[index], 2, 1, row["tflops"] * 0.5)


def test_verifier_and_runtime_draw_identical_rows_and_schedules() -> None:
    runtime = load_module("scoring_draw_runtime", "runtime/grouped_fp8_runtime.py")
    cases, _edges = _manifests()
    for case in cases[::37]:
        for seed in (case["seed"], verifier.trial_seed(case["seed"], 4), 12345):
            assert verifier.draw_m_sizes(case, seed) == runtime.draw_m_sizes(case, seed)
        if case["family"] == "sequence":
            assert verifier.sequence_schedule(case, 7) == runtime.sequence_schedule(case, 7)
            assert verifier.sequence_hints(case) == runtime.sequence_hints(case)
        else:
            assert verifier.case_hint(case) == runtime.case_hint(case)
    for name in ("SEQUENCE_LAYERS", "SEQUENCE_WEIGHT_POOL", "SEQUENCE_A_POOL", "SEQUENCE_OUT_POOL",
                 "MAX_HINT", "MAX_CAPACITY_BYTES", "MIN_DEVICE_MEMORY_BYTES", "TRIAL_SEED_STRIDE",
                 "DECOY_SEED_SALT", "WARMUP_SEED_SALT", "SEQUENCE_HINT_SALT", "FAMILIES", "MAX_M_MAX"):
        assert getattr(verifier, name) == getattr(runtime, name), name
