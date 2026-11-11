"""Verifier and adapter behaviour on synthetic records."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest

from conftest import TASK_ROOT, _load, judge_only_file

runtime = _load("kv_runtime_j", "runtime/kv_transfer_runtime.py")
verifier = _load("kv_verifier_j", "images/judge/verify_kv_transfer.py")
adapter = _load("kv_adapter_j", "judge/run_judge.py")

PUBLIC = tuple(
    verifier.validate_scored_case(case)
    for case in json.loads((TASK_ROOT / "images/judge/public_cases.json").read_text())["cases"]
)
# A same-run probe record as the trusted runtime would emit on an H100; any
# admitted device produces its own.
ENVELOPES = {
    "protocol": runtime.ENVELOPE_PROTOCOL,
    "probe_bytes": runtime.ENVELOPE_PROBE_BYTES,
    "repeats": runtime.ENVELOPE_PROBE_REPEATS,
    "margin": runtime.ENVELOPE_MARGIN,
    "device_gbps": 1456.0,
    "h2d_gbps": 55.2,
    "d2h_gbps": 55.4,
    "zerocopy_h2d_gbps": 33.1,
    "zerocopy_d2h_gbps": 52.7,
    "host_gbps": 55.4,
    "duplex_gbps": 74.7,
}


def _gm(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def clean_worker_result(cases=PUBLIC, suite_id=None) -> dict[str, Any]:
    edge_rows = [
        {"case_id": case_id, "arm": "device", "scored": False, "waves": 1, "correct": True}
        for case_id in verifier.EDGE_CASE_IDS
    ]
    rows = []
    family_values: dict[str, list[dict]] = {name: [] for name in runtime.FAMILIES}
    for case in cases:
        logical = verifier.case_logical_bytes(case)
        envelope = verifier.case_envelope_gbps(case, ENVELOPES)
        base = logical / (envelope * 0.5) / 1.0e9
        trials = int(case["trials"])
        trial_seconds = [base * (1.0 + 0.001 * ordinal) for ordinal in range(trials)]
        median = verifier.statistics_median(sorted(trial_seconds))
        gbps = logical / median / 1.0e9
        row = {
            "case_id": case["case_id"],
            "family": case["family"],
            "arm": case["arm"],
            "foreground": case["foreground"],
            "stratum": case["stratum"],
            "tier": case["tier"],
            "scored": True,
            "waves": len(case["waves"]),
            "trials": trials,
            "logical_bytes": logical,
            "trial_seconds": trial_seconds,
            "host_wall_seconds": list(trial_seconds),
            "event_seconds": [seconds * 0.98 for seconds in trial_seconds],
            "median_seconds": median,
            "gbps": gbps,
        }
        score = gbps
        if case["arm"] == "overlap":
            fg_alone = [0.5] * trials
            fg_overlap = [0.5 + 0.2 * seconds for seconds in trial_seconds]
            losses = [min(max((fg_overlap[i] - fg_alone[i]) / trial_seconds[i], 0.0),
                          runtime.MAX_OVERLAP_LOSS) for i in range(trials)]
            row.update({
                "alone_seconds": list(trial_seconds),
                "fg_overlap_seconds": fg_overlap,
                "fg_alone_seconds": fg_alone,
                "fg_iterations": [40] * trials,
                "fg_loss": losses,
                "fg_loss_median": verifier.statistics_median(sorted(losses)),
            })
            score = gbps * (1.0 - verifier.statistics_median(sorted(losses)))
        row["case_score"] = score
        family_values[case["family"]].append(row)
        rows.append(row)
    families = {name: runtime.family_score(values) for name, values in family_values.items()}
    groups = {group: _gm([families[n] for n in names])
              for group, names in runtime.GROUP_FAMILIES.items()}
    score = math.exp(math.fsum(runtime.GROUP_WEIGHTS[n] * math.log(groups[n])
                               for n in runtime.GROUP_WEIGHTS))
    return {
        "suite_id": suite_id or verifier.PUBLIC_SUITE_ID,
        "library_sha256": "0" * 64,
        "entrypoint": verifier.ENTRYPOINT,
        "hardware": {
            "device_name": "NVIDIA H100 80GB HBM3",
            "capability": "sm_90",
            "total_memory_bytes": 85_000_000_000,
            "multi_processor_count": 132,
            "torch_version": "2.8.0+cu129",
            "cuda_version": "12.9",
        },
        "envelopes": dict(ENVELOPES),
        "constants": dict(verifier.EXPECTED_CONSTANTS),
        "binary_report": {"ok": True, "undefined_symbols": [], "problems": []},
        "edge_fixtures": edge_rows,
        "cases": rows,
        "family_scores": families,
        "group_scores": groups,
        "group_weights": dict(runtime.GROUP_WEIGHTS),
        "score_gbps": score,
        "scored_cases": len(rows),
    }


def _validate(result, cases=PUBLIC, suite_id=None):
    return verifier.validate_runtime_result(
        result, cases, expected_suite_id=suite_id or verifier.PUBLIC_SUITE_ID)


def test_clean_result_validates() -> None:
    result = clean_worker_result()
    score, groups, families = _validate(result)
    assert math.isclose(score, result["score_gbps"], rel_tol=1e-12)
    assert set(groups) == set(runtime.GROUP_WEIGHTS)
    assert set(families) == set(runtime.FAMILIES)


def _first(result, family):
    return next(row for row in result["cases"] if row["family"] == family)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda r: r.__setitem__("suite_id", "other"),
        lambda r: r["hardware"].__setitem__("capability", "sm_75"),
        lambda r: r["hardware"].__setitem__("total_memory_bytes", 40_000_000_000),
        lambda r: r["envelopes"].__setitem__("device_gbps", 50.0),
        lambda r: r["envelopes"].__setitem__("margin", 5.0),
        lambda r: r["envelopes"].__setitem__("protocol", "frozen-h100"),
        lambda r: r.pop("envelopes"),
        lambda r: r["constants"].__setitem__("poison", 0),
        lambda r: r["constants"].__setitem__("max_overlap_loss", 0.5),
        lambda r: r["binary_report"].__setitem__("ok", False),
        lambda r: r["cases"][0].__setitem__("gbps", r["cases"][0]["gbps"] * 1.01),
        lambda r: r["cases"][0].__setitem__("case_score", r["cases"][0]["case_score"] * 1.01),
        lambda r: r["cases"][0].__setitem__("logical_bytes", 1),
        lambda r: r["cases"][0].__setitem__("median_seconds", 1e-9),
        lambda r: r["cases"][0].__setitem__("family", "trace"),
        lambda r: r["cases"][0].__setitem__("stratum", "elsewhere"),
        lambda r: r["envelopes"].__setitem__("duplex_gbps", 1.0),
        lambda r: _first(r, "overlap").__setitem__("fg_loss", [0.0] * 3),
        lambda r: _first(r, "overlap").__setitem__(
            "fg_overlap_seconds", [s * 0.5 for s in _first(r, "overlap")["fg_overlap_seconds"]]),
        lambda r: _first(r, "overlap").pop("fg_alone_seconds"),
        lambda r: _first(r, "overlap").__setitem__("fg_iterations", [0, 0, 0]),
        lambda r: r.__setitem__("score_gbps", r["score_gbps"] * 1.001),
        lambda r: r["family_scores"].__setitem__("trace", r["family_scores"]["trace"] * 1.1),
        lambda r: r["group_scores"].pop("overlap"),
        lambda r: r["edge_fixtures"].pop(),
        lambda r: r["edge_fixtures"][0].__setitem__("correct", False),
        lambda r: r["cases"].pop(),
        lambda r: r["group_weights"].__setitem__("device", 0.6),
    ),
)
def test_mutated_results_rejected(mutate) -> None:
    result = copy.deepcopy(clean_worker_result())
    mutate(result)
    with pytest.raises(RuntimeError):
        _validate(result)


def test_envelope_violation_rejected() -> None:
    result = copy.deepcopy(clean_worker_result())
    row = result["cases"][0]
    factor = 3.0
    for key in ("trial_seconds", "host_wall_seconds", "event_seconds"):
        row[key] = [s / factor for s in row[key]]
    row["median_seconds"] /= factor
    row["gbps"] *= factor
    row["case_score"] *= factor
    with pytest.raises(RuntimeError):
        _validate(result)


def test_overlap_loss_is_clamped() -> None:
    assert runtime.overlap_loss(2.0, 1.0, 0.1) == runtime.MAX_OVERLAP_LOSS
    assert runtime.overlap_loss(0.9, 1.0, 0.1) == 0.0
    assert math.isclose(runtime.overlap_loss(1.05, 1.0, 0.1), 0.5)


def _detail_from(result: dict[str, Any]) -> dict[str, Any]:
    groups = result["group_scores"]
    return {
        "schema_version": 1,
        "profile_ok": 1.0,
        "suite_ok": 1.0,
        "source_ok": 1.0,
        "hardware_ok": 1.0,
        "binary_ok": 1.0,
        "correctness_ok": 1.0,
        "edge_ok": 1.0,
        "envelope_ok": 1.0,
        "profile_code": 2.0,
        "score_gbps": result["score_gbps"],
        "device_gbps": groups["device"],
        "host_gbps": groups["host"],
        "trace_gbps": groups["trace"],
        "overlap_gbps": groups["overlap"],
        **{f"family_{name.replace('-', '_')}_gbps": value
           for name, value in result["family_scores"].items()},
        "final_version": 1.0,
        "num_candidate_versions": 1.0,
        "evaluation_profile": "final",
        "suite_id": verifier.FINAL_SUITE_ID,
        "manifest_sha256": "a" * 64,
        "evaluation_policy": adapter.EVALUATION_POLICY,
        "selection_protocol_id": adapter.SELECTION_PROTOCOL_ID,
        "case_count": float(verifier.EXPECTED_FINAL_COUNT),
        "edge_case_count": float(len(verifier.EDGE_CASE_IDS)),
        "reward": result["score_gbps"],
    }


def test_adapter_accepts_clean_detail() -> None:
    result = clean_worker_result()
    detail = _detail_from(result)
    reward = {key: value for key, value in detail.items() if isinstance(value, (int, float))}
    structured = adapter.build_structured_result(
        reward, detail, verifier_completed=True, verifier_returncode=0)
    assert structured["valid"], structured["summary"]
    assert math.isclose(structured["score"], result["score_gbps"], rel_tol=1e-12)
    for key in adapter.FAMILY_KEYS:
        assert key in structured["metrics"]


@pytest.mark.parametrize(
    "corrupt",
    (
        lambda d: d.__setitem__("score_gbps", d["score_gbps"] * 1.001),
        lambda d: d.__setitem__("device_gbps", d["device_gbps"] * 1.5),
        lambda d: d.__setitem__("overlap_gbps", d["overlap_gbps"] * 1.5),
        lambda d: d.__setitem__("family_trace_gbps", d["family_trace_gbps"] * 1.5),
        lambda d: d.pop("family_host_duplex_gbps"),
        lambda d: d.__setitem__("case_count", 20.0),
        lambda d: d.__setitem__("edge_case_count", 14.0),
        lambda d: d.__setitem__("suite_id", "lmcache-kv-transfer-final-holdout-v1"),
        lambda d: d.__setitem__("binary_ok", 0.0),
    ),
)
def test_adapter_rejects_corrupted_detail(corrupt) -> None:
    result = clean_worker_result()
    detail = _detail_from(result)
    corrupt(detail)
    reward = {key: value for key, value in detail.items() if isinstance(value, (int, float))}
    structured = adapter.build_structured_result(
        reward, detail, verifier_completed=True, verifier_returncode=0)
    assert not structured["valid"]
    assert structured["score"] == 0.0


def test_final_manifest_validates_and_scores() -> None:
    payload = json.loads(judge_only_file("judge/private/final_cases.json").read_text())
    excluded_shapes = frozenset(verifier.shape_identity(w) for c in PUBLIC for w in c["waves"])
    excluded_seeds = frozenset(c["seed"] for c in PUBLIC)
    cases = verifier.validate_final_manifest(payload, excluded_shapes, excluded_seeds)
    assert len(cases) == verifier.EXPECTED_FINAL_COUNT
    result = clean_worker_result(cases, verifier.FINAL_SUITE_ID)
    score, _groups, _families = _validate(result, cases, verifier.FINAL_SUITE_ID)
    assert score > 0.0


def test_final_manifest_rejects_a_public_shape() -> None:
    payload = copy.deepcopy(json.loads(judge_only_file("judge/private/final_cases.json").read_text()))
    public_wave = PUBLIC[0]["waves"][0]
    for wave in payload["cases"][0]["waves"]:
        wave.update({key: public_wave[key] for key in ("layout", "nl", "nh", "hs")})
    excluded_shapes = frozenset(verifier.shape_identity(w) for c in PUBLIC for w in c["waves"])
    with pytest.raises(RuntimeError):
        verifier.validate_final_manifest(payload, excluded_shapes, frozenset())
