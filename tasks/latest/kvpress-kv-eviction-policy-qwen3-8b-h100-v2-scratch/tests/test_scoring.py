from __future__ import annotations

import math

import pytest

import kv_eviction_runtime as rt


def _rows(fid_by_family: dict[str, float], *, select: float = 0.01, prefill: float = 0.1) -> list[dict]:
    rows = []
    for family, fid in fid_by_family.items():
        for ratio in rt.RATIOS:
            rows.append({"item_id": f"{family}-1", "family": family, "task": f"t.{family}", "ratio": ratio, "fid": fid, "n_tokens": 32,
                         "select_time_sum": select, "prefill_s": prefill, "indices_ok": True, "immutable_ok": True, "rng_ok": True,
                         "budgets_ok": True, "memory_delta": 0})
    return rows


def test_constants() -> None:
    assert rt.RATIOS == (0.5, 0.75, 0.875)
    assert rt.FAMILIES == ("retrieval", "aggregation", "qa", "generation")
    assert (rt.LATENCY_FRACTION, rt.LATENCY_ITEM_CAP, rt.MEMORY_LIMIT_GIB, rt.R_MIN, rt.R_MAX) == (0.30, 0.60, 8, 16, 48)


def test_geometric_mean_over_four_families() -> None:
    summary = rt.aggregate(_rows({"retrieval": 0.8, "aggregation": 0.8, "qa": 0.1, "generation": 0.8}))
    expected = (0.8 * 0.8 * 0.1 * 0.8) ** 0.25
    assert all(math.isclose(v, expected) for v in summary["ratio_score"].values())
    assert math.isclose(summary["primary"], 100 * expected)
    assert set(summary["family_score"]) == set(rt.FAMILIES)


def test_latency_gate_suite_and_item() -> None:
    assert rt.aggregate(_rows(dict.fromkeys(rt.FAMILIES, 0.9), select=0.03))["gates"]["latency_ok"] is True
    assert rt.aggregate(_rows(dict.fromkeys(rt.FAMILIES, 0.9), select=0.031))["gates"]["latency_ok"] is False
    rows = _rows(dict.fromkeys(rt.FAMILIES, 0.9), select=0.0)
    rows[0]["select_time_sum"] = 0.061  # one item over 60% of its prefill
    assert rt.aggregate(rows)["gates"]["latency_ok"] is False


@pytest.mark.parametrize("bad", (0.0, -0.1, 1.5, float("nan")))
def test_fidelity_must_be_in_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError):
        rt.aggregate(_rows({"retrieval": bad, "aggregation": 0.8, "qa": 0.8, "generation": 0.8}))


def test_task_breakdown() -> None:
    breakdown = rt.task_breakdown(_rows({"qa": 0.5, "generation": 0.25}))
    assert breakdown == {"t.generation": {str(r): 0.25 for r in rt.RATIOS}, "t.qa": {str(r): 0.5 for r in rt.RATIOS}}
