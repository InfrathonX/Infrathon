"""Aggregation: geometric means, suite coverage, and refusal to invent a score."""

from __future__ import annotations

import math

import pytest

from conftest import _load

runtime = _load("mamba2_ssd_runtime", "runtime/mamba2_ssd_runtime.py")
GEOMETRY = {"batch": "mamba2_2p7b", "serving": "codestral_7b", "long": "falcon_h1_34b", "real": "granite_h_micro"}


def row(family: str, index: int, tflops: float) -> dict:
    return {"family": family, "geometry": GEOMETRY[family], "batch": 1, "seqlen": 1000 + index,
            "scored": True, "tflops": tflops, "case_id": f"{family}-{index}"}


def full_suite(value_of) -> list[dict]:
    return [
        row(family, index, value_of(family, index))
        for family, count in runtime.FINAL_FAMILY_COUNTS.items()
        for index in range(count)
    ]


def test_geometric_mean() -> None:
    assert runtime.geometric_mean([1.0, 4.0, 16.0]) == pytest.approx(4.0)
    assert runtime.geometric_mean([2.5]) == pytest.approx(2.5)


def test_geometric_mean_rejects_degenerate_input() -> None:
    for bad in ([], [0.0], [-1.0], [float("nan")], [float("inf")]):
        with pytest.raises(RuntimeError):
            runtime.geometric_mean(bad)


def test_uniform_suite_scores_its_own_value() -> None:
    summary = runtime.aggregate(full_suite(lambda f, i: 42.0))
    assert summary["score_tflops"] == pytest.approx(42.0)
    assert summary["scored_cases"] == runtime.FINAL_CASE_COUNT == 138
    assert set(summary["family_tflops"]) == set(runtime.SCORE_FAMILIES)
    assert set(summary["geometry_tflops"]) == set(GEOMETRY.values())


def test_score_is_the_geometric_mean_of_family_means() -> None:
    values = {"batch": 10.0, "serving": 40.0, "long": 90.0, "real": 20.0}
    summary = runtime.aggregate(full_suite(lambda f, i: values[f]))
    for family, value in values.items():
        assert summary["family_tflops"][family] == pytest.approx(value)
    assert summary["score_tflops"] == pytest.approx((10.0 * 40.0 * 90.0 * 20.0) ** 0.25)


def test_each_family_carries_a_quarter_whatever_its_case_count() -> None:
    """A 20% faster family moves the score by 1.2**(1/4), however many cases it has."""
    base = full_suite(lambda f, i: 50.0)
    reference = runtime.aggregate(base)["score_tflops"]
    for family in runtime.SCORE_FAMILIES:
        improved = [dict(r, tflops=r["tflops"] * 1.2) if r["family"] == family else r for r in base]
        gain = runtime.aggregate(improved)["score_tflops"] / reference
        assert gain == pytest.approx(1.2 ** 0.25)


def test_a_single_slow_family_cannot_be_averaged_away() -> None:
    balanced = runtime.aggregate(full_suite(lambda f, i: 60.0))["score_tflops"]
    lopsided = runtime.aggregate(
        full_suite(lambda f, i: 6.0 if f == "serving" else 90.0)
    )["score_tflops"]
    assert lopsided < balanced
    assert lopsided < (90.0 * 3 + 6.0) / 4.0


def test_incomplete_suite_refuses_to_score() -> None:
    rows = full_suite(lambda f, i: 30.0)[:-1]
    with pytest.raises(RuntimeError, match="family counts"):
        runtime.aggregate(rows)
    partial = runtime.aggregate(rows, require_full_suite=False)
    assert partial["scored_cases"] == 137


def test_no_scored_case_refuses_to_score() -> None:
    with pytest.raises(RuntimeError):
        runtime.aggregate([{"family": "serving", "scored": False, "tflops": 10.0}])


def test_unscored_rows_are_ignored() -> None:
    rows = full_suite(lambda f, i: 25.0)
    rows.append({"family": "edge", "geometry": "codestral_7b", "batch": 1, "seqlen": 1,
                 "scored": False, "tflops": 1.0})
    assert runtime.aggregate(rows)["scored_cases"] == 138


def test_weighted_score_matches_plain_geometric_mean_for_equal_weights() -> None:
    means = {"batch": 12.0, "serving": 33.0, "long": 71.0, "real": 50.0}
    expected = math.exp(sum(math.log(v) for v in means.values()) / 4.0)
    assert runtime.weighted_family_score(means) == pytest.approx(expected, rel=1e-12)


def test_finite_positive_guards_the_timer() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(RuntimeError):
            runtime.finite_positive(bad, "measured seconds")
    assert runtime.finite_positive(1.5, "x") == 1.5
