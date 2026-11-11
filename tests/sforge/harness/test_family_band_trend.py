from __future__ import annotations

from copy import deepcopy

from sforge.harness.judge_server import (
    _add_family_band_trends,
    _safe_valid_family_bands,
)


def _report(large_k: float, large_n: float, *, valid: bool = True) -> dict:
    def entry(lower: float) -> dict:
        return {
            "status": "passed",
            "warmed_latency_band_ms": {
                "lower_ms": lower,
                "upper_ms": lower * 1.25,
            },
        }

    return {
        "valid": valid,
        "metrics": {
            "deployment_family_feedback": {
                "large-K": entry(large_k),
                "large-N/high-D": entry(large_n),
            }
        },
    }


def _history_entry(
    large_k: float,
    large_n: float,
    *,
    round_id: str = "agent-1",
    valid: bool = True,
) -> dict:
    report = _report(large_k, large_n, valid=valid)
    return {
        "type": "submission",
        "status": "completed",
        "round": round_id,
        "valid": valid,
        "deployment_family_feedback": _safe_valid_family_bands(report),
    }


def test_first_valid_manual_result_has_no_fabricated_trend() -> None:
    report = _report(100.0, 200.0)
    _add_family_band_trends(report, [])
    for entry in report["metrics"]["deployment_family_feedback"].values():
        assert "trend_vs_best_valid_band" not in entry


def test_trend_compares_each_family_to_best_prior_valid_manual_band() -> None:
    report = _report(80.0, 250.0)
    history = [
        _history_entry(100.0, 200.0),
        _history_entry(90.0, 300.0, round_id="agent-2"),
        _history_entry(1.0, 1.0, round_id="auto-1"),
        _history_entry(1.0, 1.0, round_id="agent-3", valid=False),
    ]
    _add_family_band_trends(report, history)
    feedback = report["metrics"]["deployment_family_feedback"]
    assert feedback["large-K"]["trend_vs_best_valid_band"] == "improved"
    assert (
        feedback["large-N/high-D"]["trend_vs_best_valid_band"]
        == "regressed"
    )


def test_equal_public_band_is_unchanged_and_invalid_never_gets_trend() -> None:
    history = [_history_entry(100.0, 200.0)]
    equal = _report(100.0, 200.0)
    _add_family_band_trends(equal, history)
    for entry in equal["metrics"]["deployment_family_feedback"].values():
        assert entry["trend_vs_best_valid_band"] == "unchanged"

    invalid = _report(50.0, 50.0, valid=False)
    before = deepcopy(invalid)
    _add_family_band_trends(invalid, history)
    assert invalid == before


def test_malformed_or_extra_family_feedback_fails_closed() -> None:
    report = _report(100.0, 200.0)
    report["metrics"]["deployment_family_feedback"]["secret"] = {
        "status": "passed"
    }
    assert _safe_valid_family_bands(report) is None

    malformed = _report(100.0, 200.0)
    malformed["metrics"]["deployment_family_feedback"]["large-K"][
        "warmed_latency_band_ms"
    ]["upper_ms"] = 126.0
    assert _safe_valid_family_bands(malformed) is None
