from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import TASK_ROOT, load_module


adapter = load_module("kv_eviction_v2_judge_adapter", "images/judge/run_judge.py")


def valid_records() -> tuple[dict[str, float], dict[str, Any]]:
    ratios = {"ratio_050": 0.8, "ratio_075": 0.7, "ratio_0875": 0.6}
    families = {"family_retrieval": 0.7, "family_aggregation": 0.75, "family_qa": 0.68, "family_generation": 0.72}
    score = 100.0 * sum(ratios.values()) / 3
    record = {
        "schema_version": 2.0,
        "item_count": 5000.0,
        "reward": score,
        "score_fidelity_auc": score,
        **ratios,
        **families,
        "latency_ratio_suite": 0.12,
        "latency_ratio_item_max": 0.31,
        "judge_gpu_count": 8.0,
        "profile_ok": 1.0,
        "suite_ok": 1.0,
        "source_ok": 1.0,
        "hardware_ok": 1.0,
        "indices_ok": 1.0,
        "budgets_ok": 1.0,
        "immutable_ok": 1.0,
        "rng_ok": 1.0,
        "latency_ok": 1.0,
        "memory_ok": 1.0,
        "fidelity_ok": 1.0,
        "profile_code": 2.0,
        "final_version": 3.0,
        "num_candidate_versions": 4.0,
    }
    detail: dict[str, Any] = dict(record)
    detail.update(
        {
            "evaluation_profile": "final",
            "suite_id": "final-holdout-v2",
            "manifest_sha256": "a" * 64,
            # Full private detail is deliberately not copied into metrics.
            "family_fid": {"0.5": {"qa": 0.8}},
            "shards": [{"index": 0, "items": 625}],
        }
    )
    return dict(record), detail


def build(
    reward: dict[str, Any],
    detail: dict[str, Any],
    *,
    completed: bool = True,
    returncode: int | None = 0,
) -> dict[str, Any]:
    return adapter.build_structured_result(
        reward,
        detail,
        verifier_completed=completed,
        verifier_returncode=returncode,
    )


def test_valid_final_exposes_only_safe_aggregates() -> None:
    assert adapter.RATIO_KEYS == ("ratio_050", "ratio_075", "ratio_0875")
    assert set(adapter.PROFILE_IDENTITIES) == {2.0}
    reward, detail = valid_records()
    result = build(reward, detail)
    assert result["valid"] is True
    assert result["score"] == pytest.approx(70.0)
    metrics = result["metrics"]
    assert metrics["source"] == "kv_eviction_v2_verifier_adapter"
    assert metrics["suite_id"] == "final-holdout-v2" and metrics["evaluation_profile"] == "final"
    assert metrics["ratio_0875"] == 0.6 and metrics["family_generation"] == 0.72
    assert metrics["latency_ratio_suite"] == 0.12 and metrics["judge_gpu_count"] == 8.0
    # Qualification recording needs the suite size in the visible metrics.
    assert metrics["item_count"] == 5000.0
    for private in ("manifest_sha256", "family_fid", "shards", "private_rows"):
        assert private not in metrics


def test_anchor_profile_no_longer_exists() -> None:
    reward, detail = valid_records()
    for record in (reward, detail):
        record["profile_code"] = 1.0
    detail.update(evaluation_profile="anchor", suite_id="kv-eviction-anchor-v1")
    assert build(reward, detail)["valid"] is False


@pytest.mark.parametrize("gate", [name for name, _label in adapter.GATES])
def test_validity_requires_every_gate(gate: str) -> None:
    reward, detail = valid_records()
    reward[gate] = detail[gate] = 0.0
    result = build(reward, detail)
    assert result["valid"] is False
    assert result["score"] == 0.0
    assert all(row["status"] == "ERROR" for row in result["details"])


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_score",
        "wrong_item_count",
        "wrong_profile",
        "mismatched_record",
        "missing_ratio",
    ),
)
def test_aggregate_count_identity_and_record_mismatches_fail_closed(mutation: str) -> None:
    reward, detail = valid_records()
    if mutation == "wrong_score":
        reward["reward"] = reward["score_fidelity_auc"] = 35.0
        detail["reward"] = detail["score_fidelity_auc"] = 35.0
    elif mutation == "wrong_item_count":
        reward["item_count"] = detail["item_count"] = 9.0
    elif mutation == "wrong_profile":
        detail["suite_id"] = "final-holdout-v1"
    elif mutation == "mismatched_record":
        detail["ratio_050"] += 0.01
    else:
        reward.pop("ratio_0875")
        detail.pop("ratio_0875")
    result = build(reward, detail)
    assert result["valid"] is False
    assert result["score"] == 0.0
    assert result["details"][-1]["status"] == "ERROR"


def test_mismatched_reward_is_rejected() -> None:
    reward, detail = valid_records(); reward["reward"] += 1.0
    result = build(reward, detail)
    assert result["valid"] is False and result["score"] == 0.0


def test_well_formed_trusted_gate_failure_is_failed_not_error() -> None:
    reward, detail = valid_records()
    for record in (reward, detail):
        record["reward"] = 0.0
        record["score_fidelity_auc"] = 0.0
        for key in adapter.RATIO_KEYS:
            record[key] = 0.0
        record["latency_ok"] = 0.0
    result = build(reward, detail, returncode=1)
    assert result["valid"] is False
    latency = next(row for row in result["details"] if row["name"] == "latency_ok")
    assert latency["status"] == "FAILED"
    assert result["details"][-1]["status"] == "FAILED"
    assert result["metrics"]["score_fidelity_auc"] == 0.0
    assert "ratio_050" not in result["metrics"]


@pytest.mark.parametrize(
    ("reward", "detail", "completed", "returncode"),
    (
        ({}, {}, True, 1),
        ({"reward": 1.0}, {}, True, 0),
        ({}, {"schema_version": 1.0}, False, None),
    ),
)
def test_missing_or_garbled_verifier_records_are_error(
    reward: dict[str, Any],
    detail: dict[str, Any],
    completed: bool,
    returncode: int | None,
) -> None:
    result = build(reward, detail, completed=completed, returncode=returncode)
    assert result["valid"] is False
    assert result["score"] == 0.0
    assert all(row["status"] == "ERROR" for row in result["details"])


@pytest.mark.parametrize(
    "payload",
    (
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
        '{"value": 1e400}',
        '{"value": ',
    ),
)
def test_nonfinite_and_malformed_json_is_rejected(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "result.json"
    path.write_text(payload, encoding="utf-8")
    assert adapter.read_json_object(path) == {}
