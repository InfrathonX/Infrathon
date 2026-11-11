from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import TASK_ROOT, load_module


adapter = load_module("kv_scoring_v1_judge_adapter", "images/judge/run_judge.py")


def valid_records() -> tuple[dict[str, float], dict[str, Any]]:
    ratios = {"ratio_010": 0.8, "ratio_020": 0.7, "ratio_030": 0.6}
    groups = {"group_retrieval": 0.6, "group_contextual_qa": 0.8, "group_redundancy": 0.7}
    score = 100.0 * sum(ratios.values()) / 3
    record = {
        "schema_version": 2.0,
        "item_count": 1812.0,
        "question_count": 6956.0,
        "overhead_item_violations": 0.0,
        "deterministic_items": 1812.0,
        "overhead_ratio_suite": 0.1,
        "overhead_ratio_item_max": 0.4,
        "judge_gpu_count": 8.0,
        "reward": score,
        "score_rel_auc": score,
        **ratios,
        **groups,
        "profile_ok": 1.0,
        "suite_ok": 1.0,
        "source_ok": 1.0,
        "assets_ok": 1.0,
        "hardware_ok": 1.0,
        "scores_ok": 1.0,
        "metric_ok": 1.0,
        "immutable_ok": 1.0,
        "rng_ok": 1.0,
        "overhead_ok": 1.0,
        "memory_ok": 1.0,
        "reference_ok": 1.0,
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
            "task_acc": {"0.1": {"squad": 0.5}},
            "shards": [{"index": 0, "items": 227}],
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
    assert set(adapter.PROFILE_IDENTITIES) == {2.0}
    reward, detail = valid_records()
    result = build(reward, detail)
    assert result["valid"] is True
    assert result["score"] == pytest.approx(70.0)
    metrics = result["metrics"]
    assert metrics["source"] == "kvzip_kv_scoring_v2_verifier_adapter" and metrics["suite_id"] == "final-holdout-v2"
    assert metrics["group_retrieval"] == 0.6 and metrics["overhead_ratio_suite"] == 0.1 and metrics["judge_gpu_count"] == 8.0
    assert metrics["item_count"] == 1812.0 and metrics["deterministic_items"] == 1812.0
    for private in ("manifest_sha256", "task_acc", "shards", "private_rows"):
        assert private not in metrics


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
        reward["reward"] = reward["score_rel_auc"] = 35.0
        detail["reward"] = detail["score_rel_auc"] = 35.0
    elif mutation == "wrong_item_count":
        reward["item_count"] = detail["item_count"] = 9.0
    elif mutation == "wrong_profile":
        detail["suite_id"] = "final-holdout-v1"
    elif mutation == "mismatched_record":
        detail["ratio_010"] += 0.01
    else:
        reward.pop("ratio_030")
        detail.pop("ratio_030")
    result = build(reward, detail)
    assert result["valid"] is False
    assert result["score"] == 0.0
    assert result["details"][-1]["status"] == "ERROR"


def test_nondeterministic_item_count_is_telemetry_only() -> None:
    reward, detail = valid_records()
    reward["deterministic_items"] = detail["deterministic_items"] = 15.0
    result = build(reward, detail)
    assert result["valid"] is True
    assert result["metrics"]["deterministic_items"] == 15.0

def test_mismatched_reward_is_rejected() -> None:
    reward, detail = valid_records(); reward["reward"] += 1.0
    result = build(reward, detail)
    assert result["valid"] is False and result["score"] == 0.0


def test_well_formed_trusted_gate_failure_is_failed_not_error() -> None:
    reward, detail = valid_records()
    for record in (reward, detail):
        record["reward"] = 0.0
        record["score_rel_auc"] = 0.0
        for key in adapter.RATIO_KEYS:
            record[key] = 0.0
        record["overhead_ok"] = 0.0
    result = build(reward, detail, returncode=1)
    assert result["valid"] is False
    latency = next(row for row in result["details"] if row["name"] == "overhead_ok")
    assert latency["status"] == "FAILED"
    assert result["details"][-1]["status"] == "FAILED"
    assert result["metrics"]["score_rel_auc"] == 0.0
    assert "ratio_010" not in result["metrics"]


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


