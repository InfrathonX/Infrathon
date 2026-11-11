from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from conftest import TASK_ROOT, load_module


adapter = load_module("attn_qat_v1_judge_adapter", "images/judge/run_judge.py")


def valid_records(profile: str = "final") -> tuple[dict[str, float], dict[str, Any]]:
    families = {"video_tflops": 100.0, "llm_tflops": 25.0}
    score = math.exp(
        math.fsum(math.log(value) for value in families.values()) / len(families)
    )
    code, suite_id = (
        (1.0, "attn-qat-anchor-v1")
        if profile == "auto"
        else (2.0, "final-holdout-v3")
    )
    record = {
        "schema_version": 1.0,
        "case_count": 287.0,
        "edge_case_count": 15.0,
        "reward": score,
        "score_tflops": score,
        **families,
        "profile_ok": 1.0,
        "suite_ok": 1.0,
        "source_ok": 1.0,
        "worker_ok": 1.0,
        "hardware_ok": 1.0,
        "correctness_ok": 1.0,
        "edge_ok": 1.0,
        "fresh_input_ok": 1.0,
        "fresh_output_ok": 1.0,
        "saved_cap_ok": 1.0,
        "profile_code": code,
        "final_version": 3.0,
        "num_candidate_versions": 4.0,
    }
    detail: dict[str, Any] = dict(record)
    detail.update(
        {
            "evaluation_policy": adapter.EVALUATION_POLICY,
            "selection_protocol_id": adapter.SELECTION_PROTOCOL_ID,
            "evaluation_profile": profile,
            "suite_id": suite_id,
            "manifest_sha256": "a" * 64,
            # Full private detail is deliberately not copied into metrics.
            "cases": [{"case_id": "holdout-c1-q1", "trial_us": [1.0]}],
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


def test_valid_complete_record_uses_square_root_and_exposes_only_safe_aggregates() -> None:
    assert adapter.FAMILY_KEYS == ("video_tflops", "llm_tflops")
    reward, detail = valid_records()
    result = build(reward, detail)
    assert result["valid"] is True
    assert result["score"] == pytest.approx((100.0 * 25.0) ** 0.5)
    assert result["metrics"]["source"] == "attn_qat_v1_verifier_adapter"
    # Family scores stay on the Judge: the agent sees the score and the gates.
    for key in adapter.FAMILY_KEYS:
        assert key not in result["metrics"]
    assert result["metrics"]["suite_id"] == "final-holdout-v3"
    assert result["metrics"]["edge_ok"] == 1.0
    assert "manifest_sha256" not in result["metrics"]
    assert "cases" not in result["metrics"]
    assert "band_tflops" not in result["metrics"]


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
        "wrong_case_count",
        "wrong_edge_count",
        "wrong_profile",
        "mismatched_record",
        "missing_family",
    ),
)
def test_aggregate_count_identity_and_record_mismatches_fail_closed(mutation: str) -> None:
    reward, detail = valid_records()
    if mutation == "wrong_score":
        reward["reward"] = reward["score_tflops"] = 35.0
        detail["reward"] = detail["score_tflops"] = 35.0
    elif mutation == "wrong_case_count":
        reward["case_count"] = detail["case_count"] = 9.0
    elif mutation == "wrong_edge_count":
        reward["edge_case_count"] = detail["edge_case_count"] = 9.0
    elif mutation == "wrong_profile":
        detail["suite_id"] = "attn-qat-anchor-v1"
    elif mutation == "mismatched_record":
        detail["video_tflops"] += 1.0
    else:
        reward.pop("llm_tflops")
        detail.pop("llm_tflops")
    result = build(reward, detail)
    assert result["valid"] is False
    assert result["score"] == 0.0
    assert result["details"][-1]["status"] == "ERROR"


def test_final_profile_identity_is_accepted() -> None:
    reward, detail = valid_records("final")
    result = build(reward, detail)
    assert result["valid"] is True
    assert result["metrics"]["evaluation_profile"] == "final"
    assert result["metrics"]["suite_id"] == "final-holdout-v3"


def test_well_formed_trusted_gate_failure_is_failed_not_error() -> None:
    reward, detail = valid_records()
    for record in (reward, detail):
        record["reward"] = 0.0
        record["score_tflops"] = 0.0
        for key in adapter.FAMILY_KEYS:
            record[key] = 0.0
        record["edge_ok"] = 0.0
    result = build(reward, detail, returncode=1)
    assert result["valid"] is False
    edge = next(row for row in result["details"] if row["name"] == "edge_ok")
    assert edge["status"] == "FAILED"
    assert result["details"][-1]["status"] == "FAILED"
    assert result["metrics"]["score_tflops"] == 0.0
    assert "video_tflops" not in result["metrics"]


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


def test_host_and_image_adapters_are_byte_identical() -> None:
    assert (TASK_ROOT / "images/judge/run_judge.py").read_bytes() == (
        TASK_ROOT / "judge/run_judge.py"
    ).read_bytes()


def test_missing_worker_result_is_named_by_the_worker_gate() -> None:
    # A crashed or timed-out candidate leaves no worker result.  The adapter has to
    # say so through worker_ok rather than reporting a hardware failure.
    reward, detail = valid_records()
    for record in (reward, detail):
        record["reward"] = 0.0
        record["score_tflops"] = 0.0
        for key in adapter.FAMILY_KEYS:
            record[key] = 0.0
        for key in (
            "worker_ok",
            "hardware_ok",
            "correctness_ok",
            "edge_ok",
            "fresh_input_ok",
            "fresh_output_ok",
            "saved_cap_ok",
        ):
            record[key] = 0.0
    result = build(reward, detail, returncode=1)
    assert result["valid"] is False
    worker = next(row for row in result["details"] if row["name"] == "worker_ok")
    assert "crashed or timed out" in worker["message"]
    hardware = next(row for row in result["details"] if row["name"] == "hardware_ok")
    assert hardware["message"] == "not evaluated: the candidate worker produced no result"
