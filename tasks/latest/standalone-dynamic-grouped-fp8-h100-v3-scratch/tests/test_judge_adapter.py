from __future__ import annotations

import math
from typing import Any

import pytest

from conftest import load_module


adapter = load_module("standalone_v4_judge_adapter", "images/judge/run_judge.py")
GATE_NAMES = [name for name, _label in adapter.GATES]


def valid_records(profile: str = "final") -> tuple[dict[str, float], dict[str, Any]]:
    families = {"prefill_tflops": 650.0, "decode_tflops": 120.0, "latency_tflops": 16.0, "sequence_tflops": 290.0}
    score = math.exp(sum(math.log(value) for value in families.values()) / 4)
    code, suite_id = (1.0, "unused-auto-profile") if profile == "auto" else (2.0, "final-holdout-v4")
    record = {
        "schema_version": 4.0,
        "case_count": 1000.0,
        "edge_cases_total": 188.0,
        "edge_cases_failed": 0.0,
        "failed_gate_code": 0.0,
        "reward": score,
        "score_tflops": score,
        **families,
        **{name: 1.0 for name in GATE_NAMES},
        "profile_code": code,
        "final_version": 3.0,
        "num_candidate_versions": 4.0,
        "sanitizer_ran": 1.0,
    }
    detail: dict[str, Any] = dict(record)
    detail.update(
        {
            "evaluation_profile": profile,
            "suite_id": suite_id,
            "manifest_sha256": "a" * 64,
            "edge_manifest_sha256": "b" * 64,
            "evaluation_policy": adapter.EVALUATION_POLICY,
            "selection_protocol_id": adapter.SELECTION_PROTOCOL_ID,
        }
    )
    return dict(record), detail


def build(reward, detail, returncode: int = 0) -> dict[str, Any]:
    return adapter.build_structured_result(
        reward, detail, verifier_completed=True, verifier_returncode=returncode
    )


def test_gate_table_is_the_v4_gate_set() -> None:
    assert GATE_NAMES == [
        "profile_ok",
        "suite_ok",
        "source_ok",
        "hardware_ok",
        "edge_correctness_ok",
        "correctness_ok",
        "memory_safety_ok",
        "harness_integrity_ok",
    ]
    verifier = load_module("adapter_test_verifier", "images/judge/verify_grouped.py")
    assert tuple(GATE_NAMES) == verifier.GATE_NAMES
    assert adapter.SCORED_CASE_COUNT == float(verifier.FINAL_CASE_COUNT)
    assert adapter.FAMILY_KEYS == tuple(f"{family}_tflops" for family in verifier.FAMILIES)
    assert adapter.EDGE_CASE_COUNT == float(verifier.EDGE_CASE_COUNT)


def test_valid_complete_records_expose_only_safe_family_aggregates() -> None:
    reward, detail = valid_records()
    result = build(reward, detail)
    assert result["valid"] is True
    assert result["score"] == reward["reward"]
    assert result["metrics"]["source"] == "standalone_grouped_fp8_v4_verifier_adapter"
    assert result["metrics"]["decode_tflops"] == 120.0
    assert result["metrics"]["sequence_tflops"] == 290.0
    assert "masked_tflops" not in result["metrics"]
    assert result["metrics"]["suite_id"] == "final-holdout-v4"
    assert "manifest_sha256" not in result["metrics"]
    assert "edge_manifest_sha256" not in result["metrics"]
    assert [row["name"] for row in result["details"]] == [*GATE_NAMES, "performance"]


def test_final_profile_identity_is_accepted_but_mismatch_fails_closed() -> None:
    reward, detail = valid_records("final")
    assert build(reward, detail)["valid"] is True
    for field, value in (
        ("suite_id", "final-holdout-v1"),
        ("evaluation_policy", "full-final-shared-v2"),
        ("edge_manifest_sha256", "short"),
    ):
        _, mutated = valid_records("final")
        mutated[field] = value
        rejected = build(reward, mutated)
        assert rejected["valid"] is False, field
        assert rejected["score"] == 0.0


@pytest.mark.parametrize("field, value", [("case_count", 32.0), ("edge_cases_total", 158.0), ("edge_cases_failed", 1.0)])
def test_incomplete_suites_are_never_valid(field: str, value: float) -> None:
    reward, detail = valid_records()
    reward[field] = value
    detail[field] = value
    assert build(reward, detail)["valid"] is False


def test_mismatched_or_nonfinite_records_fail_closed() -> None:
    reward, detail = valid_records()
    detail["score_tflops"] += 1.0
    result = build(reward, detail)
    assert result["valid"] is False
    assert result["score"] == 0.0
    assert all(row["status"] == "ERROR" for row in result["details"][:-1])


@pytest.mark.parametrize("gate", ["edge_correctness_ok", "memory_safety_ok", "harness_integrity_ok", "correctness_ok"])
def test_well_formed_hard_gate_failure_reports_the_failing_gate(gate: str) -> None:
    reward, detail = valid_records()
    for record in (reward, detail):
        record["reward"] = 0.0
        record["score_tflops"] = 0.0
        for key in adapter.FAMILY_KEYS:
            record[key] = 0.0
        record[gate] = 0.0
        record["failed_gate_code"] = float(GATE_NAMES.index(gate) + 1)
        record["edge_cases_failed"] = 7.0 if gate == "edge_correctness_ok" else 0.0
    result = build(reward, detail, returncode=1)
    assert result["valid"] is False
    row = next(row for row in result["details"] if row["name"] == gate)
    assert row["status"] == "FAILED"
    statuses = {row["name"]: row["status"] for row in result["details"][:-1]}
    assert [name for name, status in statuses.items() if status == "FAILED"] == [gate]
    assert all(status in ("PASSED", "FAILED") for status in statuses.values())
    assert result["metrics"]["score_tflops"] == 0.0
    assert result["metrics"]["failed_gate"] == gate
    assert "decode_tflops" not in result["metrics"]
    assert gate in result["summary"]
    if gate == "edge_correctness_ok":
        assert result["metrics"]["edge_cases_failed"] == 7.0
        assert "7 of 188 private edge cases failed" in result["summary"]


def test_host_and_image_adapters_are_byte_identical() -> None:
    image = adapter.Path(adapter.__file__).read_bytes()
    host = (adapter.Path(adapter.__file__).parents[2] / "judge/run_judge.py").read_bytes()
    assert image == host


def test_gates_after_the_first_failure_are_reported_as_skipped() -> None:
    reward, detail = valid_records()
    for record in (reward, detail):
        record.update(reward=0.0, score_tflops=0.0, **{key: 0.0 for key in adapter.FAMILY_KEYS})
        for name in ("source_ok", "hardware_ok", "edge_correctness_ok", "correctness_ok",
                     "memory_safety_ok", "harness_integrity_ok"):
            record[name] = 0.0
        record["failed_gate_code"] = float(GATE_NAMES.index("source_ok") + 1)
    result = build(reward, detail, returncode=1)
    statuses = {row["name"]: row["status"] for row in result["details"]}
    assert statuses["profile_ok"] == statuses["suite_ok"] == "PASSED"
    assert statuses["source_ok"] == "FAILED"
    assert all(statuses[name] == "SKIPPED" for name in GATE_NAMES[3:])
    assert statuses["performance"] == "FAILED"


def test_a_result_without_the_sanitizer_record_is_never_valid() -> None:
    reward, detail = valid_records()
    del reward["sanitizer_ran"], detail["sanitizer_ran"]
    assert build(reward, detail)["valid"] is False


def test_a_deployment_that_skips_the_sanitizer_is_valid_and_says_so() -> None:
    reward, detail = valid_records()
    reward["sanitizer_ran"] = detail["sanitizer_ran"] = 0.0
    result = build(reward, detail)
    assert result["valid"] is True
    assert result["metrics"]["sanitizer_ran"] == 0.0
    assert "compute-sanitizer pass not run" in result["summary"]
    memory = next(item for item in result["details"] if item["name"] == "memory_safety_ok")
    assert memory["status"] == "PASSED" and "not run in this deployment" in memory["message"]
    reward["sanitizer_ran"] = detail["sanitizer_ran"] = 1.0
    ran = build(reward, detail)
    assert ran["metrics"]["sanitizer_ran"] == 1.0 and "not run" not in ran["summary"]
