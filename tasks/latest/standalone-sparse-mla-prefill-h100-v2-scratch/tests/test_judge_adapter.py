from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from conftest import TASK_ROOT, load_module


adapter = load_module("sparse_mla_v2_judge_adapter", "images/judge/run_judge.py")
verifier = load_module("sparse_mla_v2_adapter_verifier", "images/judge/verify_sparse_mla.py")
GATE_NAMES = [name for name, _label in adapter.GATES]


def valid_records() -> tuple[dict[str, float], dict[str, Any]]:
    families = {"v32_tflops": 8.0, "glm5_tflops": 2.0, "v4_flash_tflops": 27.0, "v4_pro_tflops": 216.0}
    score = math.exp(sum(math.log(value) for value in families.values()) / len(families))
    record = {
        "schema_version": 3.0,
        "case_count": float(verifier.FINAL_CASE_COUNT),
        "edge_cases_total": float(verifier.EDGE_CASE_COUNT),
        "edge_cases_failed": 0.0,
        "failed_gate_code": 0.0,
        "reward": score,
        "score_tflops": score,
        **families,
        **{name: 1.0 for name in GATE_NAMES},
        "profile_code": 2.0,
        "final_version": 3.0,
        "num_candidate_versions": 4.0,
        "sanitizer_ran": 1.0,
        "sanitizer_inconclusive": 0.0,
    }
    detail: dict[str, Any] = dict(record)
    detail.update(
        {
            "evaluation_profile": "final",
            "suite_id": verifier.FINAL_SUITE_ID,
            "manifest_sha256": "a" * 64,
            "edge_manifest_sha256": "b" * 64,
            "evaluation_policy": adapter.EVALUATION_POLICY,
            "selection_protocol_id": adapter.SELECTION_PROTOCOL_ID,
            # Full private detail is deliberately not copied into metrics.
            "tier_tflops": {"v32:evidence": 20.0},
            "group_tflops": {"v32:anchor": 20.0},
            "private_runtime_result": {"cases": [{"case_id": "final-0000-anchor-v32", "trial_us": [1.0]}]},
            "sanitizer": {"status": "passed", "cases": verifier.EDGE_SANITIZE_COUNT},
        }
    )
    return dict(record), detail


def build(reward, detail, returncode: int | None = 0, completed: bool = True) -> dict[str, Any]:
    return adapter.build_structured_result(
        reward, detail, verifier_completed=completed, verifier_returncode=returncode
    )


def gate_failure(gate: str, *, edge_failed: float = 0.0) -> tuple[dict[str, Any], dict[str, Any]]:
    reward, detail = valid_records()
    for record in (reward, detail):
        record.update(reward=0.0, score_tflops=0.0, **{key: 0.0 for key in adapter.FAMILY_KEYS})
        for name in GATE_NAMES[GATE_NAMES.index(gate):]:
            record[name] = 0.0
        record["failed_gate_code"] = float(GATE_NAMES.index(gate) + 1)
        record["edge_cases_failed"] = edge_failed
    return reward, detail


def test_gate_table_is_the_v2_gate_set() -> None:
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
    assert tuple(GATE_NAMES) == verifier.GATE_NAMES
    assert adapter.SCORED_CASE_COUNT == float(verifier.FINAL_CASE_COUNT)
    assert adapter.EDGE_CASE_COUNT == float(verifier.EDGE_CASE_COUNT)
    assert adapter.SCHEMA_VERSION == float(verifier.SCHEMA_VERSION)
    assert adapter.FAMILY_KEYS == ("v32_tflops", "glm5_tflops", "v4_flash_tflops", "v4_pro_tflops")
    assert adapter.SCORED_CASE_COUNT == 997.0 and adapter.EDGE_CASE_COUNT == 470.0
    assert adapter.PROFILE_IDENTITIES == {2.0: ("final", "final-holdout-v4")}
    assert tuple(f"{family}_tflops" for family in verifier.FAMILIES) == adapter.FAMILY_KEYS
    assert "compute capability >= 8.0, >= 32 GiB" in dict(adapter.GATES)["hardware_ok"]


def test_valid_complete_records_expose_only_safe_family_aggregates() -> None:
    reward, detail = valid_records()
    result = build(reward, detail)
    assert result["valid"] is True
    assert result["score"] == pytest.approx(math.sqrt(8.0 * 2.0 * 27.0 * 216.0) ** 0.5)
    metrics = result["metrics"]
    assert metrics["source"] == "standalone_sparse_mla_prefill_v2_verifier_adapter"
    assert metrics["v32_tflops"] == 8.0 and metrics["v4_flash_tflops"] == 27.0
    assert metrics["glm5_tflops"] == 2.0
    assert metrics["v4_pro_tflops"] == 216.0 and metrics["sanitizer_inconclusive"] == 0.0
    assert metrics["suite_id"] == "final-holdout-v4"
    assert metrics["edge_cases_failed"] == 0.0
    assert metrics["edge_cases_total"] == float(verifier.EDGE_CASE_COUNT)
    assert metrics["sanitizer_ran"] == 1.0
    for private in ("manifest_sha256", "edge_manifest_sha256", "tier_tflops", "group_tflops",
                    "private_runtime_result", "sanitizer", "cases"):
        assert private not in metrics
    assert [row["name"] for row in result["details"]] == [*GATE_NAMES, "performance"]
    assert all(row["status"] == "PASSED" for row in result["details"])


@pytest.mark.parametrize("field, value", [
    ("suite_id", "final-holdout-v1"),
    ("evaluation_policy", "full-final-shared-v2"),
    ("selection_protocol_id", "sparse-mla-shared-final-v2"),
    ("evaluation_profile", "auto"),
    ("manifest_sha256", "A" * 64),
    ("edge_manifest_sha256", "short"),
])
def test_final_profile_identity_mismatch_fails_closed(field: str, value: str) -> None:
    reward, detail = valid_records()
    detail[field] = value
    rejected = build(reward, detail)
    assert rejected["valid"] is False
    assert rejected["score"] == 0.0


@pytest.mark.parametrize("field, value", [
    ("case_count", 10.0),
    ("edge_cases_total", float(verifier.EDGE_CASE_COUNT - 1)),
    ("edge_cases_failed", 1.0),
    ("schema_version", 1.0),
    ("profile_code", 1.0),
    ("sanitizer_ran", 0.5),
    ("sanitizer_inconclusive", 0.5),
    ("final_version", 0.0),
])
def test_incomplete_or_foreign_records_are_never_valid(field: str, value: float) -> None:
    reward, detail = valid_records()
    reward[field] = value
    detail[field] = value
    assert build(reward, detail)["valid"] is False


@pytest.mark.parametrize("mutation", ["wrong_score", "mismatched_record", "missing_family"])
def test_aggregate_and_record_mismatches_fail_closed(mutation: str) -> None:
    reward, detail = valid_records()
    if mutation == "wrong_score":
        for record in (reward, detail):
            record["reward"] = record["score_tflops"] = 35.0
    elif mutation == "mismatched_record":
        detail["v32_tflops"] += 1.0
    else:
        reward.pop("v4_pro_tflops")
        detail.pop("v4_pro_tflops")
    result = build(reward, detail)
    assert result["valid"] is False
    assert result["score"] == 0.0
    assert all(row["status"] == "ERROR" for row in result["details"])


@pytest.mark.parametrize("gate", GATE_NAMES)
def test_validity_requires_every_gate(gate: str) -> None:
    reward, detail = valid_records()
    reward[gate] = detail[gate] = 0.0
    result = build(reward, detail)
    assert result["valid"] is False
    assert result["score"] == 0.0
    assert all(row["status"] == "ERROR" for row in result["details"])


@pytest.mark.parametrize("gate", ["correctness_ok", "edge_correctness_ok", "memory_safety_ok", "harness_integrity_ok"])
def test_well_formed_hard_gate_failure_reports_the_failing_gate(gate: str) -> None:
    edge_failed = 7.0 if gate == "edge_correctness_ok" else 0.0
    reward, detail = gate_failure(gate, edge_failed=edge_failed)
    result = build(reward, detail, returncode=1)
    assert result["valid"] is False
    statuses = {row["name"]: row["status"] for row in result["details"][:-1]}
    assert [name for name, status in statuses.items() if status == "FAILED"] == [gate]
    assert all(status in ("PASSED", "FAILED", "SKIPPED") for status in statuses.values())
    assert result["details"][-1]["status"] == "FAILED"
    metrics = result["metrics"]
    assert metrics["score_tflops"] == 0.0
    assert metrics["failed_gate"] == gate
    assert not any(key in metrics for key in adapter.FAMILY_KEYS)
    assert gate in result["summary"]
    assert metrics["edge_cases_total"] == float(verifier.EDGE_CASE_COUNT)
    assert metrics["edge_cases_failed"] == edge_failed
    if gate == "edge_correctness_ok":
        assert f"7 of {verifier.EDGE_CASE_COUNT} private edge cases failed" in result["summary"]


def test_gates_after_the_first_failure_are_reported_as_skipped() -> None:
    reward, detail = gate_failure("source_ok")
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
    # An inconclusive record is impossible without a sanitizer pass.
    reward["sanitizer_ran"] = detail["sanitizer_ran"] = 0.0
    reward["sanitizer_inconclusive"] = detail["sanitizer_inconclusive"] = 1.0
    assert build(reward, detail)["valid"] is False


def test_an_inconclusive_sanitizer_pass_is_valid_and_says_so() -> None:
    reward, detail = valid_records()
    reward["sanitizer_inconclusive"] = detail["sanitizer_inconclusive"] = 1.0
    result = build(reward, detail)
    assert result["valid"] is True
    assert result["metrics"]["sanitizer_inconclusive"] == 1.0
    assert "inconclusive" in result["summary"]
    memory = next(item for item in result["details"] if item["name"] == "memory_safety_ok")
    assert memory["status"] == "PASSED" and "inconclusive" in memory["message"]


@pytest.mark.parametrize(
    ("reward", "detail", "completed", "returncode"),
    (
        ({}, {}, True, 1),
        ({"reward": 1.0}, {}, True, 0),
        ({}, {"schema_version": 3.0}, False, None),
    ),
)
def test_missing_or_garbled_verifier_records_are_error(reward, detail, completed, returncode) -> None:
    result = build(reward, detail, returncode=returncode, completed=completed)
    assert result["valid"] is False
    assert result["score"] == 0.0
    assert all(row["status"] == "ERROR" for row in result["details"])


@pytest.mark.parametrize(
    "payload",
    ('{"value": NaN}', '{"value": Infinity}', '{"value": -Infinity}', '{"value": 1e400}', '{"value": '),
)
def test_nonfinite_and_malformed_json_is_rejected(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "result.json"
    path.write_text(payload, encoding="utf-8")
    assert adapter.read_json_object(path) == {}


def test_verifier_files_round_trip_through_the_adapter(tmp_path: Path) -> None:
    """The verifier's own result files, read back by the adapter, on both outcomes."""
    _reward, detail = valid_records()
    passing = verifier.failure_detail("", profile="final", suite_id=verifier.FINAL_SUITE_ID)
    passing.update({key: value for key, value in detail.items() if key != "reward"})
    verifier.write_result(tmp_path / "pass", passing)
    result = build(
        adapter.read_json_object(tmp_path / "pass/reward.json"),
        adapter.read_json_object(tmp_path / "pass" / adapter.DETAIL_NAME),
    )
    assert result["valid"] is True and result["score"] == pytest.approx(detail["score_tflops"])
    assert (tmp_path / "pass/reward.txt").read_text() == f"{passing['score_tflops']}\n"

    failing = verifier.failure_detail("evaluation did not complete", profile="final",
                                      suite_id=verifier.FINAL_SUITE_ID, final_version=2, versions=2)
    failing.update(profile_ok=1.0, suite_ok=1.0, source_ok=1.0, hardware_ok=1.0, sanitizer_ran=1.0,
                   edge_cases_total=float(verifier.EDGE_CASE_COUNT), edge_cases_failed=3.0)
    verifier._mark_failed_gate(failing, "edge_correctness_ok")
    verifier.write_result(tmp_path / "fail", failing)
    result = build(
        adapter.read_json_object(tmp_path / "fail/reward.json"),
        adapter.read_json_object(tmp_path / "fail" / adapter.DETAIL_NAME),
        returncode=1,
    )
    statuses = {row["name"]: row["status"] for row in result["details"]}
    assert statuses["hardware_ok"] == "PASSED"
    assert statuses["edge_correctness_ok"] == "FAILED"
    assert statuses["correctness_ok"] == statuses["memory_safety_ok"] == "SKIPPED"
    assert result["metrics"]["edge_cases_failed"] == 3.0
    assert "evaluation did not complete" not in str(result)


def test_host_and_image_adapters_are_byte_identical() -> None:
    assert (TASK_ROOT / "images/judge/run_judge.py").read_bytes() == (
        TASK_ROOT / "judge/run_judge.py"
    ).read_bytes()
