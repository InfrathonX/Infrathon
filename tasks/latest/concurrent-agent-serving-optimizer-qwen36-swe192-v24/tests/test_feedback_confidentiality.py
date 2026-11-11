from __future__ import annotations

import json


RAW_NONCE = "SYNTHETIC_PRIVATE_NONCE_SENTINEL_DO_NOT_DISCLOSE"
PRIVATE_BASELINE_PATH = "/home/evaluator/.cache/private/hidden-direct-oracle.json"
HIDDEN_TASK_ID = "django__django-11099"


def _serialized(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def test_structured_feedback_reads_only_sanitized_summary(
    judge_module,
    valid_baseline,
    valid_candidate,
    valid_evaluation,
) -> None:
    hard_gate = judge_module.canonical_hard_gate_result(
        valid_baseline,
        valid_candidate,
        valid_evaluation,
    )
    raw_report = {
        "status": "complete",
        "candidate_dir": "/host/private/candidate-eval-123",
        "hidden_workload_nonce": RAW_NONCE,
        "baseline_evidence": {
            "path": PRIVATE_BASELINE_PATH,
            "workload_nonce_sha256": "a" * 64,
        },
        "leg": {
            **valid_candidate,
            "workload_nonce": RAW_NONCE,
            "workflows": [
                {
                    "task_id": HIDDEN_TASK_ID,
                    "run_id": "private-run-id",
                    "messages": ["private prompt"],
                }
            ],
        },
        "evaluation": valid_evaluation,
        "mlsysforge_evaluation": hard_gate,
    }

    result = judge_module.structured_result(raw_report)
    rendered = _serialized(result)

    assert result["valid"] is True
    assert result["score"] == 80.0
    for forbidden in (
        RAW_NONCE,
        PRIVATE_BASELINE_PATH,
        HIDDEN_TASK_ID,
        "/host/private/candidate-eval-123",
        "private-run-id",
        "private prompt",
    ):
        assert forbidden not in rendered
    for forbidden_key in (
        "candidate_dir",
        "baseline_evidence",
        "workflows",
        "task_id",
        "run_id",
        "workload_nonce",
    ):
        assert f'"{forbidden_key}"' not in rendered


def test_invalid_reasons_are_allowlisted_not_copied_from_untrusted_report(
    judge_module,
    valid_baseline,
    valid_candidate,
    valid_evaluation,
) -> None:
    valid_evaluation["valid"] = False
    valid_evaluation["all_invalid_reasons"] = [
        f"controller_error nonce={RAW_NONCE} path={PRIVATE_BASELINE_PATH}"
    ]
    hard_gate = judge_module.canonical_hard_gate_result(
        valid_baseline,
        valid_candidate,
        valid_evaluation,
    )
    result = judge_module.structured_result(
        {
            "status": "complete",
            "mlsysforge_evaluation": hard_gate,
        }
    )
    rendered = _serialized(result)

    assert result["valid"] is False
    assert RAW_NONCE not in rendered
    assert PRIVATE_BASELINE_PATH not in rendered
    assert len(result["summary"]) <= 700
    assert all(len(str(item.get("message", ""))) <= 700 for item in result["details"])


def test_infrastructure_failure_never_echoes_exception_secrets(judge_module) -> None:
    unsafe = (
        f"runner failed with hidden nonce {RAW_NONCE}; "
        f"baseline={PRIVATE_BASELINE_PATH}; task={HIDDEN_TASK_ID}"
    )
    result = judge_module.infrastructure_failure(unsafe)
    rendered = _serialized(result)

    assert result["valid"] is False
    assert result["score"] in (0, 0.0, None)
    assert RAW_NONCE not in rendered
    assert PRIVATE_BASELINE_PATH not in rendered
    assert HIDDEN_TASK_ID not in rendered
    assert "infrastructure" in rendered.lower() or "judge" in rendered.lower()
